#!/usr/bin/env python3
"""
A/B test: ALIGN+SFace  vs  BBOX-CROP+SFace
Extended: save ROC curves and similarity histograms for positive/negative pairs.

Usage:
  python ab_test_sface.py --data_dir ./dataset \
                         --det_model face_detection_yunet_2023mar.onnx \
                         --fr_model face_recognition_sface_2021dec.onnx \
                         --det_input_size 320 320 \
                         --limit_per_person 10 \
                         --out_dir ./results

Requirements:
  pip install opencv-contrib-python numpy scikit-learn tqdm matplotlib
  (Or OpenCV built with contrib + dnn + face modules)

This is an enhanced version of the previous script: it will
- compute pairwise verification scores for both pipelines (ALIGN and BBOX),
- save CSV with all pair scores,
- plot ROC curve (and save PNG),
- plot histogram of similarities for positive vs negative pairs (and save PNG).
"""
import os
import argparse
import cv2 as cv
import numpy as np
from itertools import combinations
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import csv

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='Dataset root, subfolders per identity')
    p.add_argument('--det_model', required=True, help='Path to face detector ONNX (YuNet)')
    p.add_argument('--fr_model', required=True, help='Path to face recognition ONNX (SFace)')
    p.add_argument('--det_input_size', nargs=2, type=int, default=[320,320], help='Detector input size (w h)')
    p.add_argument('--score_threshold', type=float, default=0.9)
    p.add_argument('--limit_per_person', type=int, default=50)
    p.add_argument('--backend', type=int, default=0)
    p.add_argument('--target', type=int, default=0)
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--out_dir', type=str, default='./results', help='Directory to save plots/CSVs')
    return p.parse_args()

def load_models(det_model, fr_model, det_input_size, backend=0, target=0, score_threshold=0.9):
    # Face detector (YuNet)
    if not hasattr(cv, 'FaceDetectorYN') and not hasattr(cv, 'face'):
        raise RuntimeError("OpenCV doesn't have FaceDetectorYN/face module. Install opencv-contrib-python or build OpenCV with contrib.")
    try:
        fd = cv.FaceDetectorYN.create(det_model, "", tuple(det_input_size), score_threshold, 0.3, 5000)
    except Exception:
        # fallback to constructor-like API
        fd = cv.FaceDetectorYN(det_model, "", tuple(det_input_size), score_threshold, 0.3, 5000)

    # Face recognizer (SFace)
    try:
        fr = cv.FaceRecognizerSF.create(fr_model, "", backend, target)
    except Exception:
        fr = cv.face.FaceRecognizerSF.create(fr_model, "", backend, target)

    # set preferable backend/target if possible
    try:
        fd.setPreferableBackend(backend); fd.setPreferableTarget(target)
    except Exception:
        pass
    try:
        fr.setPreferableBackend(backend); fr.setPreferableTarget(target)
    except Exception:
        pass

    return fd, fr

def bbox_crop_resize(img, bbox, size=(112,112)):
    x, y, w, h = bbox
    x1 = int(round(x)); y1 = int(round(y)); x2 = int(round(x + w)); y2 = int(round(y + h))
    h_img, w_img = img.shape[:2]
    x1 = max(0, min(x1, w_img-1)); x2 = max(0, min(x2, w_img))
    y1 = max(0, min(y1, h_img-1)); y2 = max(0, min(y2, h_img))
    if x1 >= x2 or y1 >= y2:
        return None
    face = img[y1:y2, x1:x2]
    face = cv.resize(face, size, interpolation=cv.INTER_LINEAR)
    return face

def l2norm(v):
    v = v.reshape(-1)
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)

def extract_embeddings(fd, fr, img_paths, det_input_size=(320,320)):
    embeddings_aligned = []
    embeddings_bbox = []
    labels = []
    skipped = 0

    for label, img_path in tqdm(img_paths, desc="Extracting"):
        img = cv.imread(img_path)
        if img is None:
            skipped += 1
            continue

        try:
            fd.setInputSize((img.shape[1], img.shape[0]))
        except Exception:
            pass

        try:
            detections = fd.detect(img)
            if isinstance(detections, tuple):
                detections = detections[1]
        except Exception as e:
            print("Detector exception on", img_path, e)
            skipped += 1
            continue

        dets = np.array(detections).reshape(-1, detections.shape[-1]) if hasattr(detections, 'shape') else np.array(detections)
        if dets.size == 0:
            skipped += 1
            continue
        row = dets[0].astype(np.float32)  # take first detection if multiple

        bbox = (row[0], row[1], row[2], row[3])

        # aligned
        face_box_mat = row.reshape(1, -1).astype(np.float32)
        try:
            aligned = np.zeros((112,112,3), dtype=np.uint8)
            fr.alignCrop(img, face_box_mat, aligned)
        except Exception:
            face_box_cv = cv.UMat(face_box_mat) if hasattr(cv, 'UMat') else face_box_mat
            aligned = np.zeros((112,112,3), dtype=np.uint8)
            fr.alignCrop(img, face_box_cv, aligned)

        # bbox-crop
        crop = bbox_crop_resize(img, bbox, size=(112,112))
        if crop is None:
            skipped += 1
            continue

        # feature extraction using fr.feature (handles internal blob creation)
        try:
            feat_aligned = fr.feature(aligned)
            feat_bbox = fr.feature(crop)
            # convert to numpy if needed
            feat_aligned = np.array(feat_aligned).squeeze().astype(np.float32)
            feat_bbox = np.array(feat_bbox).squeeze().astype(np.float32)
        except Exception:
            # fallback: use underlying network if accessible
            blob_a = cv.dnn.blobFromImage(aligned, 1.0, (112,112), (0,0,0), swapRB=True, crop=False)
            blob_b = cv.dnn.blobFromImage(crop, 1.0, (112,112), (0,0,0), swapRB=True, crop=False)
            try:
                fr.getNetwork().setInput(blob_a); feat_aligned = fr.getNetwork().forward()
                fr.getNetwork().setInput(blob_b); feat_bbox = fr.getNetwork().forward()
                feat_aligned = np.squeeze(np.array(feat_aligned)).astype(np.float32)
                feat_bbox = np.squeeze(np.array(feat_bbox)).astype(np.float32)
            except Exception as e:
                print("Feature extraction fallback failed for", img_path, e)
                skipped += 1
                continue

        embeddings_aligned.append(l2norm(feat_aligned))
        embeddings_bbox.append(l2norm(feat_bbox))
        labels.append(label)

    return np.vstack(embeddings_aligned), np.vstack(embeddings_bbox), np.array(labels), skipped

def collect_image_paths(data_dir, limit_per_person=50, seed=42):
    persons = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    all_pairs = []
    rng = random.Random(seed)
    for p in persons:
        files = sorted([os.path.join(data_dir, p, f) for f in os.listdir(os.path.join(data_dir,p)) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        if len(files) == 0:
            continue
        if limit_per_person and len(files) > limit_per_person:
            files = rng.sample(files, limit_per_person)
        for f in files:
            all_pairs.append((p, f))
    return all_pairs

def prepare_pairs(labels, max_pairs=None, seed=42):
    rng = random.Random(seed)
    idx_by_label = {}
    for i, l in enumerate(labels):
        idx_by_label.setdefault(l, []).append(i)

    pos_pairs = []
    for l, idxs in idx_by_label.items():
        if len(idxs) < 2:
            continue
        for a, b in combinations(idxs, 2):
            pos_pairs.append((a, b, 1))

    neg_pairs = []
    all_idx = list(range(len(labels)))
    attempts = 0
    while len(neg_pairs) < max(len(pos_pairs), 1000) and attempts < len(pos_pairs)*10 + 100000:
        a = rng.choice(all_idx); b = rng.choice(all_idx)
        if labels[a] != labels[b] and a != b:
            neg_pairs.append((a, b, 0))
        attempts += 1

    if max_pairs:
        pos_pairs = pos_pairs[:max_pairs]
        neg_pairs = neg_pairs[:max_pairs]

    pairs = pos_pairs + neg_pairs
    random.shuffle(pairs)
    return pairs

def compute_pair_scores(embs, labels, pairs):
    y_true = []
    scores = []
    for a, b, y in pairs:
        fa = embs[a]; fb = embs[b]
        score = float(np.dot(fa, fb))  # cosine because normalized
        scores.append(score)
        y_true.append(y)
    return np.array(scores), np.array(y_true)

def compute_metrics_from_scores(scores, y_true):
    fpr, tpr, thresh = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    best_acc = 0.0
    if len(thresh) == 0:
        best_acc = 0.0
    else:
        for th in thresh:
            preds = (scores >= th).astype(int)
            acc = (preds == y_true).mean()
            if acc > best_acc:
                best_acc = acc
    return {'fpr': fpr, 'tpr': tpr, 'thresh': thresh, 'auc': roc_auc, 'eer': eer, 'best_acc': best_acc}

def save_pairs_csv(out_path, pairs, scores, labels):
    with open(out_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['idx_a', 'idx_b', 'label', 'score', 'label_a', 'label_b'])
        for (a,b,y), s in zip(pairs, scores):
            wr.writerow([a, b, y, float(s), labels[a], labels[b]])

def plot_roc(fpr, tpr, auc_val, out_png, title='ROC'):
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC={auc_val:.4f}')
    plt.plot([0,1],[0,1],'k--', alpha=0.4)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_histogram(scores, y_true, out_png, title='Similarity histogram', bins=50):
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    plt.figure(figsize=(7,5))
    plt.hist(neg, bins=bins, alpha=0.6, label=f'neg (n={len(neg)})', color='C1', density=False)
    plt.hist(pos, bins=bins, alpha=0.6, label=f'pos (n={len(pos)})', color='C0', density=False)
    plt.xlabel('Cosine similarity')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def identification_1nn(embs, labels):
    n = embs.shape[0]
    correct = 0
    sims = embs.dot(embs.T)
    for i in range(n):
        sims[i,i] = -999
        nn = np.argmax(sims[i])
        if labels[nn] == labels[i]:
            correct += 1
    return correct / n

def main():
    args = parse_args()
    np.random.seed(args.random_seed); random.seed(args.random_seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading models...")
    fd, fr = load_models(args.det_model, args.fr_model, args.det_input_size, backend=args.backend, target=args.target, score_threshold=args.score_threshold)

    print("Collecting images...")
    img_list = collect_image_paths(args.data_dir, limit_per_person=args.limit_per_person)
    if len(img_list) < 10:
        print("Warning: dataset small. Results may be noisy.")

    print("Extracting embeddings (this may take a while)...")
    embs_aligned, embs_bbox, labels, skipped = extract_embeddings(fd, fr, img_list, det_input_size=args.det_input_size)
    print(f"Done. Extracted {len(labels)} embeddings, skipped {skipped} images.")

    pairs = prepare_pairs(labels, max_pairs=5000, seed=args.random_seed)

    # ALIGN pipeline
    scores_align, y_true = compute_pair_scores(embs_aligned, labels, pairs)
    metrics_align = compute_metrics_from_scores(scores_align, y_true)
    id_acc_align = identification_1nn(embs_aligned, labels)

    # BBOX pipeline
    scores_bbox, _ = compute_pair_scores(embs_bbox, labels, pairs)
    metrics_bbox = compute_metrics_from_scores(scores_bbox, y_true)
    id_acc_bbox = identification_1nn(embs_bbox, labels)

    # Save CSVs
    save_pairs_csv(os.path.join(args.out_dir, 'pair_scores_align.csv'), pairs, scores_align, labels)
    save_pairs_csv(os.path.join(args.out_dir, 'pair_scores_bbox.csv'), pairs, scores_bbox, labels)

    # Plot ROC
    plot_roc(metrics_align['fpr'], metrics_align['tpr'], metrics_align['auc'], os.path.join(args.out_dir, 'roc_align.png'), title='ROC - ALIGN pipeline')
    plot_roc(metrics_bbox['fpr'], metrics_bbox['tpr'], metrics_bbox['auc'], os.path.join(args.out_dir, 'roc_bbox.png'), title='ROC - BBOX pipeline')

    # Plot histograms
    plot_histogram(scores_align, y_true, os.path.join(args.out_dir, 'hist_align.png'), title='Similarity Histogram - ALIGN')
    plot_histogram(scores_bbox, y_true, os.path.join(args.out_dir, 'hist_bbox.png'), title='Similarity Histogram - BBOX')

    # Print summary
    print("\nRESULTS:")
    print("ALIGN pipeline: AUC={:.4f}, EER={:.4f}, best_verif_acc={:.4f}, 1NN_id_acc={:.4f}".format(metrics_align['auc'], metrics_align['eer'], metrics_align['best_acc'], id_acc_align))
    print("BBOX  pipeline: AUC={:.4f}, EER={:.4f}, best_verif_acc={:.4f}, 1NN_id_acc={:.4f}".format(metrics_bbox['auc'], metrics_bbox['eer'], metrics_bbox['best_acc'], id_acc_bbox))

    print(f"\nSaved outputs in: {os.path.abspath(args.out_dir)}")
    print("Files:")
    for fn in ['pair_scores_align.csv','pair_scores_bbox.csv','roc_align.png','roc_bbox.png','hist_align.png','hist_bbox.png']:
        print(" -", fn)

if __name__ == '__main__':
    main()