import streamlit as st
import cv2
import numpy as np
import pandas as pd
from camera import ThreadedCamera
from user_management import UserManagement
from access_history import AccessHistory
import time
import asyncio
from datetime import datetime
from typing import Tuple
import faiss
# import csv
import json
import os
import queue
from websocket_client import WebSocketClient
import threading


# Giả sử đây là module bạn dùng, nếu tên khác hãy thay đổi
import face_recognition_pybind as FR
# FR.init_neural_networks()


st.set_page_config(page_title="Face Recognition System", layout="wide")

# Session state for WebSocket
if "ws_client" not in st.session_state:
    st.session_state.ws_client = None
if "ws_status" not in st.session_state:
    st.session_state.ws_status = "Disconnected"
if "ws_status_queue" not in st.session_state:
    st.session_state.ws_status_queue = queue.Queue()

# Session state for persistent variables
if "camera" not in st.session_state:
    st.session_state.camera = None
if "stable_start_time" not in st.session_state:
    st.session_state.stable_start_time = None
if "last_center" not in st.session_state:
    st.session_state.last_center = None
if "recognized_this_session" not in st.session_state:
    st.session_state.recognized_this_session = False

# Thêm các biến trạng thái để lưu kết quả nhận diện cuối cùng
if "last_known_box" not in st.session_state:
    st.session_state.last_known_box = None
if "last_known_text" not in st.session_state:
    st.session_state.last_known_text = ""

# Thêm counter để đếm số frame không có face
if "no_face_counter" not in st.session_state:
    st.session_state.no_face_counter = 0

if "cos_sim_thresh" not in st.session_state:
    st.session_state.cos_sim_thresh = 0.6

dims = 128  # vector dims
if "faiss_index" not in st.session_state:
    file_path = "facial_faiss_index.bin"
    if os.path.exists(file_path):
        loaded_index = faiss.read_index(file_path)
        # Check if loaded index supports add_with_ids
        if not isinstance(loaded_index, faiss.IndexIDMap):
            print(f"⚠️ Loaded index is not IndexIDMap. Rebuilding as IndexIDMap...")
            # Create new IndexIDMap
            base_index = faiss.index_factory(dims, "Flat", faiss.METRIC_INNER_PRODUCT)
            new_index = faiss.IndexIDMap(base_index)
            
            # Copy vectors from old index with sequential IDs (0, 1, 2, ...)
            if loaded_index.ntotal > 0:
                vectors = np.zeros((loaded_index.ntotal, dims), dtype='float32')
                for i in range(loaded_index.ntotal):
                    vectors[i] = loaded_index.reconstruct(int(i))
                ids = np.arange(loaded_index.ntotal, dtype='int64')
                new_index.add_with_ids(vectors, ids)
                print(f"✅ Migrated {loaded_index.ntotal} vectors to IndexIDMap")
            
            st.session_state.faiss_index = new_index
            # Save the migrated index
            faiss.write_index(new_index, file_path)
        else:
            st.session_state.faiss_index = loaded_index
        print(f"FAISS index loaded. Contains {st.session_state.faiss_index.ntotal} vectors.")
    else:
        # Create a new index with IDMap to support custom IDs
        base_index = faiss.index_factory(dims, "Flat", faiss.METRIC_INNER_PRODUCT)
        st.session_state.faiss_index = faiss.IndexIDMap(base_index)
        print("New FAISS IndexIDMap created.")

if "user_management" not in st.session_state:
    st.session_state.user_management = UserManagement()

# Load user data as a dictionary {int_id: {uuid, name, ...}}
if "user_data" not in st.session_state:
    st.session_state.user_data = st.session_state.user_management.get_all_user_data()
    print(f"User data loaded: {len(st.session_state.user_data)} users.")
if "access_history" not in st.session_state:
    st.session_state.access_history = AccessHistory()
    st.session_state.history = st.session_state.access_history.load_history()
if "last_log_status" not in st.session_state:
    st.session_state.last_log_status = "No recent activity"
if "last_log_time" not in st.session_state:
    st.session_state.last_log_time = ""
if "history_needs_refresh" not in st.session_state:
    st.session_state.history_needs_refresh = False
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = 0


# user_management = UserManagement()
# access_history = AccessHistory()
stable_threshold = 20  # pixels
stable_duration = 1.2  # seconds

def start_camera(cam_src):
    try:
        cam_index = int(cam_src)
        st.session_state.camera = ThreadedCamera(cam_index)
    except ValueError:
        st.session_state.camera = ThreadedCamera(cam_src)
    st.session_state.camera.start()

def stop_camera():
    if st.session_state.camera:
        st.session_state.camera.stop()
        st.session_state.camera = None
    stop_ws_client() # Also stop websocket client when stopping camera

def stop_ws_client():
    if st.session_state.ws_client:
        st.session_state.ws_client.stop()
        st.session_state.ws_client = None
        st.session_state.ws_status = "Disconnected"
        # Clear the queue
        while not st.session_state.ws_status_queue.empty():
            try:
                st.session_state.ws_status_queue.get_nowait()
            except queue.Empty:
                break

def reset_users():
    st.session_state.user_management.reset_users()
    st.session_state.user_data = st.session_state.user_management.get_all_user_data() # Reload empty data
    
    # Recreate the index to ensure it's an IndexIDMap
    base_index = faiss.index_factory(dims, "Flat", faiss.METRIC_INNER_PRODUCT)
    st.session_state.faiss_index = faiss.IndexIDMap(base_index)
    print("DEBUG: FAISS index has been recreated as IndexIDMap during reset.")

    faiss.write_index(st.session_state.faiss_index, "facial_faiss_index.bin")

    # Reset access history
    if hasattr(st.session_state, 'access_history'):
        st.session_state.access_history.reset_history()

    st.success("All users and access history have been reset.")
    update_history_display()

def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:  # Handle zero vectors
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def update_history_display():
    """Update history table display - xóa bảng cũ hoàn toàn và hiển thị bảng mới"""
    try:
        # Bước 1: Xóa hoàn toàn nội dung placeholder cũ (Double check)
        history_placeholder.empty()
        
        # Bước 2: Đợi một chút để đảm bảo clear hoàn toàn
        import time
        time.sleep(0.01)
        
        # Bước 3: Xóa lần nữa để đảm bảo
        history_placeholder.empty()
        
        # Bước 4: Tạo nội dung mới hoàn toàn
        with history_placeholder.container():
            st.subheader("📊 Access History")
            
            # Hiển thị trạng thái log gần nhất
            if hasattr(st.session_state, 'last_log_status') and st.session_state.last_log_status:
                col_status1, col_status2 = st.columns([3, 1])
                with col_status1:
                    st.write(f"**Last Activity:** {st.session_state.last_log_status}")
                with col_status2:
                    if hasattr(st.session_state, 'last_log_time') and st.session_state.last_log_time:
                        st.write(f"**Time:** {st.session_state.last_log_time}")
            
            # Hiển thị bảng lịch sử mới
            history_data = get_access_history()
            if history_data:
                df = pd.DataFrame(history_data)
                st.dataframe(df, use_container_width=True)
                print(f"✅ Successfully displayed history table with {len(history_data)} entries")
            else:
                st.info("No access history available.")
                print("ℹ️ No access history data available")
                
    except Exception as e:
        print(f"❌ Error updating history display: {e}")
        # Đảm bảo xóa placeholder nếu có lỗi
        history_placeholder.empty()
        history_placeholder.error(f"Error loading access history: {e}")



def register_user(name):
    if not name:
        st.warning("Please enter a name.")
        return

    feature_vectors = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # ===== REGISTRATION MODE: SINGLE FACE (Modified 2025-11-20) =====
    # Only capture and use the FIRST face for registration
    status_text.text(f"Capturing face...")
    frame = st.session_state.camera.get_frame()
    
    if frame is not None:
        print(f"DEBUG: Attempting to register user '{name}' - Single face capture.")
        
        # Pass frame to C++ backend for feature extraction
        result, facial_feature = FR.register_user(frame, frame.shape[0], frame.shape[1], frame.shape[2])
        
        if result == 0 and len(facial_feature) == 128:
            print(f"DEBUG: Successfully extracted feature.")
            feature_vectors.append(np.array(facial_feature, dtype='float32'))
        else:
            print(f"DEBUG WARNING: Failed to extract feature. Result: {result}")
    else:
        print(f"DEBUG WARNING: Failed to capture frame (frame is None).")
    
    progress_bar.progress(1.0)
    time.sleep(0.3)

    # ===== OLD METHOD: 3-FACE AVERAGING (Commented for rollback) =====
    # for i in range(3):
    #     status_text.text(f"Capturing image {i+1}/3...")
    #     frame = st.session_state.camera.get_frame()
    #     
    #     if frame is not None:
    #         print(f"DEBUG: Attempting to register user '{name}' - Image {i+1}/3.")
    #         
    #         # Pass frame to C++ backend for feature extraction
    #         result, facial_feature = FR.register_user(frame, frame.shape[0], frame.shape[1], frame.shape[2])
    #         
    #         if result == 0 and len(facial_feature) == 128:
    #             print(f"DEBUG: Successfully extracted feature {i+1}.")
    #             feature_vectors.append(np.array(facial_feature, dtype='float32'))
    #         else:
    #             print(f"DEBUG WARNING: Failed to extract feature from image {i+1}. Result: {result}")
    #     else:
    #         print(f"DEBUG WARNING: Failed to capture image {i+1} (frame is None).")
    #     
    #     progress_bar.progress((i + 1) / 3)
    #     time.sleep(0.5)

    status_text.empty()
    progress_bar.empty()

    if not feature_vectors:
        st.error("Failed to extract valid facial features. Please try again.")
        return

    print(f"DEBUG: Collected {len(feature_vectors)} valid feature vectors.")
    
    # Use single feature (first one)
    avg_feature = feature_vectors[0].reshape(1, dims)
    faiss.normalize_L2(avg_feature)
    
    # ===== ALTERNATIVE: STORE ALL 3 VECTORS (Commented, can enable if needed) =====
    # # Store all 3 vectors separately in FAISS (no averaging)
    # # This gives better matching flexibility
    # success, message, new_int_id, new_uuid = st.session_state.user_management.register_user(name)
    # if success:
    #     for idx, feature_vec in enumerate(feature_vectors):
    #         normalized_vec = feature_vec.reshape(1, dims)
    #         faiss.normalize_L2(normalized_vec)
    #         # Add with ID like: new_int_id * 10 + idx (e.g., 1 -> 10, 11, 12)
    #         vector_id = new_int_id * 10 + idx
    #         st.session_state.faiss_index.add_with_ids(normalized_vec, np.array([vector_id]).astype('int64'))
    #     faiss.write_index(st.session_state.faiss_index, "facial_faiss_index.bin")
    #     print(f"DEBUG: Stored {len(feature_vectors)} vectors for user '{name}'")
    # ===== END ALTERNATIVE =====
    
    # ===== OLD METHOD: AVERAGING (Commented for rollback) =====
    # avg_feature = np.mean(feature_vectors, axis=0).reshape(1, dims)
    # faiss.normalize_L2(avg_feature)

    # Use the new register_user method which returns IDs
    success, message, new_int_id, new_uuid = st.session_state.user_management.register_user(name)

    if success:
        print(f"DEBUG: UserManagement registration successful for '{name}'. New ID: {new_int_id}, UUID: {new_uuid}")
        # Add the vector with its new integer ID
        st.session_state.faiss_index.add_with_ids(avg_feature, np.array([new_int_id]).astype('int64'))
        faiss.write_index(st.session_state.faiss_index, "facial_faiss_index.bin")
        
        # Reload user data into session state
        st.session_state.user_data = st.session_state.user_management.get_all_user_data()
        print(f"DEBUG: User data reloaded. Total users: {len(st.session_state.user_data)}")
        st.success(f"{message} (ID: {new_int_id}, UUID: {new_uuid})")
    else:
        print(f"DEBUG ERROR: UserManagement registration failed for '{name}'. Message: {message}")
        st.error(message)

def get_access_history():
    # Force reload from file to ensure fresh data
    st.session_state.access_history.load_history()
    
    # Kiểm tra xem có phải JSON format hay text format
    if hasattr(st.session_state.access_history, 'history_data'):
        # JSON format - lấy trực tiếp từ dictionary
        history_data = st.session_state.access_history.history_data
        
        print(f"🔍 DEBUG - JSON format: {len(history_data)} users")
        
        lines = []
        for username, timestamps in history_data.items():
            if timestamps:  # Nếu user có ít nhất 1 lần access
                last_access = timestamps[-1]  # Lấy timestamp cuối cùng
                total_count = len(timestamps)   # Đếm tổng số lần access
                
                lines.append({
                    "Name": username,
                    "Last Checkin": last_access,
                    "Total Access": total_count
                })
                print(f"  ✅ Added: {username} - Last: {last_access} - Total: {total_count}")
    else:
        # Text format - fallback cho compatibility
        history = st.session_state.access_history.get_history()
        
        print(f"🔍 DEBUG - Text format: {len(history)} entries")
        
        # Nhóm theo user để đếm
        user_data = {}
        for entry in history:
            if entry and entry.strip():
                parts = entry.strip().split('\t')
                if len(parts) == 2:
                    name, timestamp = parts
                    if name not in user_data:
                        user_data[name] = []
                    user_data[name].append(timestamp)
        
        lines = []
        for username, timestamps in user_data.items():
            if timestamps:
                # Sắp xếp timestamps và lấy cái mới nhất
                timestamps.sort(reverse=True)
                last_access = timestamps[0]
                total_count = len(timestamps)
                
                lines.append({
                    "Name": username,
                    "Last Checkin": last_access,
                    "Total Access": total_count
                })
                print(f"  ✅ Added: {username} - Last: {last_access} - Total: {total_count}")
    
    # Sắp xếp theo last checkin mới nhất trước
    lines.sort(key=lambda x: x.get("Last Checkin", ""), reverse=True)
    print(f"📊 Final display: {len(lines)} users")
    return lines

# Các hàm không sử dụng nữa có thể được xóa hoặc giữ lại nếu cần
# def process_frame(): ...
# def detect_face(tmp=0): ...

def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

################# main #################

st.title("Check-in System")

# Tạo container cho history table để có thể update real-time
history_container = st.container()
history_placeholder = st.empty()  # Placeholder cho real-time update

# Check for status updates from the WebSocket thread
try:
    while not st.session_state.ws_status_queue.empty():
        latest_status = st.session_state.ws_status_queue.get_nowait()
        if latest_status == "CMD:RESET_USERS":
            reset_users()
            st.session_state.ws_status = "Users reset by remote command."
        else:
            st.session_state.ws_status = latest_status
except queue.Empty:
    pass # No new status

col1, col2 = st.columns([2, 1])

with col2:
    cam_src = st.text_input("Camera Source", value="rtsp://adminOnvif:Vision123@10.60.3.157/Streaming/Channels/102?transportmode=unicast&profile=Profile_1")
    name = st.text_input("Name")

    # WebSocket Registration
    st.subheader("WebSocket Registration")
    
    ws_col1, ws_col2 = st.columns([3, 1])
    with ws_col1:
        server_ip = st.text_input("Server IP", value="10.60.1.90")
    with ws_col2:
        server_port = st.number_input("Port", value=8005, min_value=1, max_value=65535)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Connect to WS"):
            stop_ws_client() # Stop any existing client
            uri = f"ws://{server_ip}:{server_port}"
            st.session_state.ws_client = WebSocketClient(
                uri,
                st.session_state.ws_status_queue,
                st.session_state.user_management, # Pass the whole user_management object
                st.session_state.faiss_index
            )
            st.session_state.ws_client.start()
            st.session_state.ws_status = "Connecting..."
            st.rerun()

    with btn_col2:
        if st.button("Disconnect from WS"):
            stop_ws_client()
            st.rerun()
            
    st.write(f"**WebSocket Status:** {st.session_state.ws_status}")
    
    # Face Recognition controls
    st.subheader("Face Recognition Settings")
    st.session_state.cos_sim_thresh = st.slider(
        "Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.cos_sim_thresh, 
        step=0.01,
        help="Ngưỡng tương đồng: ≥threshold = Nhận diện, <threshold = Người lạ"
    )
    
    # System Status
    st.subheader("System Status")
    st.success("✅ Face Recognition: Ready")
    
    if st.button("Start Camera"):
        stop_camera()
        start_camera(cam_src)
    if st.button("Register"):
        # if st.session_state.camera:
        if st.session_state.camera is None:
            st.error("❌ Please start the camera first before registering.")
        else:
            register_user(name)
    
    if st.button("Test Face Detection"):
        if st.session_state.camera is None:
            st.error("❌ Please start the camera first.")
        else:
            frame = st.session_state.camera.get_frame()
            if frame is not None:
                # REMOVED: C++ now handles resizing
                # frame = resize_with_pad(np.array(frame), (640, 640), padding_color=(0, 0, 0))
                # frame_rgb = frame
                try:
                    # Pass the original frame directly to the C++ function
                    left, top, right, bottom, spoof_confidence, facial_feature = FR.detect_face(frame, frame.shape[0], frame.shape[1], frame.shape[2])
                    
                    if left > 0 and top > 0 and right > 0 and bottom > 0:
                        st.success(f"✅ Face detected! Position: ({left}, {top}, {right}, {bottom})")
                        st.success("✅ Ready for registration!")
                    else:
                        st.warning("⚠️ No face detected in current frame")
                        st.info("💡 Try adjusting your position, lighting, or camera angle")
                        
                except Exception as e:
                    st.error(f"❌ Error during face detection test: {e}")
            else:
                st.warning("❌ No frame available from camera")
                
    if st.button("Reset Users"):
        reset_users()
    if st.button("Stop Camera"):
        stop_camera()

with col1:
    st.subheader("Camera Feed")
    frame_placeholder = st.empty()
    show_feed = st.checkbox("Show Camera Feed", value=True)

if st.session_state.camera and show_feed:
    circle = 2
    tmp = 0
    while True:
        # Check for remote commands inside the loop
        try:
            while not st.session_state.ws_status_queue.empty():
                msg = st.session_state.ws_status_queue.get_nowait()
                if msg == "CMD:RESET_USERS":
                    reset_users()
                    st.toast("Users reset by remote command!")
                elif msg == "CMD:RELOAD_DATA":
                    # Reload user data and FAISS index after WebSocket registration
                    st.session_state.user_data = st.session_state.user_management.get_all_user_data()
                    if os.path.exists("facial_faiss_index.bin"):
                        st.session_state.faiss_index = faiss.read_index("facial_faiss_index.bin")
                    print(f"🔄 Reloaded: {len(st.session_state.user_data)} users, {st.session_state.faiss_index.ntotal} vectors")
                elif msg.startswith("Connect") or msg.startswith("Disconnect"):
                     st.session_state.ws_status = msg
        except queue.Empty:
            pass

        # 1. Luôn lấy frame mới từ camera
        frame = st.session_state.camera.get_frame()
        if frame is None:
            frame_placeholder.warning("No frame available.")
            time.sleep(0.05)
            continue
        # frame = resize_with_pad(np.array(frame), (640, 640), padding_color=(0, 0, 0)) # REMOVED: C++ now handles resizing
        # Keep frame in BGR format - C++ will handle BGR to RGB conversion

        # 2. Chỉ chạy nhận diện mỗi 4 frame để tối ưu (gọi SOM)
        if tmp == 0:
            # Gọi SOM để detect face
            display_text = ""
            
            left, top, right, bottom, spoof_confidence, facial_feature = FR.detect_face(frame, frame.shape[0], frame.shape[1], frame.shape[2])
                
            print(f"🔍 SOM call #{st.session_state.no_face_counter + 1}: Face detection")
            
            # ✅ KIỂM TRA NẾU FRAME BỊ SKIP TỪ C++ (empty result)
            if left == 0 and top == 0 and right == 0 and bottom == 0:
                # Frame bị skip từ C++, không có thông tin face detection
                # Tăng counter vì đây vẫn là lần gửi xuống SOM
                st.session_state.no_face_counter += 1
                print(f"⏭️ No face detected - SOM call {st.session_state.no_face_counter}/10")
                
                # Xóa bounding box sau 10 lần liên tiếp không detect
                if st.session_state.no_face_counter >= 10:
                    st.session_state.last_known_box = None
                    st.session_state.no_face_counter =0 
                    st.session_state.last_known_text = ""
                    print(f"🗑️ Cleared bounding box after {st.session_state.no_face_counter} frames without face")
            else:
                # Frame được xử lý, tiếp tục logic bình thường
                
                # ✅ CHỈ CẬP NHẬT BOUNDING BOX KHI CÓ FACE DETECTION THÀNH CÔNG
                # Kiểm tra nếu có face được detect (left, top, right, bottom > 0)
                if left > 0 and top > 0 and right > 0 and bottom > 0:
                    st.session_state.last_known_box = (left, top, right, bottom)
                    st.session_state.no_face_counter = 0  # Reset counter khi có face
                    print(f"✅ Face detected! Reset counter to 0")
                    print(f"📦 Updated bounding box: ({left}, {top}, {right}, {bottom})")
                else:
                    # Tăng counter khi không có face
                    st.session_state.no_face_counter += 1
                    print(f"❌ No face detected - SOM call {st.session_state.no_face_counter}/10")
                    
                    # Xóa bounding box sau 10 lần liên tiếp không detect
                    if st.session_state.no_face_counter >= 10:
                        st.session_state.last_known_box = None
                        st.session_state.last_known_text = ""
                        print(f"🗑️ Cleared bounding box after {st.session_state.no_face_counter} frames without face")
                
                # Proceed with face recognition if face detected
                if left > 0 and top > 0 and right > 0 and bottom > 0 and st.session_state.faiss_index.ntotal > 0:
                    if len(facial_feature) == 128:
                        # Reload user data to ensure fresh data after WebSocket registration
                        st.session_state.user_data = st.session_state.user_management.get_all_user_data()
                        
                        print(f"\n{'='*60}")
                        print(f"🔍 [RECOGNITION DEBUG]")
                        print(f"   FAISS index vectors: {st.session_state.faiss_index.ntotal}")
                        print(f"   User database count: {len(st.session_state.user_data)}")
                        print(f"   User IDs in database: {list(st.session_state.user_data.keys())}")
                        if st.session_state.user_data:
                            for uid, udata in st.session_state.user_data.items():
                                print(f"      - ID {uid}: {udata.get('name', 'Unknown')}")
                        print(f"{'='*60}\n")
                        
                        facial_feature= np.array(facial_feature, dtype='float32').reshape(1, dims)
                        faiss.normalize_L2(facial_feature)
                        k = 3
                        distances, indices = st.session_state.faiss_index.search(facial_feature, k)
                        
                        best_match_name = None
                        best_distance = -1
                        
                        if len(indices) > 0 and len(indices[0]) > 0:
                            for i, int_id in enumerate(indices[0]):
                                distance = distances[0][i]
                                
                                # FAISS returns -1 for invalid IDs in IndexIDMap
                                if int_id != -1:
                                    # Look up user info by the integer ID (keys in JSON are strings)
                                    user_info = st.session_state.user_data.get(str(int_id))
                                    if user_info:
                                        user_name = user_info['name']
                                        print(f"Nearest neighbor {i+1}: ID: {int_id}, Distance: {distance:.4f}, Name: {user_name}")
                                        
                                        # Check if this is the best match above the threshold
                                        if distance >= st.session_state.cos_sim_thresh and distance > best_distance:
                                            best_match_name = user_name
                                            best_distance = distance
                                    else:
                                        print(f"Nearest neighbor {i+1}: ID: {int_id} not found in user data.")
                        
                        if best_match_name:
                            display_text = best_match_name
                            print(f"✅ Best match: {best_match_name} with distance: {best_distance:.4f}")
                        else:
                            print(f"❌ No match found above threshold {st.session_state.cos_sim_thresh}")
                            display_text = "Unknown Person"
                
                # ✅ CẬP NHẬT TEXT CHỈ KHI CÓ KẾT QUẢ MỚI
                if display_text and not display_text.startswith("⚠️"):
                    st.session_state.last_known_text = display_text
                    # Chỉ ghi lịch sử nếu nhận diện thành công (không phải Unknown Person)
                    if display_text != "Unknown Person":
                        print(f"🚪 REAL-TIME LOGGING: Attempting to log access for: {display_text}")
                        
                        # Đảm bảo access_history được khởi tạo
                        if not hasattr(st.session_state, 'access_history') or st.session_state.access_history is None:
                            print(f"🔧 Initializing AccessHistory...")
                            st.session_state.access_history = AccessHistory()
                        
                        try:
                            # Gọi log_access và nhận kết quả
                            log_result = st.session_state.access_history.log_access(display_text)
                            current_time = datetime.now().strftime("%H:%M:%S")
                            
                            # Luôn refresh bảng ngay sau mỗi lần nhận diện thành công
                            st.session_state.access_history.load_history()
                            
                            if log_result:
                                print(f"✅ REAL-TIME SUCCESS: Logged and saved to file for: {display_text}")
                                st.session_state.last_log_status = f"✅ Logged: {display_text}"
                                st.session_state.last_log_time = current_time
                            else:
                                print(f"⏭️ REAL-TIME SKIP: Not logged due to 5-minute rule for: {display_text}")
                                st.session_state.last_log_status = f"⏭️ Skipped: {display_text} (5-min rule)"
                                st.session_state.last_log_time = current_time
                            
                            # 🔄 REFRESH BẢNG NGAY LẬP TỨC sau khi nhận diện thành công
                            print(f"🔄 [REAL-TIME] Starting history table refresh for: {display_text}")
                            print(f"🗑️ [REAL-TIME] Clearing old table...")
                            
                            # Đảm bảo xóa bảng cũ trước
                            history_placeholder.empty()
                            
                            print(f"🆕 [REAL-TIME] Creating new table...")
                            update_history_display()
                            
                            print(f"✅ [REAL-TIME] History table refresh completed for: {display_text}")
                        except Exception as e:
                            print(f"❌ REAL-TIME ERROR: {e}")
                            st.session_state.last_log_status = "Error logging access"
                            st.session_state.last_log_time = ""
                            st.session_state.last_log_time = ""
        
        tmp = (tmp + 1) % 4  # Chỉ nhận diện mỗi 4 frame
        
        # 3. Hiển thị kết quả lên streamlit
        # Create a copy of frame for display to avoid modifying the original
        frame_display = frame.copy()
        
        if st.session_state.last_known_box:
            left, top, right, bottom = st.session_state.last_known_box
            frame_display = cv2.rectangle(frame_display, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Hiển thị text nhận diện được
            display_text = st.session_state.last_known_text
            if display_text and not display_text.startswith("⚠️"):
                # Hiển thị tên người nhận diện được
                frame_display = cv2.putText(frame_display, display_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        # Frame_display is currently in BGR format (OpenCV default)
        # Convert to RGB for proper display on Streamlit
        frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        
        # Hiển thị frame lên streamlit with RGB format
        frame_placeholder.image(frame_display, channels="RGB")
        
        # Giảm độ trễ giữa các lần hiển thị
        time.sleep(0.1)

# Hiển thị bảng lịch sử truy cập ban đầu (chỉ khi camera chưa chạy)
if not (st.session_state.camera and show_feed):
    print("📊 Displaying initial history table (camera not running)")
    update_history_display()

# Nút refresh thủ công
if st.button("🔄 Refresh History"):
    if hasattr(st.session_state, 'access_history'):
        st.session_state.access_history.load_history()
        print("🔄 Manual refresh triggered")
        history_placeholder.empty()  # Xóa bảng cũ trước
        update_history_display()
