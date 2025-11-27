/****************************************************************************
*   Unified Post-Process Header for Yunet and Sface
****************************************************************************/
#ifndef _VNN_POST_PROCESS_H_
#define _VNN_POST_PROCESS_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include "vsi_nn_pub.h"

// Structure to hold detected face with landmarks (from Yunet)
struct YunetFaceObject {
    cv::Rect rect;              // Bounding box (x, y, w, h)
    float landmarks[10];        // 5 landmarks x 2 coordinates (x, y)
    float confidence;           // Detection confidence score
};

// Yunet Post Process
vsi_status vnn_PostProcessYunet(
    vsi_nn_graph_t *graph,
    std::vector<YunetFaceObject>& faces,
    int inputW,
    int inputH
);

const vsi_nn_postprocess_map_element_t * vnn_GetPostProcessMapYunet();
uint32_t vnn_GetPostProcessMapCountYunet();

// Sface Post Process
vsi_status vnn_PostProcessSface(vsi_nn_graph_t *graph);
std::vector<float> vnn_PostProcessSfaceExtractFeature(vsi_nn_graph_t *graph);

const vsi_nn_postprocess_map_element_t * vnn_GetPostProcessMapSface();
uint32_t vnn_GetPostProcessMapCountSface();

#endif
