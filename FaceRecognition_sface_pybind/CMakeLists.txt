cmake_minimum_required(VERSION 3.4...3.18)
project(face_recognition)

set(PYBIND11_FINDPYTHON OFF)
set(PYBIND11_NOPYTHON ON)
# set(PYBIND11_PYTHON_VERSION 3.8)

# Update ROOT_DIR to match build_wrapper.sh
set(ROOT_DIR "/mnt/i/AnhTu/Internship/working")
set(PYTHON_MODULE_EXTENSION ".so" CACHE INTERNAL "Cross python lib extension")
set(PYTHONLIBS_FOUND TRUE CACHE INTERNAL "")

# Update Python include path
set(PYTHON_INCLUDE_DIRS /mnt/i/AnhTu/VNEMEX/face-recog/python-3.8-headers)
message(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
include_directories(${PYTHON_INCLUDE_DIRS})


add_subdirectory(pybind11)

add_library(face_recognition_pybind MODULE face_recognition_pybind.cpp)
target_link_libraries(face_recognition_pybind PRIVATE pybind11::module)
set_target_properties(face_recognition_pybind PROPERTIES PREFIX "" SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Update OpenCV paths
set(OpenCV_INCLUDE_DIRS ${ROOT_DIR}/opencv4.10/include)
set(OpenCV_LIB_DIR ${ROOT_DIR}/opencv4.10/lib)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS}/opencv2)

# Link against the wrapper library built in the sibling directory
target_link_libraries(face_recognition_pybind PRIVATE ${CMAKE_SOURCE_DIR}/../face_recognition_sface_2021dec_ovx/libFaceRecog_wrapper.so)