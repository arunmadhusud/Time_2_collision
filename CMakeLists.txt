cmake_minimum_required(VERSION 3.12)

project(camera_fusion)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the location of ONNX Runtime headers and libraries
set(ONNX_RUNTIME_INCLUDE_DIR "/home/arun/Adv_CV/onnx_rt_new/onnxruntime/include/onnxruntime/core/session")
set(ONNX_RUNTIME_LIB_DIR "/home/arun/Adv_CV/onnx_rt_new/onnxruntime/build/Linux/RelWithDebInfo")

# Find the ONNX Runtime library
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime libonnxruntime
    HINTS ${ONNX_RUNTIME_LIB_DIR}
    PATH_SUFFIXES lib
    REQUIRED
)

set(OpenCV_DIR /home/arun/opencv/build)
find_package(OpenCV 4 REQUIRED)

# Executable for create matrix exercise
add_executable (3D_object_tracking src/camFusion_Student.cpp src/FinalProject_Camera.cpp src/lidarData.cpp src/matching2D_Student.cpp src/onnxruntime_inference/inference.cpp src/cvdnn_inference/inference.cpp src/openvino_inference/inference.cpp)

# Link libraries
target_include_directories(3D_object_tracking PRIVATE ${ONNX_RUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} /opt/intel/openvino_2024/runtime/include)

# Link to ONNX Runtime
target_link_libraries(3D_object_tracking PRIVATE ${ONNXRUNTIME_LIB} ${OpenCV_LIBRARIES} /opt/intel/openvino_2024/runtime/lib/intel64/libopenvino.so)