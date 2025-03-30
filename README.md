# Time_2_Collision

## 1. Introduction

This project is part of the Sensor Fusion Nanodegree at Udacity. The goal is to detect and track 3D objects using a combination of camera and LiDAR data, then calculate the Time to Collision (TTC) for each object in the ego lane using camera and LiDAR data. Calculating TTC is crucial for collision avoidance in autonomous driving, where precise timing can prevent accidents.

The steps followed in the project are illustrated in the flow diagram below:

![project_flow](./misc/project_flow.png)

## 2. 2D Object Detection using YOLO v8

The first step of the project involves detecting objects in camera images using YOLO v8. To improve inference speed on CPU-based systems, I integrated YOLOv8 with OpenVINO and applied static INT8 quantization. Additionally, you can switch between ONNX Runtime or OpenCV::DNN by passing command-line arguments.

Before finalizing this inference method, I explored various options, including OpenCV-DNN, ONNX Runtime, and OpenVINO, for deploying YOLOv8 in C++ on CPU. The code for model quantization, along with benchmarking details for using ONNX Runtime, OpenVINO, and OpenCV’s DNN module, is available [here](https://github.com/arunmadhusud/Fast_YOLOv8_CPP)
![yolo_output](./misc/yolo.png)

*Figure: Output of YOLO v8 object detection*

## 3. LiDAR Point Cloud Processing

The LiDAR point cloud data is processed to detect objects within the ego lane. Point cloud data represents the environment in 3D by capturing points in space. The data is first filtered to remove points outside the ego lane. Then, the 3D LiDAR points are projected onto the image plane using the camera projection matrix, aligning the LiDAR data with the camera image. Points within the bounding boxes of detected objects are then clustered together. Below is a top view of the clustered LiDAR points within the ego lane:

![lidar_output](./misc/lidar_output.png)

*Figure: Top view of clustered LiDAR points in the ego lane. The points are shown only for the rear part of the vehicle in the ego lane.*

## 4. Tracking 2D Bounding Boxes

The 2D bounding boxes detected using YOLO are tracked across frames using a keypoint matching method. Keypoints are detected using the Shi-Tomasi corner detection algorithm, and descriptors are calculated using the BRISK algorithm. Keypoints are matched between consecutive frames using the Brute-Force matcher. The bounding box in the next frame is tracked based on the number of matched keypoints.

Users can choose from several keypoint detection algorithms: Shi-Tomasi, Harris, FAST, BRISK, ORB, and SIFT. Descriptor calculation options include BRISK, ORB, FAST, and SIFT. For keypoint matching, Brute Force and FLANN matching are implemented, with options for Nearest Neighbour and k-Nearest Neighbour matching.

## 5. Tracking 3D Objects

The 2D bounding boxes are tracked as explained above. The 3D points projected to the image plane and within the tracked 2D bounding box are considered part of the tracked 3D object.

## 6. Calculating Time to Collision (TTC)

With the tracked 3D objects in the ego lane and 2D bounding boxes in the camera image, the Time to Collision (TTC) can be calculated using the camera and LiDAR data .

The TTC for LiDAR data is calculated using the following formula:


```
TTC = -dT * d1 / (d0 - d1) 
```

Where:
- `d0` and `d1` are the minimum distances between the ego vehicle and the object in the previous and current frames, respectively.
- `dT` is the time elapsed between the frames.

This method is illustrated below:

![lidar_ttc](./misc/ttc_lidar.png)

*Figure: Illustration of TTC calculation using LiDAR data*

## 7. Installation


Generate the yolov8 weights file using the instrunctions given in the [repository](https://github.com/arunmadhusud/Fast_YOLOv8_CPP) and place the weights file in the dat folder. The folder structure should look like this:
data
  - yolov8n_int8.xml # OpenVINO IR file
  - yolov8n_int8.bin # OpenVINO IR file
  - yolov8n_st_quant.onnx # ONNX model file

Run the following commands to build the project:

```bash
# Clone the repository
git clone https://github.com/your-repository-link
cd Time_2_Collision

# Download YOLO weights
wget -O data/yolov3.weights https://pjreddie.com/media/files/yolov3.weights

# Build the project
mkdir build && cd build
cmake ..
make

# Run the executable
./3D_object_tracking [cvdnn|onnx|openvino]
```

## 3. Results

The results of the TTC estimated using camera data and LiDAR data are shown in the video below:

![TTC](misc/ttc_results.gif)

*Figure: Results of TTC estimation using camera and LiDAR data*

## 4. Conclusion

The project demonstrates the use of camera and LiDAR data to detect and track 3D objects in the ego lane. The TTC is calculated using camera data and LiDAR data separately. The results show that the TTC calculated using camera data and LiDAR data separately are not consistent with each other. Sometimes the TTC calculated were way off from the expected value. The results can be improved by using a more robust sensor fusion algorithm such as Kalman filter or Unscented Kalman filter.
