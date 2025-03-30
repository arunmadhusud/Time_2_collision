
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <chrono>
// #include <opencv2/xfeatures2d.hpp>
// #include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"
#include "onnxruntime_inference/inference.h"
#include "cvdnn_inference/inference.h"
#include "openvino_inference/inference.h"


void onnxDetector(OnnxRuntimeInference::YOLO_V8& yoloDetector, cv::Mat& img,std::vector<BoundingBox>& bBoxes) {
        if (img.empty()) {
            std::cerr << "Error loading image" << std::endl;
            return;
        }        
        yoloDetector.RunSession(img,bBoxes);        
        return;
}

void cvdnnDetector(CvdnnInference::Inference& inf, cv::Mat& frame, std::vector<BoundingBox>& bBoxes) {
        if (frame.empty()) {
            std::cerr << "Error loading image" << std::endl;
            return;
        }

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::vector<CvdnnInference::Detection> output = inf.runInference(frame,bBoxes);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        double inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        int fps_ = static_cast<int>(1000.0 / inference_time);

        // std::cout << "Inference time = " << inference_time  << "[ms]" << std::endl;
        // std::cout << "FPS = " << fps_ << "[fps]" << std::endl;

        std::string fpsText = "FPS: " + std::to_string(fps_);
        // cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, 8);
        return;
}

void openvinoDetector(OpenVinoInference::Inference& inference,cv::Mat& image, std::vector<BoundingBox>& bBoxes) {       
        // Check if the image was successfully loaded
        if (image.empty()) {
            std::cerr << "ERROR: image is empty" << std::endl;
            return;
        }        
        // Run inference on the input image
        inference.RunInference(image,bBoxes); 
        return;
    
}


using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 50;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // Parse command line argument for inference method (openvino, onnxruntime, cv::dnn)
    std::string inferenceMethod = "openvino"; // Default to openvino
    if (argc > 1) {
        inferenceMethod = argv[1]; // Read from command-line argument
    }
    else {
        std::cout << "Default inference method: " << inferenceMethod << std::endl;
        std::cout << "For other inference methods, please provide the method name as an argument. Options are: openvino, onnxruntime, cvdnn" << std::endl;
        std::cout << "Example: ./3D_object_tracking openvino" << std::endl;
    }

    //onnxruntime
    OnnxRuntimeInference::YOLO_V8 yoloDetector;
    OnnxRuntimeInference::DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.5;
    params.iouThreshold = 0.5;
    params.modelPath = dataPath + "dat/yolov8n_st_quant.onnx";
    params.imgSize = {640, 640};
    yoloDetector.CreateSession(params);

    //cvdnn
    const cv::String  modelConfig  = dataPath + "dat/yolov8n_int8.xml";
    const cv::String  modelWeights = dataPath + "dat/yolov8n_int8.bin";
    CvdnnInference::Inference inf(modelConfig,modelWeights, cv::Size(640, 640));

    //openvino
    const float confidence_threshold = 0.5;
    const float NMS_threshold = 0.5;
    const std::string model_path = dataPath + "dat/yolov8n_int8.xml";
    OpenVinoInference::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);
    
    bool openvinoinference = false;
    bool onnxruntimeinference = false;
    bool cvdnninference = false;

    // Determine which inference method to use
    if (inferenceMethod == "openvino") {
        std::cout << "Using OpenVino Inference" << std::endl;
        openvinoinference = true;
        onnxruntimeinference = false;
        cvdnninference = false;
    } 
    else if (inferenceMethod == "onnxruntime") {
        std::cout << "Using ONNX Runtime Inference" << std::endl;
        openvinoinference = false;
        onnxruntimeinference = true;
        cvdnninference = false;
    }
    else if (inferenceMethod == "cvdnn") {
        std::cout << "Using cv::dnn Inference" << std::endl;
        openvinoinference = false;
        onnxruntimeinference = false;
        cvdnninference = true;
    }
    else {
        std::cerr << "Unknown inference method! Please specify 'openvino', 'onnxruntime', or 'cv::dnn'" << std::endl;
        return -1;
    }


    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.2;
        
        auto start = std::chrono::high_resolution_clock::now();

        // Call the Detector function
        if (onnxruntimeinference) onnxDetector(yoloDetector, (dataBuffer.end() - 1)->cameraImg,(dataBuffer.end() - 1)->boundingBoxes);
        if (cvdnninference) cvdnnDetector(inf, (dataBuffer.end() - 1)->cameraImg,(dataBuffer.end() - 1)->boundingBoxes);
        if (openvinoinference) openvinoDetector(inference, (dataBuffer.end() - 1)->cameraImg,(dataBuffer.end() - 1)->boundingBoxes);

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = end - start; 
        cout << "Time taken by detectObjects: " 
            << std::chrono::duration<double, std::milli>(duration).count() << " ms" << endl;

        cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // print number of Lidar points associated with ROI
        // for (auto it = (dataBuffer.end()-1)->boundingBoxes.begin(); it != (dataBuffer.end()-1)->boundingBoxes.end(); ++it)
        // {
        //     cout << "#3 : LIDAR points in bounding box " << it->boxID << " : " << it->lidarPoints.size() << endl;
        // }

        // Visualize 3D objects
        bVis = false;
        if(bVis)
        {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
        }
        bVis = false;

        cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        
        
        // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
        // continue; // skips directly to the next image without processing what comes beneath

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        // attributes of keypoints are point, size (diameter of the meaningful keypoint neighborhood), angle (orientation of the keypoint), response (the strength of the keypoint), octave (pyramid layer which the keypoint was detected), class_id (object id)
        string detectorType = "SHITOMASI";

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else if ((detectorType.compare("FAST") == 0) || (detectorType.compare("BRISK") == 0) || (detectorType.compare("ORB") == 0) || (detectorType.compare("SIFT") == 0))
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        else
        {
            cerr << "#5 : DETECT KEYPOINTS failed. Wrong detectorType - " << detectorType << ". Use one of the following detectors: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT" << endl;
            exit(-1);
        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors; // each row in the matrix is a descriptor for a keypoint
        string descriptorType = "BRISK"; // BRISK, ORB, SIFT
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches; 
            // attributes of DMatch are queryIdx (index of the keypoint in the first image), trainIdx (index of the keypoint in the second image), distance (distance between descriptors)
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches


            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;

            // print bbBestMatches
            // std::cout << "bbBestMatches: " << endl;
            // for (auto it = bbBestMatches.begin(); it != bbBestMatches.end(); ++it)
            // {
            //     std::cout << "prev frame boxID: " << it->first << " curr frame boxID: " << it->second << endl;
            // }
            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                    }
                }
                
                // print prevBB and currBB
                // std::cout << "prevBB: " << prevBB->boxID << " currBB: " << currBB->boxID << endl;

                // compute TTC for current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                {
                    //// compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar; 
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);

                    // assign enclosed keypoint matches to bounding box (clusterKptMatchesWithROI)
                    //  compute time-to-collision based on camera (computeTTCCamera)
                    // double ttcCamera;
                    // clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                    // computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);


                    bVis = true;
                    if (bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        char str[200];
                        sprintf(str, "TTC Lidar : %f s", ttcLidar);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(1);
                    }
                    bVis = false;

                } // eof TTC computation
            } // eof loop over all BB matches            

        }

    } // eof loop over all images

    return 0;
}
