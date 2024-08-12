
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "objectDetection2D.hpp"


using namespace std;

// detects objects in an image using the YOLO library and a set of pre-trained objects from the COCO database;
// a set of 80 classes is listed in "coco.names" and pre-trained weights are stored in "yolov3.weights"
// The bBoxes vector will be updated with the following
    // * 
void detectObjects(cv::Mat& img, std::vector<BoundingBox>& bBoxes, float confThreshold, float nmsThreshold, 
                   std::string basePath, std::string classesFile, std::string modelConfiguration, std::string modelWeights, bool bVis)
{
    std ::vector<std::string> classes;
    // read class names from file
    std::ifstream ifs(classesFile);
    std::string line;
    while (std::getline(ifs, line)) 
    {
        classes.push_back(line);
    }

    // load neural network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); // set backend as OpenCV. The model will be optimized for OpenCV.
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // set target as CPU for computation. The computation will be done on CPU.

    // generate 4D blob from input image. This is equivalent to tensor in PyTorch.
    cv::Mat blob;
    double scalefactor = 1/255.0;
    cv::Size size(416, 416);
    cv::Scalar mean(0,0,0);
    bool swapRB = false;
    bool crop = false;
    cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);

    // Get names of output layers    
    std::vector<int> outLayers = net.getUnconnectedOutLayers(); // get indices of output layers, i.e. layers with unconnected outputs
    std::vector<cv::String> names(outLayers.size());
    std::vector<cv::String> layersNames = net.getLayerNames(); // get names of all layers in the network

    for (size_t i = 0; i < outLayers.size(); ++i) // Get the names of the output layers in names
    {
        names[i] = layersNames[outLayers[i] - 1];
    }

    // invoke forward propagation through network
    std::vector<cv::Mat> netOutput;
    net.setInput(blob);
    net.forward(netOutput, names);

    // netOutput is a vector of 3 Mat elements. Each Mat has 85 channels (columns). The number of rows is equal to the number of bounding boxes.
    // The first 4 channels are for center_x, center_y, width, height. The 5th channel is for confidence. The rest of the channels are for class probabilities. 

    // Scan through all bounding boxes and keep only the ones with high confidence
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Note that std::cout << netOutput[i].size() will display the number of columns and rows in the Mat.[width x height]
    for (size_t i = 0; i < netOutput.size(); ++i)
    {
        for (int j = 0; j < netOutput[i].rows; ++j)
        {
            float* data = netOutput[i].ptr<float>(j);
            cv::Mat scores(1, netOutput[i].cols - 5, CV_32F, data + 5);  // scores will have rows = 1, size = netOutput[i].cols - 5, type = CV_32F, pointer = data + 5.
            cv::Point classId;
            double confidence;

            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classId); // ptr to minimum value = 0, ptr to maximum value = &confidence, location of minimum value = 0, location of maximum value = &classId
            if (confidence > confThreshold) // if confidence is greater than threshold, then keep the bounding box
            {
                // The bounding box dimensions are normalized. We need to convert them to the original image dimensions.
                int centerX = (int)(data[0] * img.cols); // data[0] is the center_x ( equivalent to columns in the image)
                int centerY = (int)(data[1] * img.rows); // data[1] is the center_y ( equivalent to rows in the image)
                int width = (int)(data[2] * img.cols); // data[2] is the width ( equivalent to columns in the image)
                int height = (int)(data[3] * img.rows); // data[3] is the height ( equivalent to rows in the image)
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                // Add the bounding box to the list
                cv::Rect box(left, top, width, height); // attributs of rect are x (column), y (row), width, height.
                boxes.push_back(box);
                classIds.push_back(classId.x); // classId.x is the index of the class (column) with maximum score.
                confidences.push_back((float)confidence);
            }
        }
    }
    // Now all the bounding boxes with high confidence are stored in the boxes vector. We need to suppress the overlapping boxes and retain only the best one using Non-Maximum Suppression (NMS).
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices); // Perform NMS on the boxes with high confidence, the indices of the retained boxes are stored in the indices vector.

    // Draw the bounding boxes on the image
    for (auto it = indices.begin(); it != indices.end(); ++it)
    {
        BoundingBox bBox;
        bBox.roi = boxes[*it];
        bBox.classID = classIds[*it];
        bBox.confidence = confidences[*it];
        bBox.boxID = (int)bBoxes.size(); // zero-based unique identifier for this bounding box
        bBoxes.push_back(bBox);            
    }
    
    // show results
    if(bVis) {
        
        cv::Mat visImg = img.clone();
        for(auto it=bBoxes.begin(); it!=bBoxes.end(); ++it) {
        // Draw rectangle displaying the bounding box
        int x, y, w, h;
        x = it->roi.x;
        y = it->roi.y;
        w = it->roi.width;
        h = it->roi.height;
        cv::rectangle(visImg, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 255, 0), 2);

        // Display class label and confidence
        std::string label = cv::format("%.2f", it->confidence);
        label = classes[it->classID] + ":" + label;

        // Display label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        y = std::max(y, labelSize.height);
        cv::rectangle(visImg, cv::Point(x, y - round(1.5*labelSize.height)), cv::Point(x + round(1.5*labelSize.width), y + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(visImg, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
            
        }
        
        string windowName = "Object classification";
        cv::namedWindow( windowName, 1 );
        cv::imshow( windowName, visImg );
        cv::waitKey(0); // wait for key to be pressed
    }
}
