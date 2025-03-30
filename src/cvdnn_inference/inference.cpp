#include "inference.h"
#include "../dataStructures.h"


namespace CvdnnInference{

Inference::Inference(const cv::String& modelconfig,const cv::String& modelweights, const cv::Size &modelInputShape)
{
    modelConfig = modelconfig;
    modelWeights = modelweights;
    modelShape = modelInputShape;
    loadOnnxNetwork();
}

cv::Mat Inference::letterbox(const cv::Mat& img, const cv::Size& new_size, int fill_value) {
    // Calculate the aspect ratio
    float aspect_ratio = std::min((float)new_size.width / img.cols, (float)new_size.height / img.rows);
    
    // Store scale factors for bbox mapping in post-processing
    scale_factor_.x = 1.0f / aspect_ratio;
    scale_factor_.y = 1.0f / aspect_ratio;
    
    // Compute new dimensions while maintaining the aspect ratio
    int new_w = int(img.cols * aspect_ratio);
    int new_h = int(img.rows * aspect_ratio);
    
    // Resize image
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    
    // Create a blank image with the fill color (default to 114 for YOLO)
    cv::Mat padded_img(new_size, img.type(), cv::Scalar(fill_value, fill_value, fill_value));
    
    // Compute padding offsets to center the resized image
    letterbox_offset_.x = (new_size.width - new_w) / 2.0f;
    letterbox_offset_.y = (new_size.height - new_h) / 2.0f;
    
    // Copy resized image into the center of the padded image
    resized_img.copyTo(padded_img(cv::Rect(letterbox_offset_.x, letterbox_offset_.y, new_w, new_h)));
    
    return padded_img;
}


std::vector<Detection> Inference::runInference(const cv::Mat &input,std::vector<BoundingBox>& bBoxes)
{   
    auto starttime_1 = std::chrono::steady_clock::now();

    cv::Mat modelInput;
    modelInput = letterbox(input,modelShape);

    cv::Mat blob;
    cv::cvtColor(modelInput, modelInput, cv::COLOR_BGR2RGB);
    modelInput.convertTo(modelInput, CV_32F, 1.0f / 255.0f);

    cv::dnn::blobFromImage(modelInput, blob);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;

    auto starttime_2 = std::chrono::steady_clock::now();

    net.forward(outputs, net.getUnconnectedOutLayersNames());

    auto starttime_3 = std::chrono::steady_clock::now();


    cv::Mat output = outputs[0];

    int rows = output.size[2];
    int dimensions = output.size[1];
    
    output = output.reshape(1, dimensions);
    cv::transpose(output, output);
    
    float *data = (float *)output.data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < output.rows; ++i)
    { 
        float *classes_scores = data+4;

        cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > modelScoreThreshold)
        {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int(((x - 0.5 * w)-letterbox_offset_.x) * scale_factor_.x);
            int top = int(((y - 0.5 * h) - letterbox_offset_.y) * scale_factor_.y);

            int width = int(w * scale_factor_.x);
            int height = int(h * scale_factor_.y);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += output.cols;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    std::vector<Detection> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);        
        result.color = cv::Scalar(dis(gen),dis(gen),dis(gen));

        result.className = classes[result.class_id];
        result.box = boxes[idx];
        detections.push_back(result);
        // DrawDetectedObject(input, result);

        BoundingBox bBox;
        bBox.roi = boxes[idx];
        bBox.classID = class_ids[idx];
        bBox.confidence = confidences[idx];
        bBox.boxID = (int)bBoxes.size(); // zero-based unique identifier for this bounding box
        bBoxes.push_back(bBox);          
    }


    auto starttime_4 = std::chrono::steady_clock::now();

    double pre_process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_2 - starttime_1).count();
    double process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_3 - starttime_2).count();
    double post_process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_4 - starttime_3).count();

    std::cout << "pre_process_time: " << pre_process_time << " ms" << std::endl;
    std::cout << "process_time: " << process_time << " ms" << std::endl;
    std::cout << "post_process_time: " << post_process_time << " ms" << std::endl;

    return detections;


}

void Inference::loadOnnxNetwork()
{   
    net = cv::dnn::readNetFromModelOptimizer(modelConfig, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    // net = cv::dnn::readNetFromONNX(modelPath);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void Inference::DrawDetectedObject(const cv::Mat &frame, const Detection &detection){
	const cv::Rect &box = detection.box;
	const float &confidence = detection.confidence;
	const int &class_id = detection.class_id;
	
	// Generate a random color for the bounding box
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(120, 255);
	const cv::Scalar &color = cv::Scalar(dis(gen), dis(gen), dis(gen));
	
	// Draw the bounding box around the detected object
	cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), color, 3);
	
	// Prepare the class label and confidence text
	std::string classString = classes[class_id] + std::to_string(confidence).substr(0, 4);


	// Get the size of the text box
	cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.75, 2, 0);
	cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
	
	// Draw the text box
	cv::rectangle(frame, textBox, color, cv::FILLED);
	
	// Put the class label and confidence text above the bounding box
	cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 2, 0);

}
}

