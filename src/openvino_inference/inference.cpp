#include "inference.h"
#include <chrono>

#include <memory>
#include <opencv2/dnn.hpp>
#include <random>
#include "../dataStructures.h"


namespace OpenVinoInference{

// Constructor to initialize the model with specified input shape
Inference::Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold) {
	model_input_shape_ = model_input_shape;
	model_confidence_threshold_ = model_confidence_threshold;
	model_NMS_threshold_ = model_NMS_threshold;
	InitializeModel(model_path);
}

void Inference::InitializeModel(const std::string &model_path) {
	ov::Core core; // OpenVINO core object
	std::shared_ptr<ov::Model> model = core.read_model(model_path); // Read the model from file

	// If the model has dynamic shapes, reshape it to the specified input shape
	if (model->is_dynamic()) {
		model->reshape({1, 3, static_cast<long int>(model_input_shape_.height), static_cast<long int>(model_input_shape_.width)});
	}

	// Preprocessing setup for the model
	ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
	ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
	ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255, 255, 255});
	ppp.input().model().set_layout("NCHW");
	ppp.output().tensor().set_element_type(ov::element::f32);
	model = ppp.build(); // Build the preprocessed model

	// Compile the model for inference
	compiled_model_ = core.compile_model(model, "AUTO");
	inference_request_ = compiled_model_.create_infer_request(); // Create inference request

	short width, height;

	// Get input shape from the model
	const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
	const ov::Shape input_shape = inputs[0].get_shape();
	height = input_shape[1];
	width = input_shape[2];
	model_input_shape_ = cv::Size2f(width, height);

	// Get output shape from the model
	const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
	const ov::Shape output_shape = outputs[0].get_shape();
	height = output_shape[1];
	width = output_shape[2];
	model_output_shape_ = cv::Size(width, height);
}

// Method to run inference on an input frame
void Inference::RunInference(cv::Mat &frame,std::vector<BoundingBox>& bBoxes) {
    // Record the start time
    auto starttime_1 = std::chrono::high_resolution_clock::now();    
    Preprocessing(frame); // Preprocess the input frame
	auto starttime_2 = std::chrono::high_resolution_clock::now();
    inference_request_.infer(); // Run inference
	auto starttime_3 = std::chrono::high_resolution_clock::now();
    PostProcessing(frame,bBoxes); // Postprocess the inference results
	auto starttime_4 = std::chrono::high_resolution_clock::now();
    
    double pre_process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_2 - starttime_1).count();
    double process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_3 - starttime_2).count();
    double post_process_time = std::chrono::duration_cast<std::chrono::milliseconds>(starttime_4 - starttime_3).count();

    std::cout << "pre_process_time: " << pre_process_time << " ms" << std::endl;
    std::cout << "process_time: " << process_time << " ms" << std::endl;
    std::cout << "post_process_time: " << post_process_time << " ms" << std::endl;

    fps_ = static_cast<int>(1000.0 / (pre_process_time + process_time + post_process_time));

	totalFPS += fps_;
    frameCount++;
	
	// Prepare FPS text
	std::cout << "FPS: " << fps_ << std::endl;
	std::string fpsText = "FPS: " + std::to_string(fps_); 

	// Put the FPS text in the top left corner of the frame
	// cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, 8);
}

// Helper function for letterbox resizing that updates class attributes
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

// Simplified preprocessing method that uses letterbox and stored attributes
void Inference::Preprocessing(const cv::Mat &frame) {
    // Apply letterbox resizing (this also updates scale_factor_ and letterbox_offset_)
    cv::Mat letterboxed_frame = letterbox(frame, model_input_shape_);
    
    // Create input tensor from the letterboxed frame
    float *input_data = (float *)letterboxed_frame.data;
    const ov::Tensor input_tensor = ov::Tensor(compiled_model_.input().get_element_type(), 
                                               compiled_model_.input().get_shape(), 
                                               input_data);
                                               
    // Set input tensor for inference
    inference_request_.set_input_tensor(input_tensor);
}


// Method to postprocess the inference results
void Inference::PostProcessing(cv::Mat &frame,std::vector<BoundingBox>& bBoxes) {
	std::vector<int> class_list;
	std::vector<float> confidence_list;
	std::vector<cv::Rect> box_list;
    
	// Get the output tensor from the inference request
	const float *detections = inference_request_.get_output_tensor().data<const float>();
	cv::Mat detection_outputs(model_output_shape_, CV_32F, (float *)detections); // Create OpenCV matrix from output tensor
	detection_outputs = detection_outputs.t();

    float* data = (float*)detection_outputs.data;
	// Iterate over detections and collect class IDs, confidence scores, and bounding boxes
	for (int i = 0; i < detection_outputs.rows; ++i) {
		float* classesScores = data + 4;
		cv::Mat scores(1,detection_outputs.cols-4,CV_32FC1, classesScores);
		cv::Point class_id;
		double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

		// Check if the detection meets the confidence threshold
		if (maxClassScore> model_confidence_threshold_) {
            confidence_list.push_back(maxClassScore);
            class_list.push_back(class_id.x);
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int(((x - 0.5 * w)-letterbox_offset_.x) * scale_factor_.x);
            int top = int(((y - 0.5 * h) - letterbox_offset_.y) * scale_factor_.y);

            int width = int(w * scale_factor_.x);
            int height = int(h * scale_factor_.y);
			box_list.push_back(cv::Rect(left, top, width, height));
		}
		data += detection_outputs.cols;
	}

	// Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
	std::vector<int> NMS_result;
	cv::dnn::NMSBoxes(box_list, confidence_list, model_confidence_threshold_, model_NMS_threshold_, NMS_result);

	// Collect final detections after NMS
	for (int i = 0; i < NMS_result.size(); ++i) {
		Detection result;
		const unsigned short id = NMS_result[i];
		result.class_id = class_list[id];
		result.confidence = confidence_list[id];
		result.box = box_list[id];

		// DrawDetectedObject(frame, result);

        BoundingBox bBox;
        bBox.roi = box_list[id];
        bBox.classID = class_list[id];
        bBox.confidence = confidence_list[id];
        bBox.boxID = (int)bBoxes.size(); // zero-based unique identifier for this bounding box
        bBoxes.push_back(bBox);		
	}
}

void Inference::DrawDetectedObject(cv::Mat &frame, const Detection &detection) const {
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
	std::string classString = classes_[class_id] + std::to_string(confidence).substr(0, 4);
	
	// Get the size of the text box
	cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.75, 2, 0);
	cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
	
	// Draw the text box
	cv::rectangle(frame, textBox, color, cv::FILLED);
	
	// Put the class label and confidence text above the bounding box
	cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 2, 0);

}
} // namespace OpenVinoInference
