#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include "../dataStructures.h"



namespace OpenVinoInference {

struct Detection {
	short class_id;
	float confidence;
	cv::Rect box;
};

class Inference {
 public:

	Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold);
	void RunInference(cv::Mat &frame,std::vector<BoundingBox>& bBoxes);

	int fps_;                          
    int totalFPS = 0;
    int frameCount = 0;
    int avgFPS = 0;

 private:
	void InitializeModel(const std::string &model_path);
	void Preprocessing(const cv::Mat &frame);
	void PostProcessing(cv::Mat &frame,std::vector<BoundingBox>& bBoxes);
	void DrawDetectedObject(cv::Mat &frame, const Detection &detections) const;
	cv::Mat letterbox(const cv::Mat& img, const cv::Size& new_size, int fill_value = 114);

	cv::Point2f scale_factor_;
	cv::Point2f letterbox_offset_;			// Scaling factor for the input frame
	cv::Size2f model_input_shape_;	        // Input shape of the model
	cv::Size model_output_shape_;		    // Output shape of the model

	ov::InferRequest inference_request_;  // OpenVINO inference request
	ov::CompiledModel compiled_model_;    // OpenVINO compiled model

	float model_confidence_threshold_;  // Confidence threshold for detections
	float model_NMS_threshold_;         // Non-Maximum Suppression threshold

	std::vector<std::string> classes_ {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
		"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
		"scissors", "teddy bear", "hair drier", "toothbrush"
	};
};

} // namespace OpenVinoInference

