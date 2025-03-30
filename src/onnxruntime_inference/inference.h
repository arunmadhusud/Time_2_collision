#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include "../dataStructures.h"


namespace OnnxRuntimeInference{
    
struct DL_INIT_PARAM
{
    std::string modelPath;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.5f;
    float iouThreshold = 0.5f;
};


struct DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
};


class YOLO_V8
{
public:
    YOLO_V8();
    ~YOLO_V8();

public:

    template <typename T>
    T vectorProduct(const std::vector<T>& v);

    std::string CreateSession(DL_INIT_PARAM& iParams);
    std::string RunSession(cv::Mat& iImg,std::vector<BoundingBox>& bBoxes);
    std::string WarmUpSession();

    std::string TensorProcess(std::chrono::steady_clock::time_point& starttime_1, cv::Mat& iImg, std::vector<float>& inputTensorValues,std::vector<float>& outputTensorValues,std::vector<BoundingBox>& bBoxes);

    std::string PreProcess(cv::Mat& img, const std::vector<int>& iImgSize, cv::Mat& oImg, int fill_value=114);


    void DrawDetectedObject(cv::Mat &frame, const DL_RESULT &detection);

    // std::vector<std::string> classes{};

    int fps_;                           // FPS value (frames per second)
    int totalFPS = 0;
    int frameCount = 0;
    int avgFPS = 0;

private:
    Ort::Env env;
    Ort::Session* session = nullptr; 
    Ort::RunOptions options;
    
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;
    std::vector<int64_t> inputDims;
    std::vector<int64_t> outputDims;
    size_t inputTensorSize;
    size_t outputTensorSize;
    std::vector<float> inputTensorValues;
    std::vector<float> outputTensorValues;
    // Ort::MemoryInfo memoryInfo;


    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    cv::Point2f scale_factor_;			// Scaling factor for the input frame
    cv::Point2f letterbox_offset_;

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

std::ostream& operator<<(std::ostream& os,const ONNXTensorElementDataType& type);

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v);

}