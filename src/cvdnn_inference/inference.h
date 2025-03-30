#include <fstream>
#include <vector>
#include <string>
#include <random>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "../dataStructures.h"


namespace CvdnnInference{
    
struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

class Inference
{
public:
    Inference(const cv::String& modelconfig,const cv::String& modelweights, const cv::Size &modelInputShape = {640, 640});
    std::vector<Detection> runInference(const cv::Mat &input,std::vector<BoundingBox>& bBoxes);
    void DrawDetectedObject(const cv::Mat &frame, const Detection &detection);
    cv::Mat letterbox(const cv::Mat& img, const cv::Size& new_size , int fill_value=114);


private:
    void loadOnnxNetwork();

    cv::String  modelConfig;
    cv::String  modelWeights;

    std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    cv::Size modelShape;

    float modelScoreThreshold = 0.5f;
    float modelNMSThreshold = 0.5f;

	cv::Point2f scale_factor_;
	cv::Point2f letterbox_offset_;

    cv::dnn::Net net;
};
}