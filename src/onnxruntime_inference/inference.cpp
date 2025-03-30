#include "inference.h"
#include <numeric>
#include <random>
#include "../dataStructures.h"

namespace OnnxRuntimeInference{

#define RET_OK "Success"

YOLO_V8::YOLO_V8() {
}
YOLO_V8::~YOLO_V8() {
    delete session;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os,const ONNXTensorElementDataType& type)
{
    switch (type)
    {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    os << "undefined";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    os << "float";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    os << "uint8_t";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    os << "int8_t";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    os << "uint16_t";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    os << "int16_t";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    os << "int32_t";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    os << "int64_t";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    os << "std::string";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    os << "bool";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    os << "float16";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    os << "double";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    os << "uint32_t";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    os << "uint64_t";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    os << "float real + float imaginary";
    break;
    case ONNXTensorElementDataType::
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    os << "double real + float imaginary";
    break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    os << "bfloat16";
    break;
    default:
    break;
    }

    return os;
}

template <typename T>
T YOLO_V8::vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}



std::string YOLO_V8::PreProcess(cv::Mat& img, const std::vector<int>& iImgSize, cv::Mat& oImg, int fill_value) 
{   
    
    float aspect_ratio = std::min((float)iImgSize[0] / img.cols, (float)iImgSize[1] / img.rows);
    
    scale_factor_.x = 1.0f / aspect_ratio;
    scale_factor_.y = 1.0f / aspect_ratio;    

    // Compute new dimensions while maintaining the aspect ratio
    int new_w = std::round(img.cols * aspect_ratio);
    int new_h = std::round(img.rows * aspect_ratio);

    // Resize image
    cv::Mat resizedImage;
    cv::resize(img, resizedImage, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

    // Create padded image with fill value
    cv::Mat padded_img(iImgSize[1], iImgSize[0], img.type(), cv::Scalar(fill_value, fill_value, fill_value));

    // Compute integer padding offsets
    letterbox_offset_.x = (iImgSize[0] - new_w) / 2.0f;
    letterbox_offset_.y = (iImgSize[1] - new_h) / 2.0f;

    // Copy resized image to the center
    resizedImage.copyTo(padded_img(cv::Rect(letterbox_offset_.x, letterbox_offset_.y, new_w, new_h)));

    // Convert to RGB and normalize in a single step
    cv::cvtColor(padded_img, oImg, cv::COLOR_BGR2RGB);
    oImg.convertTo(oImg, CV_32F, 1.0f / 255.0f);


    return "OK";  
}


std::string YOLO_V8::CreateSession(DL_INIT_PARAM& iParams){
    std::string Ret = RET_OK;
    rectConfidenceThreshold = iParams.rectConfidenceThreshold;
    iouThreshold = iParams.iouThreshold;
    imgSize = iParams.imgSize;
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,"Yolo");
    Ort::SessionOptions sessionOption;

    // https://fs-eire.github.io/onnxruntime/docs/execution-providers/OpenVINO-ExecutionProvider.html#configuration-options
    std::unordered_map<std::string, std::string> opt;
    opt["device_type"] = "CPU";
    // opt["precision"] = "FP32";
    // opt["num_of_threads"] = "8";
    // opt["num_streams"] = "1";
    sessionOption.AppendExecutionProvider("OpenVINO", opt);
    
    /* OpenVINO™ backend performs hardware, dependent as well as independent optimizations 
    on the graph to infer it on the target hardware with best possible performance.
    In most cases it has been observed that passing the ONNX input graph as is without 
    explicit optimizations would lead to best possible optimizations at kernel level by OpenVINO™. 
    For this reason, it is advised to turn off high level optimizations performed by ONNX Runtime for OpenVINO™ Execution Provider
    */
    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    
    session = new Ort::Session(env, iParams.modelPath.c_str(), sessionOption);

    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session->GetInputCount();
    inputNodeNames.resize(numInputNodes);
    for (size_t i = 0; i < numInputNodes; i++)
    {
        Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
        size_t len = strlen(input_node_name.get()) + 1;  // Ensure correct length
        char* temp_buf = new char[len];  // Allocate exact memory
        strcpy(temp_buf, input_node_name.get());
        inputNodeNames[i] = temp_buf;
    }
    Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0); // get the type information of the first input node
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo(); // get the tensor type and shape information of the first input node
    inputDims = inputTensorInfo.GetShape();
    inputTensorSize = vectorProduct(inputDims);   

    size_t numOutputNodes = session->GetOutputCount();
    outputNodeNames.resize(numOutputNodes);
    for (size_t i = 0; i < numOutputNodes; i++)
    {
        Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
        size_t len = strlen(output_node_name.get()) + 1;  // Ensure correct length
        char* temp_buf = new char[len];  // Allocate exact memory
        strcpy(temp_buf, output_node_name.get());
        outputNodeNames[i] = temp_buf;
    }
    Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0); // get the type information of the first output node
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo(); // get the tensor type and shape information of the first output node
    outputDims = outputTensorInfo.GetShape();
    outputTensorSize = vectorProduct(outputDims); 
    
    inputTensorValues.resize(inputTensorSize);
    outputTensorValues.resize(outputTensorSize);
    
    options = Ort::RunOptions{ nullptr };
    WarmUpSession();
    return RET_OK;
}

std::string YOLO_V8::WarmUpSession(){
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg,blobImage;
    PreProcess(iImg, imgSize, processedImg);
    cv::dnn::blobFromImage(processedImg, blobImage);   

    std::copy(blobImage.begin<float>(),blobImage.end<float>(),inputTensorValues.begin());

    std::vector<Ort::Value> inputTensors; // create a vector to store the input tensors
    std::vector<Ort::Value> outputTensors; // create a vector to store the output tensors

    // create memory information for the input and output tensors in CPU memory
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data() /*return pointer to the data*/, inputTensorSize /*number of elements*/, inputDims.data() /*pointer to the dimensions*/, inputDims.size() /*number of dimensions*/));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data()/*return pointer to the data*/, outputTensorSize /*number of elements*/, outputDims.data() /*pointer to the dimensions*/, outputDims.size() /*number of dimensions*/));

    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(),inputTensors.data(), 1 /*Number of inputs*/, outputNodeNames.data(),outputTensors.data(), 1 /*Number of outputs*/);

    return RET_OK;

}

std::string YOLO_V8::RunSession(cv::Mat& iImg,std::vector<BoundingBox>& bBoxes) {
    auto starttime_1 = std::chrono::steady_clock::now();
    std::string Ret = RET_OK;
    cv::Mat processedImg,blobImage;
    PreProcess(iImg, imgSize, processedImg);    


    cv::dnn::blobFromImage(processedImg, blobImage);
    
    // std::copy(blobImage.begin<float>(),blobImage.end<float>(),inputTensorValues.begin());

    std::memcpy(inputTensorValues.data(), blobImage.data, blobImage.total() * sizeof(float));
    auto starttime_p3 = std::chrono::steady_clock::now();

    TensorProcess(starttime_1, iImg, inputTensorValues,outputTensorValues,bBoxes);

    totalFPS += fps_;
    frameCount++;

    // Prepare FPS text
	std::cout << "FPS: " << fps_ << std::endl;
	std::string fpsText = "FPS: " + std::to_string(fps_); 

	// Put the FPS text in the top left corner of the frame
	cv::putText(iImg, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, 8);
    return Ret;
}

// template<typename N>
std::string YOLO_V8::TensorProcess(std::chrono::steady_clock::time_point& starttime_1, cv::Mat& iImg, std::vector<float>& inputTensorValues,std::vector<float>& outputTensorValues, std::vector<BoundingBox>& bBoxes) {
    std::string Ret = RET_OK;

    std::vector<Ort::Value> inputTensors; // create a vector to store the input tensors
    std::vector<Ort::Value> outputTensors; // create a vector to store the output tensors

    // create memory information for the input and output tensors in CPU memory
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data() /*return pointer to the data*/, inputTensorSize /*number of elements*/, inputDims.data() /*pointer to the dimensions*/, inputDims.size() /*number of dimensions*/));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data()/*return pointer to the data*/, outputTensorSize /*number of elements*/, outputDims.data() /*pointer to the dimensions*/, outputDims.size() /*number of dimensions*/));
    
    auto starttime_2 = std::chrono::steady_clock::now();

    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(),inputTensors.data(), 1 /*Number of inputs*/, outputNodeNames.data(),outputTensors.data(), 1 /*Number of outputs*/);

    auto starttime_3 = std::chrono::steady_clock::now();


    auto outputData = outputTensors.front().GetTensorMutableData<std::remove_pointer<float>::type>();

    int signalResultNum = outputDims[1]; // 84
    int strideNum = outputDims[2]; // 8400
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::Mat rawData;
    rawData = cv::Mat(signalResultNum, strideNum, CV_32F, outputData);
    rawData = rawData.t();

    float* data = (float*)rawData.data;

    for (int i = 0; i < strideNum; i++){
        float* classesScores = data + 4;
        cv::Mat scores(1, signalResultNum-4, CV_32FC1, classesScores);
        cv::Point class_id;
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        if (maxClassScore > rectConfidenceThreshold)
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
        data += signalResultNum;
    }

    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
    for (int i = 0; i < nmsResult.size(); i++){
        int idx = nmsResult[i];
        DL_RESULT result;
        result.classId = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        // DrawDetectedObject(iImg, result);

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

    fps_ = static_cast<int>(1000.0 / (pre_process_time + process_time + post_process_time));

    return Ret;


}

void YOLO_V8::DrawDetectedObject(cv::Mat &frame, const DL_RESULT &detection){
	const cv::Rect &box = detection.box;
	const float &confidence = detection.confidence;
	const int &class_id = detection.classId;
	
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

}


