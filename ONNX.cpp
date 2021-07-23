#include <iostream>
#include <opencv.hpp>
using namespace std;

int main()
{
    double t_end, t_begin;
    cout << "Hello World!" << endl;

    //Initialize Neurel Network
    cv::dnn::Net net = cv::dnn::readNetFromONNX("/home/ace_guard/ACE-RMVision-Infantry/ACERMVision-gui/parameter/other/Lenet5_v1.onnx");

    //Load image
//    cv::Mat image = cv::imread("/home/ace_guard/ACE-RMVision-Infantry/ACERMVision-gui/parameter/other/986.png");
    cv::Mat image = cv::imread("/home/ace_guard/ACE-RMVision-Infantry/ACERMVision-gui/parameter/other/ONNX模型及图片测试集/2/985.png");
    if(image.empty())
        std::cout << "load image failed\n" << std::endl;

    //ImageProcess
    std::vector<float> mean_value = {0.485, 0.456, 0.406};
    std::vector<float> std_value = {0.229, 0.224, 0.225};

    t_begin = cv::getTickCount();

    cv::Mat dst;
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(image, bgrChannels);
//    for(auto i = 0; i < bgrChannels.size(); i++)
//        bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0/std_value[i], (0.0 - mean_value[i]) / std_value[i]);
    for(auto i = 0; i < bgrChannels.size(); i++){
        bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / 255, -mean_value[i]);
        bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / std_value[i], 0.0);
    }
//    cv::imshow("BLUE",bgrChannels.at(0));
//    cv::imshow("GREEN",bgrChannels.at(1));
//    cv::imshow("RED",bgrChannels.at(2));
//    cv::waitKey(0);
    //TODO
    //运行时间达到42ms，resnet18好像太大了，可以考虑换成Lenet5
    //虽然convertTo函数是个很不错的选择，但仍然可以将其与eigen库对比，选择更快的一种
    cv::merge(bgrChannels, dst);



    cv::Mat blob = cv::dnn::blobFromImage(dst);
    net.setInput(blob);
    cv::Mat pred = net.forward();
    std::cout << "pred:\n" << pred << std::endl;

    t_end = cv::getTickCount();
    std::cout << "spend " << ((t_end - t_begin)/ cv::getTickFrequency() *1000) << " ms"  << std::endl;

    return 0;
}
