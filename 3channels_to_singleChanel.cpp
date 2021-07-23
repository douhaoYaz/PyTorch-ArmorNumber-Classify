#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
	std::vector<cv::String> filenames;
	cv::String folder = "C:\\Users\\49636\\PycharmProjects\\Armor_number_classification\\2021_7_20工业相机获取装甲板号码数据集\\train\\1";
	int index = 1;

	cv::glob(folder, filenames);

	for(size_t i = 0; i < filenames.size(); ++i){
		cv::Mat srcImage = cv::imread(filenames[i]);
		cv::Mat dstImage;
		if(index % 20 == 0)
			std::cout << "nice!" << std::endl;
		if(!srcImage.data)
			std::cerr << "Problem loading image!!!" << std::endl;

		cv::cvtColor(srcImage, dstImage, cv::COLOR_BGR2GRAY);
		cv::String imgName = "C:\\Users\\49636\\PycharmProjects\\Armor_number_classification\\2021_7_23工业相机获取装甲板号码数据集\\train\\1\\" + std::to_string(index) + ".png";
		cv::imwrite(imgName, dstImage);
		index++;
	}
	std::cout << "complete!" << std::endl;
}