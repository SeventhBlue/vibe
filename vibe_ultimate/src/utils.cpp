#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.hpp"

// 返回格式化时间：2020/04/26 15:09:25
std::string getLocSTDTime() {
	struct tm t;              //tm结构指针
	time_t now;               //声明time_t类型变量
	time(&now);               //获取系统日期和时间
	localtime_s(&t, &now);    //获取当地日期和时间

	std::string time_std = cv::format("%d", t.tm_year + 1900) + "/" + cv::format("%.2d", t.tm_mon + 1) + "/" + cv::format("%.2d", t.tm_mday) + " " +
		cv::format("%.2d", t.tm_hour) + ":" + cv::format("%.2d", t.tm_min) + ":" + cv::format("%.2d", t.tm_sec);
	return time_std;
}

// 返回格式化时间：20200426_150925
std::string getLocNameTime() {
	struct tm t;              //tm结构指针
	time_t now;               //声明time_t类型变量
	time(&now);               //获取系统日期和时间
	localtime_s(&t, &now);    //获取当地日期和时间

	std::string time_name = cv::format("%d", t.tm_year + 1900) + cv::format("%.2d", t.tm_mon + 1) + cv::format("%.2d", t.tm_mday) + "_" +
		cv::format("%.2d", t.tm_hour) + cv::format("%.2d", t.tm_min) + cv::format("%.2d", t.tm_sec);
	return time_name;
}

// 整副图片的梯度
void getImgGradient(const cv::Mat& gray, cv::Mat& gray_grad) {
	cv::Mat x_mat = (cv::Mat_<int>(5, 5) << -1, 0, 0, 0, 1, -2, 0, 0, 0, 2, -3, 0, 0, 0, 3, -2, 0, 0, 0, 2, -1, 0, 0, 0, 1);
	cv::Mat y_mat = (cv::Mat_<int>(5, 5) << -1, -2, -3, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1);

	cv::Mat gray_x = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);
	cv::Mat gray_y = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);

	filter2D(gray, gray_x, CV_16SC1, x_mat);
	filter2D(gray, gray_y, CV_16SC1, y_mat);

	// 对矩阵求绝对值
	gray_x = abs(gray_x);
	gray_y = abs(gray_y);

	addWeighted(gray_x, 0.5, gray_y, 0.5, 0, gray_grad);

	/*cv::Mat show_grad = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);
	gray_grad.convertTo(show_grad, CV_8UC1);
	cv::imshow("gradient", show_grad);
	cv::waitKey(1);*/
}

double getPartImgMeanGradient(const cv::Mat& img_grad, cv::Mat& mask, cv::Rect& rect) {
	cv::Mat mask_rect = cv::Mat::zeros(img_grad.rows, img_grad.cols, CV_8UC1);
	mask_rect(rect).setTo(255);

	cv::Mat mask_part = cv::Mat::zeros(img_grad.rows, img_grad.cols, CV_8UC1);
	cv::bitwise_and(mask, mask, mask_part, mask_rect);

	cv::Mat mask_erode = cv::Mat::zeros(img_grad.rows, img_grad.cols, CV_8UC1);
	cv::Mat element_3 = cv::Mat(3, 3, CV_8UC1);
	cv::erode(mask_part, mask_erode, element_3);

	cv::Mat mask_edge = cv::Mat(img_grad.rows, img_grad.cols, CV_8UC1);
	cv::subtract(mask_part, mask_erode, mask_edge);
	//cv::imshow("edge", mask_edge);
	//cv::waitKey(1);

	cv::Mat grad_ret = cv::Mat::zeros(img_grad.rows, img_grad.cols, CV_16SC1);
	cv::bitwise_and(img_grad, img_grad, grad_ret, mask_edge);

	cv::Scalar p_sum = sum(grad_ret);
	unsigned long non_zeros = countNonZero(grad_ret);
	double mean;
	if (non_zeros == 0) {
		mean = 0;
	}
	else {
		mean = p_sum[0] / non_zeros;
	}
	return mean;
}

void drawingLine(cv::Mat& img, std::vector<cv::Point> tri){
	for (int i = 0; i < tri.size(); i++)
	{
		if (i == (tri.size() - 1))
		{
			line(img, tri[0], tri[i], cv::Scalar(0, 0, 255), 2);
		}
		else
		{
			line(img, tri[i], tri[i + 1], cv::Scalar(0, 0, 255), 2);
		}

	}
}

void points2Mask(cv::Mat& src, std::vector<cv::Point> mask_points) {

	std::vector<std::vector<cv::Point>> mask_area;
	mask_area.push_back(mask_points);

	polylines(src, mask_area, 1, cv::Scalar(0, 0, 0));
	fillPoly(src, mask_area, cv::Scalar(255, 255, 255));
}

void onMouse(int event, int x, int y, int flags, void *utsc)
{
	callbackP cp = *(callbackP*)utsc;  // 先转换类型，再取数据

	if (event == cv::EVENT_LBUTTONUP)      // 响应鼠标左键事件
	{
		circle((*(callbackP*)utsc).src, cv::Point(x, y), 2, cv::Scalar(255, 255, 255), 4);  //标记选中点
		imshow("wait ", (*(callbackP*)utsc).src);
		(*(callbackP*)utsc).srcTri.push_back(cv::Point(x, y));
		std::cout << "x:" << x << " " << "y:" << y << std::endl;
	}
}

std::vector<cv::Point> getPoints(cv::Mat img) {
	callbackP utsc;
	utsc.src = img.clone();
	cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
	imshow("src", utsc.src);
	cv::setMouseCallback("src", onMouse, (void*)&utsc);  //类型转换
	cv::waitKey();
	cv::destroyAllWindows();
	return utsc.srcTri;
}