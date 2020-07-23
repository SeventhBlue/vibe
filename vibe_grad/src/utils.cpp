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

cv::UMat getImgGradient(cv::UMat gray) {
	//cv::Mat k_x_mat = (cv::Mat_<int>(5, 5) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	//cv::Mat k_y_mat = (cv::Mat_<int>(5, 5) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	cv::Mat k_x_mat = (cv::Mat_<int>(5, 5) << -1, 0, 0, 0, 1, -2, 0, 0, 0, 2, -3, 0, 0, 0, 3, -2, 0, 0, 0, 2, -1, 0, 0, 0, 1);
	cv::Mat k_y_mat = (cv::Mat_<int>(5, 5) << -1, -2, -3, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1);

	cv::UMat k_x = cv::UMat(5, 5, CV_16SC1, cv::Scalar::all(0));
	cv::UMat k_y = cv::UMat(5, 5, CV_16SC1, cv::Scalar::all(0));
	k_x_mat.copyTo(k_x);
	k_y_mat.copyTo(k_y);

	cv::UMat gray_x = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);
	cv::UMat gray_y = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);

	filter2D(gray, gray_x, CV_16SC1, k_x);
	filter2D(gray, gray_y, CV_16SC1, k_y);

	// 对矩阵求绝对值
	cv::Mat gray_x_mat = cv::Mat(gray.rows, gray.cols, CV_16SC1);
	cv::Mat gray_y_mat = cv::Mat(gray.rows, gray.cols, CV_16SC1);
	gray_x.copyTo(gray_x_mat);
	gray_y.copyTo(gray_y_mat);
	gray_x_mat = abs(gray_x_mat);
	gray_y_mat = abs(gray_y_mat);
	gray_x_mat.copyTo(gray_x);
	gray_y_mat.copyTo(gray_y);

	cv::UMat gray_grad = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);
	addWeighted(gray_x, 0.5, gray_y, 0.5, 0, gray_grad);

	/*cv::UMat show_grad = cv::UMat::zeros(gray.rows, gray.cols, CV_8UC1);
	gray_grad.convertTo(show_grad, CV_8UC1);
	cv::imshow("gradient", show_grad);
	cv::waitKey(1);*/

	return gray_grad;
}

double getPartImgMeanGradient_1(cv::UMat img_grad, cv::UMat mask, cv::Rect rect) {
	cv::UMat mask_rect = cv::UMat::zeros(img_grad.rows, img_grad.cols, CV_8UC1);
	mask_rect(rect).setTo(255);

	cv::UMat mask_part = cv::UMat::zeros(img_grad.rows, img_grad.cols, CV_8UC1);
	cv::bitwise_and(mask, mask, mask_part, mask_rect);

	cv::UMat mask_erode = cv::UMat::zeros(img_grad.rows, img_grad.cols, CV_8UC1);
	cv::UMat mask_dilate = cv::UMat::zeros(img_grad.rows, img_grad.cols, CV_8UC1);

	//cv::UMat element_11 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
	cv::Mat element_5_mat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::UMat element_5 = cv::UMat(5, 5, CV_16SC1);
	element_5_mat.copyTo(element_5);

	erode(mask_part, mask_erode, element_5);
	dilate(mask_part, mask_dilate, element_5);

	cv::UMat mask_edge = cv::UMat(img_grad.rows, img_grad.cols, CV_8UC1);
	cv::subtract(mask_dilate, mask_erode, mask_edge);

	cv::UMat grad_ret = cv::UMat::zeros(img_grad.rows, img_grad.cols, CV_16SC1);
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

double getPartImgMeanGradient_2(cv::UMat img_grad, cv::UMat mask, cv::Rect rect) {
	cv::UMat mask_rect = cv::UMat::zeros(img_grad.rows, img_grad.cols, CV_8UC1);
	mask_rect(rect).setTo(255);

	cv::UMat mask_part = cv::UMat::zeros(img_grad.rows, img_grad.cols, CV_8UC1);
	cv::bitwise_and(mask, mask, mask_part, mask_rect);

	cv::UMat mask_erode = cv::UMat::zeros(img_grad.rows, img_grad.cols, CV_8UC1);
	cv::Mat element_3_mat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::UMat element_3 = cv::UMat(3, 3, CV_16SC1);
	element_3_mat.copyTo(element_3);
	erode(mask_part, mask_erode, element_3);

	cv::UMat mask_edge = cv::UMat(img_grad.rows, img_grad.cols, CV_8UC1);
	cv::subtract(mask_part, mask_erode, mask_edge);
	/*cv::imshow("edge", mask_edge);
	cv::waitKey(1);*/

	cv::UMat grad_ret = cv::UMat::zeros(img_grad.rows, img_grad.cols, CV_16SC1);
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

void points2Mask(cv::Mat& src, std::vector<cv::Point> mask_points) {

	std::vector<std::vector<cv::Point>> mask_area;
	mask_area.push_back(mask_points);

	polylines(src, mask_area, 1, cv::Scalar(0, 0, 0));
	fillPoly(src, mask_area, cv::Scalar(255, 255, 255));
}

void drawingLine(cv::Mat& img,std::vector<cv::Point> tri)
{
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