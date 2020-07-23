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