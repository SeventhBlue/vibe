#pragma once
#include <opencv2/core/core.hpp>
#include <time.h>

// 遗留物的目标区域以及时间
struct SuspFgMask
{
	cv::Rect rect;
	time_t startTime;
	time_t endTime;
	double meanGradOld;                                         // 没有目标时该区域的平均梯度
	double meanGradNew;                                         // 目标边缘的平局梯度
};

struct callbackP
{
	cv::Mat src;
	std::vector<cv::Point> srcTri;
};

std::string getLocSTDTime();                   // 获取本地格式化时间：  2020/04/24 14:04:20
std::string getLocNameTime();                  // 获取以时间为名的格式：20200424_140420

void getImgGradient(const cv::Mat& gray, cv::Mat& gray_grad);
double getPartImgMeanGradient(const cv::Mat&, cv::Mat&, cv::Rect&);      // 计算图片局部部分目标边缘的平均梯度：mask没有处理，边缘像素等于1

void points2Mask(cv::Mat& src, std::vector<cv::Point> mask_points); // 将点集转成mask
void drawingLine(cv::Mat& img, std::vector<cv::Point> tri);
void onMouse(int event, int x, int y, int flags, void *utsc);
std::vector<cv::Point> getPoints(cv::Mat img);