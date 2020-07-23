#pragma once
#include <opencv2/core/core.hpp>

// 遗留物的目标区域以及时间
struct SuspFgMask
{
	cv::Rect rect;
	time_t startTime;
	time_t endTime;
	double meanGradPre;                                         // 没有目标时该区域的平均梯度
	double meanGradCur;                                         // 目标边缘的平局梯度
};

std::string getLocSTDTime();                   // 获取本地格式化时间：  2020/04/24 14:04:20
std::string getLocNameTime();                  // 获取以时间为名的格式：20200424_140420

cv::UMat getImgGradient(cv::UMat);                                  // 获取图片的梯度
double getPartImgMeanGradient_1(cv::UMat, cv::UMat, cv::Rect);      // 计算图片局部部分目标边缘的平均梯度;mask经过腐蚀膨胀，边缘像素大于3
double getPartImgMeanGradient_2(cv::UMat, cv::UMat, cv::Rect);      // 计算图片局部部分目标边缘的平均梯度：mask没有处理，边缘像素等于1

void points2Mask(cv::Mat& src, std::vector<cv::Point> mask_points); // 将点集转成mask
void drawingLine(cv::Mat& img, std::vector<cv::Point> tri);