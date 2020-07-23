#pragma once
#include <opencv2/core/core.hpp>

// 遗留物的目标区域以及时间
struct SuspFgMask
{
	cv::Rect rect;
	time_t startTime;
	time_t endTime;
};

std::string getLocSTDTime();                                  // 获取本地格式化时间：  2020/04/24 14:04:20
std::string getLocNameTime();                                 // 获取以时间为名的格式：20200424_140420