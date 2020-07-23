#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include"remnants.hpp"
#include"utils.hpp"
#include "ViaDef.h"

class Wrapper
{
public:
	void init(AlgInitSt *paramPtr);
	void runningBuMingWu(cv::UMat* src, cv::UMat* zone, int cameraID, std::vector<SuspFgMask>* vt_ret);
	void resizeImg(cv::UMat* src, cv::UMat* dst, float* wScale, float* hScale);
private:
	struct buMingWuInfo {
		Remnamts* algt;
		time_t startTime;
		int stayTime;
	};
	std::map<int, buMingWuInfo> buMingWu;
	const int _w = 1280;
	const int _h = 720;
};