#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include"wrapper.h"
#include"utils.hpp"
#include"bumingwu.h"

void *InitialBuMingWuFunct(AlgInitSt *paramPtr) {
	Wrapper* buMingWu = new Wrapper;
	buMingWu->init(paramPtr);
	return (void*)buMingWu;
}

void ReleaseBuMingWuFunct(void *handle) {
	Wrapper* buMingWu = (Wrapper*)handle;
	if (!buMingWu) {
		return;
	}
	else {
		delete buMingWu;
		return;
	}
}

int DoBuMingWuObjectFunct(InDataSt *inDataPtr, void *handle, ObjectBoxSurvSt *objBoxeList) {
	// 转格式：将其他格式转成UMat
	cv::Mat rgb;
	if (inDataPtr->imgInform->imgFormat == 1 && inDataPtr->imgInform->codeformat == 1) {
		cv::Mat bgr(inDataPtr->imgInform->height, inDataPtr->imgInform->width, CV_8UC3);
		int len = inDataPtr->imgInform->width * inDataPtr->imgInform->height * 3;
		memcpy(bgr.data, inDataPtr->imgInform->dataPtr, len);
		cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
	}
	else if (inDataPtr->imgInform->imgFormat == 0 && inDataPtr->imgInform->codeformat == 0) {
		int len = inDataPtr->imgInform->width * inDataPtr->imgInform->height * 3 / 2;
		cv::Mat yuvImg;
		yuvImg.create(inDataPtr->imgInform->height * 3 / 2, inDataPtr->imgInform->width, CV_8UC1);
		memcpy(yuvImg.data, inDataPtr->imgInform->dataPtr, len);
		cv::cvtColor(yuvImg, rgb, cv::COLOR_YUV2RGB_I420);
	}
	cv::Mat gray = cv::Mat::zeros(inDataPtr->imgInform->height, inDataPtr->imgInform->width, CV_8UC1);
	cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
	cv::UMat gray_umat = cv::UMat::zeros(inDataPtr->imgInform->height, inDataPtr->imgInform->width, CV_8UC1);
	gray.copyTo(gray_umat);

	Wrapper* buMingWu = (Wrapper*)handle;
	double start_time = (double)cv::getTickCount();
	float wScale = 0;
	float hScale = 0;
	cv::UMat dst;
	buMingWu->resizeImg(&gray_umat, &dst, &wScale, &hScale);


	// 获取监控区域
	std::vector<cv::Point> points;
	cv::Mat zone;
	if ((gray.size().width > 1280) || gray.size().height > 720) {
		zone = cv::Mat::zeros(720, 1280, CV_8UC1);
	}
	else {
		zone = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);
	}
	for (int i = 0; i < inDataPtr->numPoly; i++) {
		points.clear();
		for (int j = 0; j < inDataPtr->polyPtr[i].numPts; j++) {
			if (inDataPtr->polyPtr[i].taskType == 11) {
				int x = (int)(inDataPtr->polyPtr[i].polyPts[j].x * gray.size().width / wScale);
				int y = (int)(inDataPtr->polyPtr[i].polyPts[j].y * gray.size().height / hScale);
				points.push_back(cv::Point(x, y));
			}
		}
		points2Mask(zone, points);
	}
	cv::UMat zone_umat;
	if ((gray.size().width > 1280) || gray.size().height > 720) {
		zone_umat = cv::UMat::zeros(720, 1280, CV_8UC1);
	}
	else {
		zone_umat = cv::UMat::zeros(gray.rows, gray.cols, CV_8UC1);
	}
	zone.copyTo(zone_umat);
	std::vector<SuspFgMask> vt_ret;
	buMingWu->runningBuMingWu(&dst, &zone_umat, inDataPtr->imgInform->cameraID, &vt_ret);

	if (vt_ret.size() > 0) {
		cv::rectangle(dst, vt_ret[0].rect, cv::Scalar(0, 0, 255), 2);
		std::string img_name = "img_" + getLocNameTime() + ".png";
		std::string mask_name = "mask_" + getLocNameTime() + ".png";
		cv::imwrite(img_name, dst);
	}

	// 返回结果
	objBoxeList->cameraID = inDataPtr->imgInform->cameraID;
	for (int i = 0; i < inDataPtr->numPoly; i++) {
		if (inDataPtr->polyPtr->taskType == 11) {
			objBoxeList[i].taskID = inDataPtr->polyPtr[i].taskID;
			objBoxeList[i].taskType = inDataPtr->polyPtr[i].taskType;
			objBoxeList[i].numObjs = vt_ret.size();
			for (int j = 0; j < objBoxeList[i].numObjs; j++) {
				float x = vt_ret[j].rect.x * wScale / gray.size().width;
				float y = vt_ret[j].rect.y * hScale / gray.size().height;
				float w = vt_ret[j].rect.width * wScale / gray.size().width;
				float h = vt_ret[j].rect.height * hScale / gray.size().height;
				objBoxeList[i].objsPtr[j].x = x;
				objBoxeList[i].objsPtr[j].y = y;
				objBoxeList[i].objsPtr[j].w = w;
				objBoxeList[i].objsPtr[j].h = h;
			}
		}
	}

	return 1;
}

void DeleteBuMingWuTaskFunct(InDataSt *inDataPtr, void *handle) {

}