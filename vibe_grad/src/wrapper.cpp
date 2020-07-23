#include "wrapper.h"
void Wrapper::init(AlgInitSt *paramPtr) {

}

void Wrapper::runningBuMingWu(cv::UMat* src, cv::UMat* zone, int cameraID, std::vector<SuspFgMask>* vt_ret) {
	// 更新时间
	time_t now;
	std::map<int, buMingWuInfo>::iterator i;
	for (i = buMingWu.begin(); i != buMingWu.end(); i++)
	{
		i->second.stayTime = time(&now) - i->second.startTime;
	}

	unsigned long fgNum;
	cv::UMat mask;
	std::map<int, buMingWuInfo>::iterator iter = buMingWu.find(cameraID);
	if (iter != buMingWu.end()) {
		iter->second.algt->findFgMask(*src, *zone);
		mask = iter->second.algt->getFGMask();
		fgNum = iter->second.algt->getFgNum();
		(*vt_ret) = iter->second.algt->getResults();
		iter->second.stayTime = time(&now) - iter->second.startTime;
		cv::imshow("mask33", mask);
	}
	else {
		Remnamts* tmp = new Remnamts(0, 0.8, cv::Size(20, 20), cv::Size(600, 400), 123, 0.9, 2, true, 120, 200);
		tmp->initSamples(*src);
		buMingWu[cameraID] = buMingWuInfo{ tmp, time(&now),0 };
	}

	//cv::imshow("src22", *src);
	//cv::imshow("zone33", *zone);
	cv::waitKey(1);

	//cv::Mat tmp = cv::Mat::zeros(src->size().height, src->size().width, CV_8UC1);
	//src->copyTo(tmp);

	/*if ((*vt_ret).size() > 0) {
		cv::rectangle(*src, (*vt_ret)[0].rect, cv::Scalar(0, 0, 255), 2);
		std::string img_name = "img_" + getLocNameTime() + ".png";
		std::string mask_name = "mask_" + getLocNameTime() + ".png";
		cv::imwrite(img_name, *src);
		cv::imwrite(mask_name, mask);
	}*/



	// 释放资源
	int relea = -1;
	std::map<int, buMingWuInfo>::iterator ii;
	for (ii = buMingWu.begin(); ii != buMingWu.end(); ii++)
	{
		if (ii->second.stayTime > 300) {      // 如果三分钟后还没有新图片传入，就释放资源
			delete ii->second.algt;
			ii->second.algt = nullptr;
			relea = ii->first;
		}
	}
	buMingWu.erase(relea);
}

void Wrapper::resizeImg(cv::UMat* src, cv::UMat* dst, float* wScale, float* hScale) {
	if ((src->size().width > _w) || (src->size().height > _h)) {
		cv::resize(*src, *dst, cv::Size(_w, _h), 0, 0, cv::INTER_AREA);
		*wScale = src->size().width*1.0 / _w;
		*hScale = src->size().height*1.0 / _h;
	}
	else {
		(*dst) = (*src).clone();
		*wScale = 1.0;
		*hScale = 1.0;
	}
}