#include <iostream>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "remnants.hpp"

// 构造函数：初始化基本参数
Remnamts::Remnamts(int cameraID,
					float thr,
					cv::Size minSize,
					cv::Size maxSize,
					uint64_t dataTime,
					float motionlessThr,
					unsigned long stayTimeThr,
					bool remnReInitSam,
					unsigned long remnStayTimeThr,
					unsigned int gradientThr) {
	_cameraID = cameraID;
	_reInitThr = thr;
	_minSize = minSize;
	_maxSize = maxSize;
	_dataTime = dataTime;
	_motionlessThr = motionlessThr;
	_stayTimeThr = stayTimeThr;
	_remnReInitSam = remnReInitSam;
	_remnStayTimeThr = remnStayTimeThr;
	_gradientThr = gradientThr;
}

// 构建图片样本集
void Remnamts::initSamples(cv::UMat& img) {
	_vibe.initSamples(img);
	_fgMask = cv::UMat::zeros(img.rows, img.cols, CV_8UC1);
	_zoneFgMask = cv::UMat::zeros(img.rows, img.cols, CV_8UC1);
	_img_h = img.size().height;
	_img_w = img.size().width;
	_fgNum = 0;
	_vt_ret.clear();
	if(_gradientThr){
		_imgGradPre = getImgGradient(img);
	}
}

// 重新构建图片图片样本集
void Remnamts::reInitSamples(cv::UMat& img) {
	_vibe.reInitSamples(img);
	_fgMask.setTo(0);
	_zoneFgMask.setTo(0);
	_img_h = img.size().height;
	_img_w = img.size().width;
	_fgNum = 0;
	_zoneFgNum = 0;
	_vt_ret.clear();
	_vt_suspRemn.clear();
	if (_gradientThr) {
		_imgGradPre = getImgGradient(img);
	}
}

void Remnamts::reInitPartSamples(cv::UMat& img, std::vector<SuspFgMask> vt) {
	_vibe.reInitPartSamples(img, vt);
}

// 侦测可能成为遗留物的前景区域
void Remnamts::findFgMask(cv::UMat& img,cv::UMat& zone) {
	//std::cout << "findFgMask_start:" << _vt_remnFgMask.size() << std::endl;
	_zoneNum = cv::countNonZero(zone);
	if (((float)_zoneFgNum / _zoneNum) >= _reInitThr) {
		reInitSamples(img);
		_vt_remn.clear();
		std::cout << "大范围灯光变化或遮挡监控区域导致重新初始化！---------------" << std::endl;
	}
	else if (_remnReInitSam && (_vt_ret.size() > 0)) {
		reInitSamples(img);
		//reInitPartSamples(img, _vt_ret);
		std::cout << "侦测到遗留物导致局部重新初始化！---------------" << std::endl;
	}
	else {
		_vibe.findFgMask(img);
		_fgNum = _vibe.getFgNum();
		_fgMask = _vibe.getFGMask(false);
		cv::bitwise_and(_fgMask, zone, _zoneFgMask);
		_zoneFgNum = cv::countNonZero(_zoneFgMask);
		if (_gradientThr) {
			_imgGradCur = getImgGradient(img);
		}
	}
}

cv::UMat Remnamts::getFGMask() {
	return _zoneFgMask;
}

unsigned long Remnamts::getFgNum() {
	return _zoneFgNum;
}

// 获取外接矩形
std::vector<cv::Rect> Remnamts::getCircumscribedRect() {
	std::vector<cv::Rect> vt_rect;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(_zoneFgMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());

	for (int i = 0; i < contours.size(); i++) {
		cv::Rect boundRect = cv::boundingRect((cv::Mat)contours[i]);        //查找每个轮廓的外接矩形
		if ((boundRect.width <= _maxSize.width) && (boundRect.width >= _minSize.width) && (boundRect.height <= _maxSize.height) && (boundRect.height >= _minSize.height)) {
			vt_rect.push_back(boundRect);
		}
	}
	return vt_rect;
}

std::vector<SuspFgMask> Remnamts::getRemn(std::vector<SuspFgMask> vt_suspRemn) {
	// 大于_stayTimeThr的不明物挑选出来
	_vt_suspRemn.clear();
	for (int i = 0; i < vt_suspRemn.size(); i++) {
		_vt_suspRemn.push_back(vt_suspRemn[i]);
		if ((vt_suspRemn[i].endTime - vt_suspRemn[i].startTime) >= _stayTimeThr) {
			if (_gradientThr) {
				if (std::abs(vt_suspRemn[i].meanGradCur - vt_suspRemn[i].meanGradPre) > std::abs(_gradientThr)) {
					_vt_ret.push_back(vt_suspRemn[i]);
				}
			}
			else {
				_vt_ret.push_back(vt_suspRemn[i]);
			}
		}
	}

	time_t now;
	// 更新时间
	for (int i = 0; i < _vt_remn.size(); i++) {
		_vt_remn[i].endTime = time(&now);
	}
	// 排除同一位置多次侦测产生的误测
	std::vector<SuspFgMask> tmp_vt_ret;
	std::vector<SuspFgMask> tmp_vt_remn;
	for (int i = 0; i < _vt_ret.size(); i++) {
		if (_vt_remn.size() == 0) {
			std::cout << 3 << std::endl;
			tmp_vt_ret.push_back(_vt_ret[i]);
		}
		else {
			for (int j = 0; j < _vt_remn.size(); j++) {
				cv::Rect is_rect = _vt_ret[i].rect | _vt_remn[j].rect;
				cv::Rect un_rect = _vt_ret[i].rect & _vt_remn[j].rect;
				double IOU = un_rect.area()*1.0 / is_rect.area();
				if (IOU >= _motionlessThr) {
					if ((_vt_remn[j].endTime - _vt_remn[j].startTime) < _remnStayTimeThr) {
						_vt_remn[j].endTime = time(&now);
						tmp_vt_remn.push_back(_vt_remn[j]);
					}
					else {
						tmp_vt_ret.push_back(_vt_ret[i]);
					}
				}
				else {
					tmp_vt_ret.push_back(_vt_ret[i]);
				}
			}
		}
	}

	if (_vt_ret.size() > 0) {
		_vt_remn.clear();
	}
	for (int i = 0; i < tmp_vt_remn.size(); i++) {
		_vt_remn.push_back(tmp_vt_remn[i]);
	}
	for (int i = 0; i < tmp_vt_ret.size(); i++) {
		time_t startTime = tmp_vt_ret[i].startTime;
		_vt_remn.push_back(SuspFgMask{ tmp_vt_ret[i].rect,startTime,time(&now),tmp_vt_ret[i].meanGradPre, tmp_vt_ret[i].meanGradCur });
	}
	return tmp_vt_ret;
}

// 获取结果
std::vector<SuspFgMask> Remnamts::getResults() {
	_vt_ret.clear();
	processFgMask();
	std::vector<cv::Rect> vt_rect = getCircumscribedRect();
	std::vector<SuspFgMask> vt_suspRemn = rectAddTime(vt_rect);
	std::vector<SuspFgMask> ret = getRemn(vt_suspRemn);

	return ret;
}

std::vector<SuspFgMask> Remnamts::rectAddTime(std::vector<cv::Rect> vt_rect) {
	time_t now;
	std::vector<SuspFgMask> vt_suspFgMask;
	// 更新时间；
	for (int i = 0; i < _vt_suspRemn.size(); i++) {
		_vt_suspRemn[i].endTime = time(&now);
	}

	int flag = 0;                                                  // 标识位：是否是原来的遗留物
	for (int i = 0; i < vt_rect.size(); i++) {
		for (int j = 0; j < _vt_suspRemn.size(); j++) {
			cv::Rect is_rect = vt_rect[i] | _vt_suspRemn[j].rect;
			cv::Rect un_rect = vt_rect[i] & _vt_suspRemn[j].rect;
			double IOU = un_rect.area()*1.0 / is_rect.area();
			if (IOU >= _motionlessThr) {                          // 通过矩形间的IOU区分是否是同一物
				if (_gradientThr) {
					double gradCur = getPartImgMeanGradient_2(_imgGradCur, _fgMask, vt_rect[i]);
					vt_suspFgMask.push_back(SuspFgMask{ vt_rect[i], _vt_suspRemn[j].startTime, time(&now), _vt_suspRemn[j].meanGradPre, gradCur });
					flag = 1;
					break;
				}
				else {
					vt_suspFgMask.push_back(SuspFgMask{ vt_rect[i], _vt_suspRemn[j].startTime, time(&now), 0, 0 });
					flag = 1;
					break;
				}
			}
		}
		if (!flag) {                                           // 不是原来的遗留物时添加新遗留物
			if (_gradientThr) {
				double gradPre = getPartImgMeanGradient_2(_imgGradPre, _fgMask, vt_rect[i]);
				double gradCur = getPartImgMeanGradient_2(_imgGradCur, _fgMask, vt_rect[i]);
				vt_suspFgMask.push_back(SuspFgMask{ vt_rect[i], time(&now), 0, gradPre, gradCur });
				flag = 0;
			}
			else {
				vt_suspFgMask.push_back(SuspFgMask{ vt_rect[i], time(&now), 0, 0, 0 });
				flag = 0;
			}
		}
	}

	return vt_suspFgMask;
}

void Remnamts::processFgMask() {
	cv::Mat element_3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	//cv::morphologyEx(_zoneFgMask, _zoneFgMask, cv::MORPH_OPEN, element_3);   // 开运算
	cv::Mat element_9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

	cv::UMat element_3_umat = cv::UMat(3, 3, CV_8UC1);
	cv::UMat element_9_umat = cv::UMat(9, 9, CV_8UC1);

	element_3.copyTo(element_3_umat);
	element_9.copyTo(element_9_umat);

	cv::erode(_zoneFgMask, _zoneFgMask, element_3_umat);          // 腐蚀
	cv::dilate(_zoneFgMask, _zoneFgMask, element_9_umat);         // 膨胀
}

void Remnamts::setMaxSize(cv::Size size) {
	//_vibe.setMaxSize(size);
	_maxSize = size;
}

void Remnamts::setMinSize(cv::Size size) {
	//_vibe.setMinSize(size);
	_minSize = size;
}

void Remnamts::setRadius(int r) {
	_vibe.setRadius(r);
}

void Remnamts::setReqMathces(int m) {
	_vibe.setReqMathces(m);
}

void Remnamts::setSubsamplingFactor(int f) {
	_vibe.setSubsamplingFactor(f);
}

void Remnamts::setGradientThr(int grad) {
	_gradientThr = grad;
}