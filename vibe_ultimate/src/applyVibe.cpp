#include "applyVibe.h"

void ApplyVibe::getRect(cv::Mat& zone, cv::Size minSize, cv::Size maxSize, std::vector<cv::Rect>& rects) {
	processFgMask();
	cv::bitwise_and(_processMask, zone, _processMask);
	_fgNum = cv::countNonZero(_processMask);
	
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(_processMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());

	for (int i = 0; i < contours.size(); i++) {
		cv::Rect boundRect = cv::boundingRect((cv::Mat)contours[i]);        //查找每个轮廓的外接矩形
		if ((boundRect.width <= maxSize.width) && (boundRect.width >= minSize.width) && (boundRect.height <= maxSize.height) && (boundRect.height >= minSize.height)) {
			rects.push_back(boundRect);
		}
	}
}

void ApplyVibe::processFgMask() {
	cv::Mat element_3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat element_9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

	cv::erode(_mask, _processMask, element_3);          // 腐蚀
	cv::dilate(_processMask, _processMask, element_9);         // 膨胀
}

void ApplyVibe::runningVibe(cv::Mat& src, cv::Mat& zone, cv::Size minSize, cv::Size maxSize, std::vector<cv::Rect>& ret) {
	_zoneNum = cv::countNonZero(zone);

	if ((_fgNum * 1.0 / _zoneNum) >= _reInitThr) {
		libvibeModel_Sequential_Free(_model);
		initVibe(src);
		std::cout << "[BuMingWu INFO] Large-scale light changes or obstruction of the monitoring area lead to reinitialization!" << std::endl;
	}
	else if (_remnReInitSam && _isReturnRet) {
		libvibeModel_Sequential_Free(_model);
		initVibe(src);
		_isReturnRet = false;
		std::cout << "[BuMingWu INFO] Detection of BuMingWu leads to partial reinitialization!" << std::endl;
	}
	else {
		libvibeModel_Sequential_Segmentation_8u_C1R(_model, src.data, _mask.data);
		libvibeModel_Sequential_Update_8u_C1R(_model, src.data, _mask.data);
		std::vector<cv::Rect> rects;
		getRect(zone, minSize, maxSize, rects);
		if (_gradientThr) {
			getImgGradient(src, _imgGradNew);
		}
		rectAddTime(rects);
		getDetectResults(ret);
		//cv::imshow("mask",_processMask);
		//cv::waitKey(1);
	}
}

void ApplyVibe::getDetectResults(std::vector<cv::Rect>& ret) {
	time_t now;
	std::vector<int> deleteData; // 待删除数据
	for (int i = 0; i < _vt_suspRemn.size(); i++) {
		if ((time(&now) - _vt_suspRemn[i].endTime) > _deleteDataThr) {
			deleteData.push_back(i);
			continue;
		}
		if (abs(_vt_suspRemn[i].endTime - _vt_suspRemn[i].startTime) >= _stayTimeThr) {
			if (_gradientThr) {    // 启用梯度过滤数据
				if (std::abs(_vt_suspRemn[i].meanGradNew - _vt_suspRemn[i].meanGradOld) > std::abs(_gradientThr)) {
					ret.push_back(_vt_suspRemn[i].rect);
					_isReturnRet = true;
				}
			}
			else {                 // 不启用梯度过滤数据
				_vt_suspRemn[i].startTime = time(&now);
				_vt_suspRemn[i].endTime = time(&now);
				ret.push_back(_vt_suspRemn[i].rect);
				_isReturnRet = true;
			}
		}
	}

	// 删除数据
	for (int i = (deleteData.size() - 1); i >= 0; i--) {
		_vt_suspRemn.erase(_vt_suspRemn.begin() + deleteData[i]);
	}

	// 如果检测到目标就更新背景模型，那么需要把检测到目标全部返回，
	// 防止漏检，但是这会造成没有满足条件的目标被返回
	if (_remnReInitSam && _isReturnRet) {
		ret.clear();
		for (int i = 0; i < _vt_suspRemn.size(); i++) {
			ret.push_back(_vt_suspRemn[i].rect);
		}
	}
}

void ApplyVibe::rectAddTime(std::vector<cv::Rect>& vt_rect) {
	time_t now;
	std::vector<SuspFgMask> vt_suspFgMask;

	int flag = 0;                                                  // 标识位：是否是原来的遗留物
	for (int i = 0; i < vt_rect.size(); i++) {
		for (int j = 0; j < _vt_suspRemn.size(); j++) {
			cv::Rect is_rect = vt_rect[i] | _vt_suspRemn[j].rect;
			cv::Rect un_rect = vt_rect[i] & _vt_suspRemn[j].rect;
			double IOU = un_rect.area()*1.0 / is_rect.area();
			if (IOU >= _motionlessThr) {                          // 通过矩形间的IOU区分是否是同一物
				if (_gradientThr) {                               // 启用梯度过滤
					double gradNew = getPartImgMeanGradient(_imgGradNew, _mask, vt_rect[i]);
					_vt_suspRemn[j].rect = vt_rect[i];
					_vt_suspRemn[j].endTime = time(&now);
					_vt_suspRemn[j].meanGradNew = gradNew;
					flag = 1;
					break;
				}
				else {
					_vt_suspRemn[j].rect = vt_rect[i];
					_vt_suspRemn[j].endTime = time(&now);
					flag = 1;
					break;
				}
			}
		}
		if (!flag) {                                           // 添加新的遗留物
			if (_gradientThr) {
				double gradOld = getPartImgMeanGradient(_imgGradOld, _mask, vt_rect[i]);
				double gradNew = getPartImgMeanGradient(_imgGradNew, _mask, vt_rect[i]);
				vt_suspFgMask.push_back(SuspFgMask{ vt_rect[i], time(&now), time(&now), gradOld, gradNew });
				flag = 0;
			}
			else {
				vt_suspFgMask.push_back(SuspFgMask{ vt_rect[i], time(&now), time(&now), 0, 0 });
				flag = 0;
			}
		}
	}

	for (int i = 0; i < vt_suspFgMask.size(); i++) {
		_vt_suspRemn.push_back(vt_suspFgMask[i]);
	}
}

int ApplyVibe::initVibe(cv::Mat& initImg) {
	_fgNum = 0;
	_vt_suspRemn.clear();
	std::vector<SuspFgMask>(_vt_suspRemn).swap(_vt_suspRemn);
	_mask = cv::Mat::zeros(initImg.rows, initImg.cols, CV_8UC1);
	_processMask = cv::Mat::zeros(initImg.rows, initImg.cols, CV_8UC1);
	if (_gradientThr) {
		_imgGradNew = cv::Mat::zeros(initImg.rows, initImg.cols, CV_16SC1);
		_imgGradOld = cv::Mat::zeros(initImg.rows, initImg.cols, CV_16SC1);
		getImgGradient(initImg, _imgGradOld);
	}
	_model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
	assert(_numberOfSamples != 0);
	assert(_matchingThreshold != 0);
	assert(_matchingNumber != 0);
	assert(_updateFactor != 0);
	if (_numberOfSamples != 0 && _matchingThreshold != 0 && _matchingNumber != 0 && _updateFactor != 0) {
		_model->numberOfSamples = _numberOfSamples;
		_model->matchingThreshold = _matchingThreshold;
		_model->matchingNumber = _matchingNumber;
		_model->updateFactor = _updateFactor;
	}
	else {
		std::cout << "Incorrect parameter setting may result in inaccurate detection!" << std::endl;
	}
	libvibeModel_Sequential_AllocInit_8u_C1R(_model, initImg.data, initImg.size().width, initImg.size().height);
	return 1;
}

void ApplyVibe::initPara(std::string &configPath) {
	readTxt(configPath);
}

void ApplyVibe::readTxt(std::string &configPath) {
	std::ifstream configTxt(configPath.c_str());
	std::string line;
	std::vector<std::string> resultVec;
	assert(configTxt.is_open());
	while (getline(configTxt, line))
	{
		std::string result;
		split(line, result);
		resultVec.push_back(result);
	}
	configTxt.close();
	assert(resultVec.size() > 3);
	_numberOfSamples = std::atof(resultVec[0].c_str());
	_matchingThreshold = std::atof(resultVec[1].c_str());
	_matchingNumber = std::atof(resultVec[2].c_str());
	_updateFactor = std::atof(resultVec[3].c_str());
	_reInitThr = std::atof(resultVec[4].c_str());
	_remnReInitSam = std::atof(resultVec[5].c_str());
	_gradientThr = std::atof(resultVec[6].c_str());
	_motionlessThr = std::atof(resultVec[7].c_str());
	_stayTimeThr = std::atof(resultVec[8].c_str());
	_deleteDataThr = std::atof(resultVec[9].c_str());
	printPara();
}
void ApplyVibe::printPara() {
	std::cout << "_numberOfSamples:  " <<_numberOfSamples << std::endl;
	std::cout << "_matchingThreshold:" << _matchingThreshold << std::endl;
	std::cout << "_matchingNumber:   " << _matchingNumber << std::endl;
	std::cout << "_updateFactor:     " << _updateFactor << std::endl;
	std::cout << "_reInitThr:        " << _reInitThr << std::endl;
	std::cout << "_remnReInitSam:    " << _remnReInitSam << std::endl;
	std::cout << "_gradientThr:      " << _gradientThr << std::endl;
	std::cout << "_motionlessThr:    " << _motionlessThr << std::endl;
	std::cout << "_stayTimeThr:      " << _stayTimeThr << std::endl;
	std::cout << "_deleteDataThr:    " << _deleteDataThr << std::endl;
}

void ApplyVibe::split(std::string& src, std::string& result)
{
	result.clear();
	std::string::iterator iter = std::find(src.begin(), src.end(), ':');
	for (; iter != src.end(); ++iter) {
		if (*iter == ':') {
			continue;
		}
		result.push_back(*iter);
	}
}

ApplyVibe::~ApplyVibe() {
	libvibeModel_Sequential_Free(_model);
}