#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//#define NDEBUG
#include <assert.h>

#include "vibe.h"
#include "utils.hpp"

class ApplyVibe {
public:
	~ApplyVibe();
	void initPara(std::string &configPath);
	int initVibe(cv::Mat& initImg);
	void runningVibe(cv::Mat& src, cv::Mat& zone, cv::Size minSize, cv::Size maxSize, std::vector<cv::Rect>& ret);
	void printPara();
private:
	int _deleteDataThr;
	unsigned long _zoneNum = 0;                                        // 监控区域像素的个数
	unsigned long _fgNum = 0;                                          // 检测出的前景的像素个数
	bool _isReturnRet = false;                                         // 是否有结果返回
	cv::Mat _mask;                                                     // vibe检测的出的前景
	cv::Mat _processMask;                                              // 经过图像处理后的前景
	cv::Mat _imgGradOld;                                               // 初始化时图片的梯度
	cv::Mat _imgGradNew;                                               // 当前帧图片的梯度
	int _gradientThr;                                                  // 启用梯度辅助判断是否是遗留物:0表示不启用；大于0表示启用,并作为阈值
	unsigned long _remnStayTimeThr;                                    // 被检测出的遗留物保存时间;防止同一位置多次检测
	bool _remnReInitSam;                                               // 侦测到遗留物是否重新初始化Samples
	float _motionlessThr;                                              // 疑似不明物和当前侦测的区域的IOU区域大于_motionlessThr,定义为静止不动
	unsigned long _stayTimeThr;                                        // 不明物静止不动才侦测为遗留物的阈值；单位为秒
	float _reInitThr;                                                  // 重新初始化样本集，主要用于消除灯光变化等造成的干扰因素
	std::vector<SuspFgMask> _vt_suspRemn;                              // 疑似不明物的存储变量
	int _numberOfSamples = 0;                                          // 每个像素的样本集数量
	int	_matchingThreshold = 0;                                        // 前景像素匹配数量，如果超过此值，则认为是背景像素
	int	_matchingNumber = 0;                                           // 匹配半径，即在该半径内则认为是匹配像素
	int	_updateFactor = 0;                                             // 随机数因子，如果为前景，每个像素有1/_updateFactor几率更新
	vibeModel_Sequential_t *_model = NULL;

	void getDetectResults(std::vector<cv::Rect>& ret);
	// 为每个Rect添加时间等信息
	void rectAddTime(std::vector<cv::Rect>& vt_rect);
	// 获取符合条件的矩形区域
	void getRect(cv::Mat& zone, cv::Size minSize, cv::Size maxSize, std::vector<cv::Rect>& rects);
	void processFgMask();                                              // 用图像处理的方法处理
	void split(std::string& src, std::string& result);
	void readTxt(std::string &configPath);
};