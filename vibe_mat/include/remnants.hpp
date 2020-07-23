#pragma once
#include <opencv2/core/core.hpp>
#include "vibe.hpp"
#include "utils.hpp"

class Remnamts {
public:
	Remnamts(int, float, cv::Size, cv::Size, cv::Mat&, uint64_t, float, unsigned long, bool, unsigned long);     // 构造函数
	void initSamples(cv::Mat& img);                                   // 初始化samples
	void reInitSamples(cv::Mat& img);                                 // 重新初始化样本集
	void reInitPartSamples(cv::Mat& img, std::vector<cv::Rect> vt);   // 局部更新samples样本集；更新的部分为_vt_rect
	void findFgMask(cv::Mat& img);									  // 侦测可能成为不明物的区域
	cv::Mat getFGMask();                                              // 返回可能成为不明物的区域
	unsigned long getFgNum();                                         // 返回可能成为不明物的像素点个数
	std::vector<cv::Rect> getRect();                                  // 返回不明物区域

	void setReqMathces(int);
	void setRadius(int);
	void setSubsamplingFactor(int);
	void setMinSize(cv::Size);
	void setMaxSize(cv::Size);
private:
	unsigned long _remnStayTimeThr;                                   // 被检测出的遗留物保存时间;防止同一位置多次检测
	bool _remnReInitSam;                                              // 侦测到遗留物是否重新初始化Samples
	float _motionlessThr;                                             // 疑似不明物和当前侦测的区域的IOU区域大于_motionlessThr,定义为静止不动
	unsigned long _stayTimeThr;                                       // 不明物静止不动才侦测为遗留物的阈值；单位为秒
	float _reInitThr;                                                 // 重新初始化样本集，主要用于消除灯光变化等造成的干扰因素
	vibe_hw::VIBE _vibe;                                              // vibe算法对象
	int _cameraID;                                                    // 摄像头的ID，目前没有启用
	cv::Mat _zone;                                                    // 监控区域的Mask
	uint64_t _dataTime;                                               // 布控开启的时间
	cv::Mat _fgMask;                                                  // 疑似不明物的Mask
	cv::Mat _zoneFgMask;                                              // 监控区域疑似不明物的Mask
	unsigned long _fgNum;                                             // _fgMask非零像素的个数
	unsigned long _zoneFgNum;                                         // _zoneFgMask非零像素的个数
	unsigned long _zoneNum;                                           // _zone非零像素的个数
	std::vector<cv::Rect> _vt_rect;                                   // 侦测为不明物的存储区域
	int _img_w;
	int _img_h;
	cv::Size _minSize;                                                // 不明物的最小高，宽
	cv::Size _maxSize;                                                // 不明物的最大高，宽
	std::vector<SuspFgMask> _vt_suspFgMask;                           // 疑似不明物的存储变量
	std::vector<SuspFgMask> _vt_remnFgMask;                           // 已经侦测为遗留物的区域存储某一段时间

	void processFgMask();                                             // 用图像处理的方法处理不明物区域
};