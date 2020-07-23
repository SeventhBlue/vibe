#pragma once
#include <opencv2/core/core.hpp>

namespace vibe
{
	class VIBE
	{
	public:
		typedef cv::Vec<uchar, 20> Vec20u;
		VIBE(int num_sam, int min_match, int radiu, int rand_sam, int fgCount, cv::Size minSize, cv::Size maxSize);
		void initBGModel(cv::Mat img);    // 初始化背景模型
		void update(cv::Mat img);       // 追求速度版本
		cv::Mat getFGMask(bool);
		int getRandom(int low, int top);  // 在[a, b]之间产生随机数
		unsigned long getFgNum();
		std::vector<cv::Rect> getRect();

	private:
		cv::Size minSize;               // 高，宽
		cv::Size maxSize;               // 高，宽
		int fgCount_thr;                // 某个点连续fgCount_thr次判断为前景，则认为是背景；0表示不作判断
		std::vector<cv::Rect> vt_rect;  // 检测目标的矩形坐标；左上角，宽、高
		unsigned long fgNum;            // 前景像素的个数
		int defaultNbSamples;           // 每个像素的样本集数量，默认20个
		int defaultReqMatches;          // 前景像素匹配数量，如果超过此值，则认为是背景像素
		int defaultRadius;              // 匹配半径，即在该半径内则认为是匹配像素
		int defaultSubsamplingFactor;   // 随机数因子，如果检测为前景，每个像素有1/defaultSubsamplingFactor几率更新样本集和领域样本集
		int background;
		int foreground;
		cv::Mat fgCount;                // 记录每个像素点连续检测为前景的次数
		cv::Mat fgMask;                 // 检测为前景的像素
		cv::Mat samples;                // 图片的样本集
		int samples_size[3];
		void buildNeighborArray(cv::Mat img);
		void processFgMask();
		void updataPixel(cv::Mat& img,int i, int j);  // 更新像素
	};
}