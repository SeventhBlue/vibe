#pragma once
#include <opencv2/core/core.hpp>
#include "utils.hpp"

namespace vibe_hw
{
	class VIBE
	{
	public:
		VIBE();
		VIBE(int num, int min_match, int radiu, int rand_sam, cv::Size minSize, cv::Size maxSize, bool update_neighborhood);
		void initSamples(cv::UMat& gray);                                        // 初始化samples
		void reInitSamples(cv::UMat& gray);                                      // 重新初始化samples
		void reInitPartSamples(cv::UMat& gray, std::vector<SuspFgMask> vt);        // samples的局部重新初始化
		void findFgMask(cv::UMat& gray);
		cv::UMat getFGMask(bool);
		unsigned long getFgNum();
		std::vector<cv::Rect> getRect();

		void setReqMathces(int);
		void setRadius(int);
		void setSubsamplingFactor(int);
		void setMinSize(cv::Size);
		void setMaxSize(cv::Size);

	private:
		bool update_neighborhood;       // 背景像素的8邻域是否更新
		int updata_index = 0;           // 样本集更新的索引
		cv::RNG rng;                    // 随机种子
		cv::Size minSize;               // 高，宽
		cv::Size maxSize;               // 高，宽
		std::vector<cv::Rect> vt_rect;  // 检测目标的矩形坐标；左上角，宽、高
		unsigned long fgNum;            // 前景像素的个数
		int nbSamples;                  // 每个像素的样本集数量，默认20个
		int reqMatches;                 // 前景像素匹配数量，如果超过此值，则认为是背景像素
		cv::UMat reqMatches_umat;
		int radius;                     // 匹配半径，即在该半径内则认为是匹配像素
		cv::UMat radius_umat;
		int subsamplingFactor;          // 随机数因子，如果检测为前景，每个像素有1/defaultSubsamplingFactor几率更新样本集和领域样本集
		int background = 0;
		int foreground = 255;
		cv::UMat fgMask;                 // 检测为前景的像素
		std::vector<cv::UMat> samples;   // 图片的样本集
		void buildNeighborArray(cv::UMat& gray);
		void processFgMask();
		void updateSamples(cv::UMat& gray); // 更新samples
	};
}