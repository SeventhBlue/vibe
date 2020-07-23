#include "vibe.hpp"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include<cmath>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

namespace vibe
{
	/*
	初始化vibe参数*/
	VIBE::VIBE(int num_sam = 20, int min_match = 2, int radiu = 20, int rand_sam = 16, int fgCount_thr=80, cv::Size minSize = cv::Size(50, 60), cv::Size maxSize = cv::Size(640,480)) :
		defaultNbSamples(num_sam),
		defaultReqMatches(min_match),
		defaultRadius(radiu),
		defaultSubsamplingFactor(rand_sam),
		fgCount_thr(fgCount_thr),
		minSize(minSize),
		maxSize(maxSize){
		background = 0;
		foreground = 255;
	}

	/*
	初始化背景模型:
		1、初始化每个像素的样本集矩阵
		2、初始化前景矩阵的mask
		3、初始化前景像素的检测次数矩阵
		参数：
		img: 传入的numpy图像素组，要求灰度图像*/
	void VIBE::initBGModel(cv::Mat img) {
		buildNeighborArray(img);
		int sz[2] = { img.rows, img.cols };  // {高，宽}
		fgCount = cv::Mat::zeros(2, sz, CV_8UC1);
		fgMask = cv::Mat::zeros(2, sz, CV_8UC1);
	}

	int VIBE::getRandom(int low, int up) {
		return ((rand() % (up - low + 1)) + low);
	}

	/*
	构建一副图像中每个像素的邻域数组(实质是每个像素值加一个数或减一个数或不变)
		参数：输入灰度图像
		返回值：每个像素9邻域数组，保存到self.samples中*/
	void VIBE::buildNeighborArray(cv::Mat img) {
		srand((int)time(0)); // 产生随机种子
		// 产生一个 width * height * defaultNbSamples 图片集
		samples = cv::Mat::zeros(img.size().height, img.size().width, CV_8UC(20));
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				for (int c = 0; c < defaultNbSamples; c++) {
					//int a = img.at<uchar>(i, j);
					if (img.at<uchar>(i, j) <= 4)
						samples.at<Vec20u>(i, j)[c] = 0;
					else if (img.at<uchar>(i, j) >= 251)
						samples.at<Vec20u>(i, j)[c] = 255;
					else {
						//int b = getRandom(-4, 4);
						samples.at<Vec20u>(i, j)[c] = img.at<uchar>(i, j) + getRandom(-4, 4);
					}
					//cout << int(img.at<uchar>(i, j)) << " ";
					//cout << samples.at<Vec20f>(i, j)[c] << " ";
				}
			}
			//cout << endl;
		}
		/*for (int i = 0; i < 10; i++)
			cout << getRandom(-4, 4) << " ";
		cout << endl;
		int x = 20;
		int y = 20;
		cout << int(img.at<uchar>(x, y)) << endl;

		for (int i = 0; i < defaultNbSamples; i++)
			cout << int(samples.at<Vec20u>(x, y)[i]) << " ";
		cout << endl;*/
	}

	/*
	这个函数有两个作用：
	第一：找到前景目标的像素；
		规则：比对当前像素值的defaultRadius领域在（对应位置）样本集的个数，小于则为前景
	第二：更新样本的样本集；
		更新背景像素的样本集，可以分两个步骤：
		1.每个背景像素有1 / defaultSubsamplingFactor几率更新自己的样本集
		  更新样本集方式为随机选取该像素样本集中的一个元素，更新为当前像素的值
		2.每个背景像素有1 / defaultSubsamplingFactor几率更新邻域的样本集
		  更新邻域样本集方式为随机选取一个邻域点，并在该邻域点的样本集中随机选择一个更新为当前像素值
		  更新自己样本集*/
	// 追求速度版本:检测前景和更新领域样本集原本不应该放到同一个循环里，但是为了追求速度放到同一个循环里
	void VIBE::update(cv::Mat img) {
		fgNum = 0;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				// 计算当前像素值与样本库中值之差小于阀值范围RADIUS的个数
				int sum = 0;
				for (int c = 0; c < defaultNbSamples; c++) {
					// 比对当前像素值的defaultRadius领域在（对应位置）样本集的个数
					if (abs(int(samples.at<Vec20u>(i, j)[c]) - int(img.at<uchar>(i, j))) < defaultRadius)
						sum += 1;
				}
				// 如果小于匹配数量阈值，为前景
				if (sum < defaultReqMatches) {
					fgMask.at<uchar>(i, j) = foreground;
					if (abs(fgCount_thr)) {
						fgCount.at<uchar>(i, j) += 1;
						// 如果某个像素连续fgCount_thr次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点
						if (fgCount.at<uchar>(i, j) >= abs(fgCount_thr)) {
							fgMask.at<uchar>(i, j) = background;
							fgCount.at<uchar>(i, j) = 0;
							updataPixel(img, i, j);
						}
						else {
							fgNum++;
						}
					}
					else {
						fgNum++;
					}
				}
				else {
					fgMask.at<uchar>(i, j) = background;  // 如果大于等于匹配数量阀值，则是背景
					if (abs(fgCount_thr)) {
						fgCount.at<uchar>(i, j) = 0;
					}
					updataPixel(img, i, j);
				}
			}
		}
	}

	void VIBE::updataPixel(cv::Mat& img, int i, int j) {
		int p1 = getRandom(1, defaultSubsamplingFactor);  // 在[1, defaultSubsamplingFactor]内随机产生一个数
		// 1 / defaultSubsamplingFactor几率更新自己的样本集
		if (p1 == 1) {
			samples.at<Vec20u>(i, j)[getRandom(0, defaultNbSamples - 1)] = img.at<uchar>(i, j);
		}
		int p2 = getRandom(1, defaultSubsamplingFactor);  // 在[1, defaultSubsamplingFactor]内随机产生一个数
		// 1 / defaultSubsamplingFactor几率更新邻域的样本集
		if (p2 == defaultSubsamplingFactor) {
			// 1:表示更新当前像素上一个像素 2:表示更新当前像素右一个像素
			// 3:表示更新当前像素下一个像素 4:表示更新当前像素左一个像素
			switch (getRandom(1, 4))
			{
			case 1:
				if ((j - 1) >= 0) {
					samples.at<Vec20u>(i, j - 1)[getRandom(0, defaultNbSamples - 1)] = img.at<uchar>(i, j);
					break;
				}
			case 2:
				if ((i + 1) < img.size().height) {
					samples.at<Vec20u>(i + 1, j)[getRandom(0, defaultNbSamples - 1)] = img.at<uchar>(i, j);
					break;
				}
			case 3:
				if ((j + 1) < img.size().width) {
					samples.at<Vec20u>(i, j + 1)[getRandom(0, defaultNbSamples - 1)] = img.at<uchar>(i, j);
					break;
				}
			case 4:
				if ((i - 1) >= 0) {
					samples.at<Vec20u>(i - 1, j)[getRandom(0, defaultNbSamples - 1)] = img.at<uchar>(i, j);
					break;
				}
			}
		}
	}

	vector<cv::Rect> VIBE::getRect() {
		vt_rect.clear();
		processFgMask();
		vector<vector<cv::Point>> contours;
		vector<cv::Vec4i> hierarchy;
		cv::findContours(fgMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
		cv::Rect boundRect;
		for (int i = 0; i < contours.size(); i++) {
			boundRect = cv::boundingRect((cv::Mat)contours[i]); //查找每个轮廓的外接矩形
			if ((boundRect.width <= maxSize.width)&&(boundRect.width >= minSize.width)&& (boundRect.height <= maxSize.height) && (boundRect.height >= minSize.height)) {
				vt_rect.push_back(cv::Rect(boundRect.x, boundRect.y, boundRect.width, boundRect.height));
			}
		}
		return vt_rect;
	}

	cv::Mat VIBE::getFGMask(bool process) {
		if (process) {
			processFgMask();
			return fgMask;
		}
		else {
			return fgMask;
		}
	}

	unsigned long VIBE::getFgNum() {
		return fgNum;
	}

	void VIBE::processFgMask() {
		cv::Mat element_3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		//cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, element_3);   // 开运算

		cv::Mat element_7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
		cv::erode(fgMask, fgMask, element_3);  // 腐蚀
		cv::dilate(fgMask, fgMask, element_7);
	}
}