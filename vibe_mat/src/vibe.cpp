#include "vibe.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace vibe_hw {
	/*
	初始化vibe参数*/
	VIBE::VIBE() {
		nbSamples = 20;
		reqMatches = 2;
		radius = 20;
		subsamplingFactor = 16;
		minSize = cv::Size(50, 50);
		maxSize = cv::Size(640, 480);
		update_neighborhood = false;

		background = 0;
		foreground = 255;
		updata_index = 0;
	}

	VIBE::VIBE(int num_sam, int min_match, int radiu, int rand_sam, cv::Size minSize, cv::Size maxSize, bool update_neighborhood) :
		nbSamples(num_sam),
		reqMatches(min_match),
		radius(radiu),
		subsamplingFactor(rand_sam),
		minSize(minSize),
		maxSize(maxSize),
		update_neighborhood(update_neighborhood) {
		background = 0;
		foreground = 255;
		updata_index = 0;
	}

	/*
	初始化背景模型:
		1、初始化每个像素的样本集矩阵
		2、初始化前景矩阵的mask
		3、初始化前景像素的检测次数矩阵
		参数：
		img: 传入的numpy图像素组，要求灰度图像*/
	void VIBE::initSamples(cv::Mat& gray) {
		buildNeighborArray(gray);
		int sz[2] = { gray.rows, gray.cols };  // {高，宽}
		fgMask = cv::Mat::zeros(2, sz, CV_8UC1);
	}

	void VIBE::reInitSamples(cv::Mat& gray) {
		fgMask.setTo(0);
		fgNum = 0;
		vt_rect.clear();
		samples.clear();
		buildNeighborArray(gray);
	}

	// 待测试
	// 局部更新samples样本集；更新的部分为vt_rect
	void VIBE::reInitPartSamples(cv::Mat& gray, std::vector<cv::Rect> vt) {
		unsigned int seed = time(NULL);
		cv::RNG rng(seed);

		cv::Mat vt_mask = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);
		for (int i = 0; i < vt.size(); i++) {
			vt_mask(vt[i]).setTo(255);
		}

		cv::Mat gray_16s_part = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);
		cv::Mat gray_16s = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);
		gray.convertTo(gray_16s, CV_16SC1);
		cv::bitwise_and(gray_16s, gray_16s, gray_16s_part, vt_mask);

		// 测试
		//cv::Mat imshow_mat = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);
		//gray_16s_part.convertTo(imshow_mat, CV_8UC1);
		//cv::imshow("part", imshow_mat);
		//cv::imshow("fgMask", vt_mask);
		//cv::waitKey(1);

		cv::Mat mat_4_part = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);
		cv::Mat mat_8u_part = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);

		cv::Mat sample_part = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);

		for (int i = 0; i < samples.size(); i++) {
			rng.fill(mat_4_part, cv::RNG::UNIFORM, -4, 4, true);
			cv::bitwise_and(mat_4_part, mat_4_part, mat_4_part, vt_mask);

			mat_4_part = mat_4_part + gray_16s_part;
			mat_4_part.convertTo(mat_8u_part, CV_8UC1);
			mat_8u_part.convertTo(mat_4_part, CV_16SC1);

			sample_part = samples[i];
			cv::bitwise_and(sample_part, sample_part, sample_part, vt_mask);
			samples[i] = samples[i] - sample_part;
			samples[i] = samples[i] + mat_4_part;
		}
	}

	/*
	构建一副图像中每个像素的邻域数组(实质是每个像素值加一个数或减一个数或不变)
		参数：输入灰度图像
		返回值：每个像素9邻域数组，保存到samples中*/
	void VIBE::buildNeighborArray(cv::Mat& gray) {
		unsigned int seed = time(NULL);
		cv::RNG rng(seed);
		int sz[2] = { gray.rows, gray.cols };  // {高，宽}

		cv::Mat img_16s = cv::Mat::zeros(2, sz, CV_16SC1);
		gray.convertTo(img_16s, CV_16SC1);

		cv::Mat mat_4 = cv::Mat::zeros(2, sz, CV_16SC1);
		cv::Mat mat_8u = cv::Mat::zeros(2, sz, CV_8UC1);

		for (int i = 0; i < nbSamples; i++) {
			cv::Mat mat_16s = cv::Mat::zeros(2, sz, CV_16SC1);
			rng.fill(mat_4, cv::RNG::UNIFORM, -4, 4, true);
			//std::cout << "随机矩阵:" << std::endl;
			//std::cout << mat_4 << std::endl;
			mat_4 = img_16s + mat_4;
			//std::cout << "相加后:" << std::endl;
			//std::cout << mat_4 << std::endl;

			// 确保像素不小于0和不大于255
			mat_4.convertTo(mat_8u, CV_8UC1);
			mat_8u.convertTo(mat_16s, CV_16SC1);
			samples.push_back(mat_16s);

			//std::cout << "最后结果:" << std::endl;
			//std::cout << mat_16s << std::endl;
			//std::cout << "-------------------------------" << std::endl;
		}
		/*std::cout << "样本图片集:" << std::endl;
		for (int i = 0; i < samples.size(); i++) {
			std::cout << samples[i] << std::endl;
		}
		std::cout << "样本图结束。" << std::endl;*/
	}
	/*
	这个函数有两个作用：
	第一：找到前景目标的像素；
		规则：比对当前像素值的defaultRadius领域在（对应位置）样本集的个数，小于则为前景
	第二：更新样本的样本集*/
	void VIBE::findFgMask(cv::Mat& gray) {
		cv::Mat img_16s = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);
		gray.convertTo(img_16s, CV_16SC1);                                 // 把图片从CV_8UC1转成CV_16SC1

		cv::Mat diff = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);     // 输入图片和样本集差的绝对值
		cv::Mat ones = cv::Mat::ones(gray.rows, gray.cols, CV_16SC1);
		cv::Mat ret = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);
		cv::Mat mask = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);;
		cv::Mat ret_tmp;

		for (int i = 0; i < samples.size(); i++) {
			//std::cout << "样本图片:" << std::endl;
			//std::cout << samples[i] << std::endl;
			ret_tmp = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);
			cv::absdiff(samples[i], img_16s, diff);
			//std::cout << "样本图片和输入图片的差:" << std::endl;
			//std::cout << diff << std::endl;
			mask = diff < radius;
			//std::cout << "掩码:" << std::endl;
			//std::cout << mask << std::endl;
			cv::bitwise_and(ones, ones, ret_tmp, mask);
			ret = ret + ret_tmp;
			//std::cout << "结果:" << std::endl;
			//std::cout << ret << std::endl;
		}
		//std::cout << "差集结果:" << std::endl;
		//std::cout << ret << std::endl;

		// 如果小于匹配数量阈值，则为前景
		fgMask = ret < reqMatches;

		fgNum = countNonZero(fgMask);

		//std::cout << "前景掩码:" << std::endl;
		//std::cout << fgMask << std::endl;
		updateSamples(img_16s);
	}

	/*
	更新背景像素的样本集，可以拆分为两个步骤(实现的时候下面两步是一起实现的)：
	1.每个背景像素有1 / defaultSubsamplingFactor几率更新自己的样本集；
		更新样本集方式为随机选取该像素样本集中的一个元素，更新为当前像素的值。
	2.每个背景像素有1 / defaultSubsamplingFactor几率更新邻域的样本集；
		更新邻域样本集方式为随机选取一个邻域点，并在该邻域点的样本集中随机选择一个更新为当前像素值
		更新自己样本集。*/
	void VIBE::updateSamples(cv::Mat& gray) {
		cv::Mat element_3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::Mat bgMask = fgMask < 100;          // 背景的掩码
		//std::cout << "背景掩码:" << std::endl;
		//std::cout << bgMask << std::endl;
		//std::cout << fgMask << std::endl;

		if (update_neighborhood)
			cv::dilate(bgMask, bgMask, element_3);  // 获得背景8领域掩码
		//std::cout << "背景8领域掩码:" << std::endl;
		//std::cout << bgMask << std::endl;

		cv::Mat mat_pbt = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);  // 概率矩阵
		rng.fill(mat_pbt, cv::RNG::UNIFORM, 1, subsamplingFactor + 1, true);
		//std::cout << "概率:" << std::endl;
		//std::cout << mat_pbt << std::endl;

		cv::Mat updata = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);   // 需要更新的像素
		cv::bitwise_and(mat_pbt, mat_pbt, updata, bgMask);
		//std::cout << "背景8领域概率:" << std::endl;
		//std::cout << updata << std::endl;

		cv::Mat updata_mask = updata == 5;   // 有1 / defaultSubsamplingFactor几率更新像素和邻域的样本集
		//std::cout << "有1/n之一更新:" << std::endl;
		//std::cout << updata_mask << std::endl;

		// 更新样本集
		cv::Mat img_16s = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);
		gray.convertTo(img_16s, CV_16SC1);
		cv::Mat img_updata = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);
		cv::bitwise_and(img_16s, img_16s, img_updata, updata_mask);
		//std::cout << "更新的像素值:" << std::endl;
		//std::cout << img_updata << std::endl;
		cv::Mat smp = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);
		cv::bitwise_and(samples[updata_index], samples[updata_index], smp, updata_mask);
		//std::cout << "旧的像素值:" << std::endl;
		//std::cout << smp << std::endl;
		samples[updata_index] = samples[updata_index] - smp;
		samples[updata_index] = samples[updata_index] + img_updata;
		updata_index++;
		updata_index = updata_index % nbSamples;
		//std::cout << updata_index % defaultNbSamples << std::endl;
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

	std::vector<cv::Rect> VIBE::getRect() {
		vt_rect.clear();
		processFgMask();
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(fgMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
		cv::Rect boundRect;
		for (int i = 0; i < contours.size(); i++) {
			boundRect = cv::boundingRect((cv::Mat)contours[i]); //查找每个轮廓的外接矩形
			if ((boundRect.width <= maxSize.width) && (boundRect.width >= minSize.width) && (boundRect.height <= maxSize.height) && (boundRect.height >= minSize.height)) {
				vt_rect.push_back(cv::Rect(boundRect.x, boundRect.y, boundRect.width, boundRect.height));
			}
		}
		return vt_rect;
	}

	void VIBE::setMaxSize(cv::Size size) {
		maxSize = size;
	}

	void VIBE::setMinSize(cv::Size size) {
		minSize = size;
	}

	void VIBE::setRadius(int r) {
		radius = r;
	}

	void VIBE::setReqMathces(int m) {
		reqMatches = m;
	}

	void VIBE::setSubsamplingFactor(int f) {
		subsamplingFactor = f;
	}
}