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
	void VIBE::initSamples(cv::UMat& gray) {
		buildNeighborArray(gray);
		int sz[2] = { gray.rows, gray.cols };  // {高，宽}
		fgMask = cv::UMat::zeros(2, sz, CV_8UC1);
		radius_umat = cv::UMat(2, sz, CV_16SC1, cv::Scalar::all(radius));
		reqMatches_umat = cv::UMat(2, sz, CV_16SC1, cv::Scalar::all(reqMatches));
	}

	void VIBE::reInitSamples(cv::UMat& gray) {
		fgMask.setTo(0);
		fgNum = 0;
		vt_rect.clear();
		samples.clear();
		buildNeighborArray(gray);
	}

	// 待测试
    // 局部更新samples样本集；更新的部分为vt_rect
	void VIBE::reInitPartSamples(cv::UMat& gray, std::vector<cv::Rect> vt) {
		unsigned int seed = time(NULL);
		cv::RNG rng(seed);

		cv::UMat vt_mask = cv::UMat::zeros(gray.rows, gray.cols, CV_8UC1);
		for (int i = 0; i < vt.size(); i++) {
			vt_mask(vt[i]).setTo(255);
		}

		cv::UMat gray_16s_part = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);
		cv::UMat gray_16s = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);
		gray.convertTo(gray_16s, CV_16SC1);
		cv::bitwise_and(gray_16s, gray_16s, gray_16s_part, vt_mask);

		// 测试
		//cv::Mat imshow_mat = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);
		//gray_16s_part.convertTo(imshow_mat, CV_8UC1);
		//cv::imshow("part", imshow_mat);
		//cv::imshow("fgMask", vt_mask);
		//cv::waitKey(1);

		cv::UMat mat_4_part = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);
		cv::UMat mat_8u_part = cv::UMat::zeros(gray.rows, gray.cols, CV_8UC1);

		cv::UMat sample_part = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);

		for (int i = 0; i < samples.size(); i++) {
			rng.fill(mat_4_part, cv::RNG::UNIFORM, -4, 4, true);
			cv::bitwise_and(mat_4_part, mat_4_part, mat_4_part, vt_mask);

			cv::add(mat_4_part, gray_16s_part, mat_4_part);
			mat_4_part.convertTo(mat_8u_part, CV_8UC1);
			mat_8u_part.convertTo(mat_4_part, CV_16SC1);

			sample_part = samples[i];
			cv::bitwise_and(sample_part, sample_part, sample_part, vt_mask);

			cv::subtract(samples[i], sample_part, samples[i]);
			cv::add(samples[i], mat_4_part, samples[i]);
		}
	}

	/*
	构建一副图像中每个像素的邻域数组(实质是每个像素值加一个数或减一个数或不变)
		参数：输入灰度图像
		返回值：每个像素9邻域数组，保存到samples中*/
	void VIBE::buildNeighborArray(cv::UMat& gray) {
		unsigned int seed = time(NULL);
		cv::RNG rng(seed);
		int sz[2] = { gray.rows, gray.cols };  // {高，宽}

		cv::UMat img_16s = cv::UMat::zeros(2, sz, CV_16SC1);
		gray.convertTo(img_16s, CV_16SC1);

		cv::UMat mat_4 = cv::UMat::zeros(2, sz, CV_16SC1);
		cv::UMat mat_8u = cv::UMat::zeros(2, sz, CV_8UC1);

		for (int i = 0; i < nbSamples; i++) {
			cv::UMat mat_16s = cv::UMat::zeros(2, sz, CV_16SC1);
			rng.fill(mat_4, cv::RNG::UNIFORM, -4, 4, true);

			cv::add(img_16s, mat_4, mat_4);

			// 确保像素不小于0和不大于255
			mat_4.convertTo(mat_8u, CV_8UC1);
			mat_8u.convertTo(mat_16s, CV_16SC1);
			samples.push_back(mat_16s);
		}
	}
	/*
	这个函数有两个作用：
	第一：找到前景目标的像素；
		规则：比对当前像素值的defaultRadius领域在（对应位置）样本集的个数，小于则为前景
	第二：更新样本的样本集*/
	void VIBE::findFgMask(cv::UMat& gray) {
		cv::UMat img_16s = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);
		gray.convertTo(img_16s, CV_16SC1);                                // 把图片从CV_8UC1转成CV_16SC1

		cv::UMat diff = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);     // 输入图片和样本集差的绝对值
		cv::UMat ones = cv::UMat::ones(gray.rows, gray.cols, CV_16SC1);
		cv::UMat ret = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);
		cv::UMat mask = cv::UMat::zeros(gray.rows, gray.cols, CV_8UC1);;
		cv::UMat ret_tmp;

		for (int i = 0; i < samples.size(); i++) {
			ret_tmp = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);
			cv::absdiff(samples[i], img_16s, diff);

			cv::compare(diff, radius_umat, mask, cv::CMP_LT);

			cv::bitwise_and(ones, ones, ret_tmp, mask);

			cv::add(ret, ret_tmp, ret);
		}

		// 如果小于匹配数量阈值，则为前景
		cv::compare(ret, reqMatches_umat, fgMask, cv::CMP_LT);

		fgNum = countNonZero(fgMask);

		updateSamples(img_16s);
	}

	/*
	更新背景像素的样本集，可以拆分为两个步骤(实现的时候下面两步是一起实现的)：
	1.每个背景像素有1 / defaultSubsamplingFactor几率更新自己的样本集；
		更新样本集方式为随机选取该像素样本集中的一个元素，更新为当前像素的值。
	2.每个背景像素有1 / defaultSubsamplingFactor几率更新邻域的样本集；
		更新邻域样本集方式为随机选取一个邻域点，并在该邻域点的样本集中随机选择一个更新为当前像素值
		更新自己样本集。*/
	void VIBE::updateSamples(cv::UMat& gray) {
		cv::Mat element_3_cpu = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::UMat element_3;
		element_3_cpu.copyTo(element_3);
		cv::UMat umat_100 = cv::UMat(gray.rows, gray.cols, CV_8UC1, cv::Scalar::all(100));
		cv::UMat bgMask;          
		cv::compare(fgMask, umat_100, bgMask, cv::CMP_LT);  // 背景的掩码

		if (update_neighborhood)
			cv::dilate(bgMask, bgMask, element_3);  // 获得背景8领域掩码

		cv::UMat mat_pbt = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);  // 概率矩阵
		rng.fill(mat_pbt, cv::RNG::UNIFORM, 1, subsamplingFactor + 1, true);

		cv::UMat updata = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);   // 需要更新的像素
		cv::bitwise_and(mat_pbt, mat_pbt, updata, bgMask);

		cv::UMat umat_5 = cv::UMat(gray.rows, gray.cols, CV_16SC1, cv::Scalar::all(5));
		cv::UMat updata_mask;   // 有1 / defaultSubsamplingFactor几率更新像素和邻域的样本集
		cv::compare(updata, umat_5, updata_mask, cv::CMP_EQ);

		// 更新样本集
		cv::UMat img_16s = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);
		gray.convertTo(img_16s, CV_16SC1);
		cv::UMat img_updata = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);
		cv::bitwise_and(img_16s, img_16s, img_updata, updata_mask);

		cv::UMat smp = cv::UMat::zeros(gray.rows, gray.cols, CV_16SC1);
		cv::bitwise_and(samples[updata_index], samples[updata_index], smp, updata_mask);

		cv::subtract(samples[updata_index], smp, samples[updata_index]);
		cv::add(samples[updata_index], img_updata, samples[updata_index]);
		updata_index++;
		updata_index = updata_index % nbSamples;
	}

	cv::UMat VIBE::getFGMask(bool process) {
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
		cv::Mat element_3_cpu = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::UMat element_3;
		element_3_cpu.copyTo(element_3);
		//cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, element_3);   // 开运算

		cv::Mat element_7_cpu = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::UMat element_7;
		element_7_cpu.copyTo(element_7);
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
		radius_umat.setTo(r);
	}

	void VIBE::setReqMathces(int m) {
		reqMatches = m;
		reqMatches_umat.setTo(m);
	}

	void VIBE::setSubsamplingFactor(int f) {
		subsamplingFactor = f;
	}
}