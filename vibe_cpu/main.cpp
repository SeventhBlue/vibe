#include <iostream>
#include "vibe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace std;
using namespace cv;

int test(int c, char** path) {
	cv::VideoCapture cap;
	if (c > 1) {
		cap.open(path[1]);
	}
	else {
		cap.open(0);
	}
	if (!cap.isOpened()) {
		int sz[2] = { 200, 1000 };
		Mat showError = Mat::zeros(2, sz, CV_8UC1);
		string error = "Please detect the path of the video or camera";
		string end = "Press any key to end";
		putText(showError, error, cv::Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
		putText(showError, end, cv::Point(5, 60), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
		imshow("frame", showError);
		waitKey();
		return 0;
	}

	Mat frame, gray;
	for(int i=0;i<10;i++)   // 去掉前面几帧
		cap >> frame;

	cvtColor(frame, gray, COLOR_BGR2GRAY);

	vibe::VIBE vibe = vibe::VIBE(20, 2, 20, 8, 0, Size(60,50), Size(680,480));
	vibe.initBGModel(gray);

	namedWindow("frame", WINDOW_AUTOSIZE);
	namedWindow("segMat", WINDOW_AUTOSIZE);

	Mat ret;
	unsigned long fgNum;
	unsigned long frameNum = 0;
	vector<Rect> vt_rect;
	string label;
	while (true) {
		cap >> frame;
		frameNum++;
		double start_time = (double)getTickCount();
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		vibe.update(gray);
		ret = vibe.getFGMask(false);
		fgNum = vibe.getFgNum();
		vt_rect.clear();
		vt_rect = vibe.getRect();

		for(int i=0;i<vt_rect.size();i++)
			rectangle(frame, vt_rect[i], Scalar(0, 0, 255), 2);
		double end_time = (double)getTickCount();
		double fps = getTickFrequency() / (end_time - start_time);
		label = "FPS:" + format("%.2f", fps) + " " + "FP:" + format("%d", fgNum) + " " + "Frames:" + format("%d", frameNum);
		putText(frame, label, cv::Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		imshow("frame", frame);
		imshow("segMat", ret);
		if (waitKey(1) == 27) {
			cap.release();
			destroyWindow("frame");
			destroyWindow("segMat");
			break;
		}
	}

	return 1;
}

int main(int argc, char** argv) {
	test(argc, argv);
	return 0;
}