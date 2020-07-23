#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Windows.h>

#include "vibe.h"
#include "applyVibe.h"
#include "utils.hpp"

#define PI 3.1415926

using namespace cv;
using namespace std;

void testVibe();
void testApplyVibe();

void testGrad() {
	VideoCapture capture(0);
	Mat frame;
	Mat gray;
	while (1)
	{
		capture >> frame;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		cv::Mat gray_grad = cv::Mat::zeros(gray.rows, gray.cols, CV_16SC1);
		getImgGradient(gray, gray_grad);
		imshow("frame", gray);
		if (waitKey(1) == 27) {
			destroyAllWindows();
			break;
		}
	}
}

void testVector() {
	std::vector<int> abc;
	abc.push_back(1);
	abc.push_back(2);
	abc.push_back(3);
	abc.push_back(4);
	abc.push_back(5);
	abc.push_back(6);
	for (int i = 0; i < abc.size(); i++) {
		cout << abc[i] << " ";
	}
	cout << endl;

	std::vector<int> deleteData;
	deleteData.push_back(2);
	deleteData.push_back(4);
	std::vector<int>::iterator it = abc.begin();
	for (int i = (deleteData.size() -1); i >= 0; i--) {
		abc.erase(abc.begin() + deleteData[i]);
	}
	for (int i = 0; i < abc.size(); i++) {
		cout << abc[i] << " ";
	}
	cout << endl;
}

double getAngle() {
	int ori_x = 50;
	int ori_y = -50;
	if ((ori_x == 0) && (ori_y == 0)) {
		return 0;
	}

	if (ori_x < 0 && ori_y < 0) {
		return 180 - acos(-ori_x / sqrt(ori_x*ori_x + ori_y*ori_y)) * 180.0f / PI;
	}
	if (ori_x > 0 && ori_y < 0) {
		return acos(ori_x / sqrt(ori_x*ori_x + ori_y*ori_y)) * 180.0f / PI;
	}
	if (ori_x < 0 && ori_y > 0) {
		return acos(-ori_x / sqrt(ori_x*ori_x + ori_y*ori_y)) * 180.0f / PI + 180;
	}
	if (ori_x > 0 && ori_y > 0) {
		return 360 - acos(ori_x / sqrt(ori_x*ori_x + ori_y*ori_y)) * 180.0f / PI;
	}
}

int main(int argc, char* argv[])
{
	//testGrad();
	//testVibe();
	testApplyVibe();

	int a;
	cin >> a;
	return EXIT_SUCCESS;
}

void testApplyVibe() {
	ApplyVibe applyVibe;
	string vibeConfigPath = "vibeConfig.txt";
	applyVibe.initPara(vibeConfigPath);

	VideoCapture capture(0);
	//VideoCapture capture("C:/Users/weiz/Desktop/testVideo/wt/wt00.mp4");
	//VideoCapture capture("C:/Users/weiz/Desktop/testVideo/pg/pg04.mp4");

	Mat frame;
	for (int i = 0; i < 10; i++)   // 去掉前面几帧
		capture >> frame;

	if (!capture.isOpened()) {
		cerr << "Unable to open video file: " << endl;
		exit(EXIT_FAILURE);
	}

	//Mat zone = Mat::zeros(720, 1280, CV_8UC1);
	Mat zone = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	vector<cv::Point> points;
	points = getPoints(frame);   // 获取点集
	points2Mask(zone, points);
	//imshow("zone", zone);
	Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	Size minSize = Size(40,40);
	Size maxSize = Size(550, 400);

	cv::Mat gray = cv::Mat::zeros(frame.size().height, frame.size().width, CV_8UC1);
	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
	double tmp_time_1 = (double)getTickCount();
	applyVibe.initVibe(gray);
	double tmp_time_2 = (double)getTickCount();
	double spend_time = (tmp_time_2 - tmp_time_1) / getTickFrequency();
	//std::cout << spend_time << std::endl;

	int keyboard = 0;
	long frameNum = 0;
	double spend_time_t = 0;
	std::vector<cv::Rect> ret;
	while ((char)keyboard != 'q' && (char)keyboard != 27) {
		if (!capture.read(frame)) {
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			break;
		}
		frameNum++;
		double start_time = (double)getTickCount();

		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		ret.clear();
		applyVibe.runningVibe(gray, zone, minSize, maxSize, ret);

		double end_time = (double)getTickCount();
		spend_time_t += (end_time - start_time) / getTickFrequency();
		//cout << spend_time_t / frameNum << endl;
		double fps = getTickFrequency() / (end_time - start_time);
		string label = "FPS:" + format("%.2f", fps);
		putText(frame, label, cv::Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);

		for (int i = 0; i < ret.size(); i++)
			rectangle(frame, ret[i], Scalar(0, 0, 255), 2);
		imshow("Frame", frame);
		keyboard = waitKey(1);
	}

	destroyAllWindows();
}

void testVibe()
{
	VideoCapture capture("C:/Users/weiz/Desktop/testVideo/wt/wt00.mp4");
	//VideoCapture capture(0);
	Mat frame1;
	for (int i = 0; i < 10; i++)   // 去掉前面几帧
		capture >> frame1;

	if (!capture.isOpened()) {
		cerr << "Unable to open video file: " << endl;
		exit(EXIT_FAILURE);
	}

	namedWindow("Frame");
	namedWindow("mask");

	/* Variables. */
	static int frameNumber = 1; /* The current frame number */
	Mat frame;                  /* Current frame. */
	Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
	int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */

								/* Model for ViBe. */
	vibeModel_Sequential_t *model = NULL; /* Model used by ViBe. */

	string label;

	/* Read input data. ESC or 'q' for quitting. */
	while ((char)keyboard != 'q' && (char)keyboard != 27) {
		/* Read the current frame. */
		if (!capture.read(frame)) {
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			exit(EXIT_FAILURE);
		}

		if ((frameNumber % 100) == 0) { cout << "Frame number = " << frameNumber << endl; }

		double start_time = (double)getTickCount();

		if (frameNumber == 1) {
			segmentationMap = Mat(frame.rows, frame.cols, CV_8UC1);
			model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
			libvibeModel_Sequential_AllocInit_8u_C3R(model, frame.data, frame.cols, frame.rows);
		}

		/* ViBe: Segmentation and updating. */
		libvibeModel_Sequential_Segmentation_8u_C3R(model, frame.data, segmentationMap.data);
		libvibeModel_Sequential_Update_8u_C3R(model, frame.data, segmentationMap.data);

		double end_time = (double)getTickCount();
		double fps = getTickFrequency() / (end_time - start_time);
		label = "FPS:" + format("%.2f", fps);
		putText(frame, label, cv::Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);

		/* Shows the current frame and the segmentation map. */
		imshow("Frame", frame);
		imshow("mask", segmentationMap);

		++frameNumber;

		/* Gets the input from the keyboard. */
		keyboard = waitKey(1);
	}

	/* Delete capture object. */
	capture.release();

	/* Frees the model. */
	libvibeModel_Sequential_Free(model);

	destroyAllWindows();
}