#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "vibe.hpp"
#include "remnants.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

void remnants_test(int c, char** path);
void vibe_test(int c, char** path);

struct callbackP
{
	Mat src;
	vector<cv::Point> srcTri;
};

void onMouse(int event, int x, int y, int flags, void *utsc)
{
	callbackP cp = *(callbackP*)utsc;  // 先转换类型，再取数据

	if (event == EVENT_LBUTTONUP)      // 响应鼠标左键事件
	{
		circle((*(callbackP*)utsc).src, cv::Point(x, y), 2, Scalar(255, 255, 255), 4);  //标记选中点
		imshow("wait ", (*(callbackP*)utsc).src);
		(*(callbackP*)utsc).srcTri.push_back(cv::Point(x, y));
		cout << "x:" << x << " " << "y:" << y << endl;
	}
}

void drawingLine(Mat& img, vector<cv::Point> tri)
{
	for (int i = 0; i < tri.size(); i++)
	{
		if (i == (tri.size() - 1))
		{
			line(img, tri[0], tri[i], Scalar(0, 0, 255), 2);
		}
		else
		{
			line(img, tri[i], tri[i + 1], Scalar(0, 0, 255), 2);
		}

	}
}

Mat getMask(Mat& src, vector<Point> mask_points) {

	vector<vector<Point>> mask_area;
	mask_area.push_back(mask_points);

	polylines(src, mask_area, 1, Scalar(0, 0, 0));
	cv::Mat mask;

	src.copyTo(mask);
	mask.setTo(cv::Scalar::all(0));
	fillPoly(mask, mask_area, Scalar(255, 255, 255));

	return mask;
}

vector<cv::Point> getPoints(Mat img) {
	callbackP utsc;
	utsc.src = img.clone();
	namedWindow("src", WINDOW_AUTOSIZE);
	imshow("src", utsc.src);
	setMouseCallback("src", onMouse, (void*)&utsc);  //类型转换
	waitKey();
	destroyAllWindows();
	return utsc.srcTri;
}


int main(int argc, char** argv) {

	//remnants_test(argc, argv);
	vibe_test(argc, argv);

	int a;
	cin >> a;
	return 0;
}

void remnants_test(int c, char** path) {
	cv::VideoCapture cap;
	if (c > 1) {
		cap.open(path[1]);
	}
	else {
		// rtsp://admin:1q2w#E$R@192.168.0.241:1554/Streaming/Channels/301  rtsp://admin:1q2w#E$R@192.168.0.243:554
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
		destroyAllWindows();
		return;
	}

	Mat frame, gray;
	for (int i = 0; i < 10; i++)   // 去掉前面几帧
		cap >> frame;

	cvtColor(frame, gray, COLOR_BGR2GRAY);
	vector<cv::Point> points;
	points = getPoints(gray);
	Mat zone = getMask(gray, points);
	
	Remnamts remn = Remnamts(0, 0.8, Size(20,20), Size(600, 400), zone, 123, 0.9, 20, false, 1800);
	remn.initSamples(gray);

	int i = 50;
	Mat mask;
	unsigned long fgNum;
	//unsigned long frameNum = 0;
	vector<Rect> vt_rect;
	string label;
	while (i) {
		//rng.fill(mat_img, RNG::UNIFORM, 40, 50, true);
		//cout << "输入的图片:" << endl;
		//cout << mat_img << endl;
		//vibe.findFgMask(mat_img);
		//i--;
		//cout << "start-----------------------------------" << endl;
		cap >> frame;
		if (!frame.data) {
			int sz[2] = { 200, 1000 };
			Mat end_img = Mat::zeros(2, sz, CV_8UC1);
			string end = "The video is over! Press any key to end";
			putText(end_img, end, cv::Point(5, 60), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
			imshow("end_img", end_img);
			waitKey();
			destroyAllWindows();
			return;
		}

		//frameNum++;
		double start_time = (double)getTickCount();
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		remn.findFgMask(gray);
		mask = remn.getFGMask();
		fgNum = remn.getFgNum();
		vt_rect.clear();
		vt_rect = remn.getRect();

		for (int i = 0; i < vt_rect.size(); i++) {
			rectangle(frame, vt_rect[i], Scalar(255, 0, 0), 2);
		}

		double end_time = (double)getTickCount();
		double fps = getTickFrequency() / (end_time - start_time);
		label = "FPS:" + format("%.2f", fps) + " " + "FP:" + format("%d", fgNum);// +" " + "GM:" + format("%.2f", meanGrad);
		putText(frame, label, cv::Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		for (int i = 0; i < points.size(); i++) {
			drawingLine(frame, points);
		}
		if (vt_rect.size() > 0) {
			string img_name = "img_" + getLocNameTime() + ".png";
			string mask_name = "mask_" + getLocNameTime() + ".png";
			cout << "侦测到遗留物，保存的图片名为:" <<img_name << endl;
			imwrite(img_name,frame);
			imwrite(mask_name, mask);
		}
		imshow("frame", frame);
		imshow("segMat", mask);
		if (waitKey(1) == 27) {
			cap.release();
			destroyAllWindows();
			break;
		}
	}
}

void vibe_test(int c, char** path) {
	cv::VideoCapture cap;
	if (c > 1) {
		cap.open(path[1]);
	}
	else {
		// rtsp://admin:1q2w#E$R@192.168.0.241:1554/Streaming/Channels/301  rtsp://admin:1q2w#E$R@192.168.0.243:554
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
		destroyAllWindows();
		return;
	}

	Mat frame, gray;
	for (int i = 0; i < 10; i++)   // 去掉前面几帧
		cap >> frame;

	cvtColor(frame, gray, COLOR_BGR2GRAY);

	unsigned int seed = time(NULL);
	RNG rng(seed);
	Mat mat_img = Mat::zeros(4, 20, CV_8UC1);
	rng.fill(mat_img, RNG::UNIFORM, 10, 20, true);
	//cout << "初始化图片:" << endl;
	//cout << mat_img << endl;



	vibe_hw::VIBE vibe = vibe_hw::VIBE(20, 2, 20, 16, Size(50, 50), Size(680, 480), false);
	vibe.initSamples(gray);

	int i = 50;
	Mat ret;
	unsigned long fgNum;
	unsigned long frameNum = 0;
	vector<Rect> vt_rect;
	string label;
	while (i) {
		//rng.fill(mat_img, RNG::UNIFORM, 40, 50, true);
		//cout << "输入的图片:" << endl;
		//cout << mat_img << endl;
		//vibe.findFgMask(mat_img);
		//i--;

		cap >> frame;
		if (!frame.data) {
			int sz[2] = { 200, 1000 };
			Mat end_img = Mat::zeros(2, sz, CV_8UC1);
			string end = "the video is over! Press any key to end";
			putText(end_img, end, cv::Point(5, 60), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
			imshow("end_img", end_img);
			waitKey();
			destroyAllWindows();
			return;
		}

		frameNum++;
		double start_time = (double)getTickCount();
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		vibe.findFgMask(gray);
		ret = vibe.getFGMask(false);
		fgNum = vibe.getFgNum();
		vt_rect.clear();
		vt_rect = vibe.getRect();

		for (int i = 0; i < vt_rect.size(); i++)
			rectangle(frame, vt_rect[i], Scalar(0, 0, 255), 2);
		double end_time = (double)getTickCount();
		double fps = getTickFrequency() / (end_time - start_time);
		label = "FPS:" + format("%.2f", fps) + " " + "FP:" + format("%d", fgNum) + " " + "Frames:" + format("%d", frameNum);
		putText(frame, label, cv::Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		imshow("frame", frame);
		imshow("segMat", ret);
		if (waitKey(1) == 27) {
			cap.release();
			destroyAllWindows();
			break;
		}
	}
}