#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "vibe.hpp"
#include "remnants.hpp"

using namespace cv;
using namespace std;

void vibe_test(int, char**);
void remnants_test(int, char**);
void gradient_test();

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

void drawingLine(UMat& img, vector<cv::Point> tri)
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

void rand_test() {
	unsigned int seed = time(NULL);
	RNG rng(seed);

	Mat R = Mat(3, 2, CV_16SC3);
	// randu(dst, low, high);dst – 输出数组或矩阵 ；low – 区间下界（闭区间）； high - 区间上界（开区间）
	randu(R, -4, 5);                         // 返回均匀分布的随机数，填入数组或矩阵
	cout << R << endl;

	// randn(dst, mean, stddev);dst – 输出数组或矩阵； mean – 均值； stddev - 标准差
	randn(R, -4, 4);                        // 返回高斯分布的随机数，填入数组或矩阵
	cout << R << endl;

	/*randShuffle(InputOutputArray dst,     输入输出数组（一维）
					double iterFactor = 1., 决定交换数值的行列的位置的一个系数...
					RNG* rng = 0)          （可选）随机数产生器，0表示使用默认的随机数产生器，即seed = -1。rng决定了打乱的方法*/
	randShuffle(R, 1, &rng);                // 将原数组（矩阵）打乱
	cout << R << endl;
}

int main(int argc, char** argv) {

	remnants_test(argc, argv);
	//vibe_test(argc,argv);
	//gradient_test();
	int a;
	cin >> a;
	return 0;
}

void gradient_test() {
	cv::VideoCapture cap;
	cap.open(0);

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

	UMat frame, gray;
	for (int i = 0; i < 10; i++)   // 去掉前面几帧
		cap >> frame;

	cvtColor(frame, gray, COLOR_BGR2GRAY);
	vibe_hw::VIBE vibe = vibe_hw::VIBE(20, 2, 20, 16, Size(50, 50), Size(680, 480), false);
	vibe.initSamples(gray);

	Mat gray_mat = Mat(gray.rows, gray.cols, CV_8UC1);
	gray.copyTo(gray_mat);
	vector<cv::Point> points;
	points = getPoints(gray_mat);
	Mat zone_mat = getMask(gray_mat, points);

	int loc = 60;

	UMat ret;
	vector<Rect>vt_rect;
	while (1) {
		cap >> frame;
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		cvtColor(frame, gray, COLOR_BGR2GRAY);
		vibe.findFgMask(gray);
		ret = vibe.getFGMask(false);
		vt_rect.clear();
		vt_rect = vibe.getRect();


		UMat grad = getImgGradient(gray);
		double meanGrad = 0;
		meanGrad = getPartImgMeanGradient_2(grad, ret, Rect(points[0].x, points[0].y, (points[2].x - points[0].x), (points[2].y - points[0].y)));

		string gm = "GM:" + format("%.2f", meanGrad);
		putText(frame, gm, Point(points[0].x, points[0].y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);

		for (int i = 0; i < points.size(); i++) {
			drawingLine(frame, points);
		}

		rectangle(frame, Rect(points[0].x, points[0].y, (points[2].x - points[0].x), (points[2].y - points[0].y)), Scalar(255, 0, 0), 2);

		imshow("zone", zone_mat);
		imshow("ret", ret);
		imshow("frame", frame);
		if (waitKey(1) == 27) {
			cap.release();
			destroyAllWindows();
			break;
		}
	}
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

	UMat frame, gray;
	for (int i = 0; i < 10; i++)   // 去掉前面几帧
		cap >> frame;

	cvtColor(frame, gray, COLOR_BGR2GRAY);
	Mat gray_mat = Mat(gray.rows, gray.cols, CV_8UC1);
	gray.copyTo(gray_mat);
	vector<cv::Point> points;
	points = getPoints(gray_mat);
	Mat zone_mat = getMask(gray_mat, points);

	UMat zone = UMat(gray.rows, gray.cols, CV_8UC1);
	zone_mat.copyTo(zone);
	Remnamts remn = Remnamts(0, 0.8, Size(20, 20), Size(600, 400), zone, 123, 0.9, 5, true, 12, 50);
	remn.initSamples(gray);

	int i = 50;
	UMat mask;
	unsigned long fgNum;
	//unsigned long frameNum = 0;
	vector<SuspFgMask> vt_ret;
	string label;
	while (i) {
		if (!cap.read(frame)) {
			int sz[2] = { 200, 1000 };
			UMat end_img = UMat::zeros(2, sz, CV_8UC1);
			string end = "the video is over! Press any key to end";
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
		vt_ret.clear();
		vt_ret = remn.getRect();

		for (int i = 0; i < vt_ret.size(); i++) {
			rectangle(frame, vt_ret[i].rect, Scalar(255, 0, 0), 2);
			string gm = "GM:" + format("%.2f", vt_ret[i].meanGradPre) + " " + format("%.2f", vt_ret[i].meanGradCur);
			putText(frame, gm, Point(vt_ret[i].rect.x, vt_ret[i].rect.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
		}

		double end_time = (double)getTickCount();
		double fps = getTickFrequency() / (end_time - start_time);
		label = "FPS:" + format("%.2f", fps) + " " + "FP:" + format("%d", fgNum);// +" " + "GM:" + format("%.2f", meanGrad);
		putText(frame, label, cv::Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		for (int i = 0; i < points.size(); i++) {
			drawingLine(frame, points);
		}
		if (vt_ret.size() > 0) {
			string img_name = "img_" + getLocNameTime() + ".png";
			string mask_name = "mask_" + getLocNameTime() + ".png";
			cout << "侦测到遗留物，保存的图片名为:" << img_name << endl;
			imwrite(img_name, frame);
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
		UMat showError = UMat::zeros(2, sz, CV_8UC1);
		string error = "Please detect the path of the video or camera";
		string end = "Press any key to end";
		putText(showError, error, cv::Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
		putText(showError, end, cv::Point(5, 60), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
		imshow("frame", showError);
		waitKey();
		destroyAllWindows();
		return;
	}

	UMat frame, gray;
	for (int i = 0; i < 10; i++)   // 去掉前面几帧
		cap >> frame;

	cvtColor(frame, gray, COLOR_BGR2GRAY);

	unsigned int seed = time(NULL);
	RNG rng(seed);
	UMat mat_img = UMat::zeros(4, 20, CV_8UC1);
	rng.fill(mat_img, RNG::UNIFORM, 10, 20, true);
	//cout << "初始化图片:" << endl;
	//cout << mat_img << endl;

	vibe_hw::VIBE vibe = vibe_hw::VIBE(20, 2, 20, 16, Size(50, 50), Size(680, 480), false);
	vibe.initSamples(gray);

	int i = 50;
	UMat ret;
	unsigned long fgNum;
	unsigned long frameNum = 0;
	vector<Rect> vt_rect;
	string label;
	while (i) {
		if (!cap.read(frame)) {
			int sz[2] = { 200, 1000 };
			UMat end_img = UMat::zeros(2, sz, CV_8UC1);
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
			destroyWindow("frame");
			destroyWindow("segMat");
			break;
		}
	}
}