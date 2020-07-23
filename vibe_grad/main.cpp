#include <iostream>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "vibe.hpp"
#include "remnants.hpp"
#include "ViaDef.h"
#include "bumingwu.h"

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

void testWrapper();
InDataSt* getInDataSt(Mat frame, int taskType);

vector<cv::Point> getPoints(Mat img) {
	callbackP utsc;
	utsc.src = img.clone();
	namedWindow("src", WINDOW_AUTOSIZE);
	cv::imshow("src", utsc.src);
	setMouseCallback("src", onMouse, (void*)&utsc);  //类型转换
	waitKey();
	destroyAllWindows();
	return utsc.srcTri;
}

cv::Mat getMask(cv::Mat& src, std::vector<cv::Point> mask_points) {

	std::vector<std::vector<cv::Point>> mask_area;
	mask_area.push_back(mask_points);

	polylines(src, mask_area, 1, cv::Scalar(0, 0, 0));
	cv::Mat mask;

	src.copyTo(mask);
	mask.setTo(cv::Scalar::all(0));
	fillPoly(mask, mask_area, cv::Scalar(255, 255, 255));

	return mask;
}

int main(int argc, char** argv) {

	//remnants_test(argc, argv);
	//vibe_test(argc,argv);
	//gradient_test();
	testWrapper();

	int a;
	cin >> a;
	return 0;
}

void testWrapper() {
	const char *modelPath = { "D:/vsProject/openPose/openPose/Models" };

	// 模型初始化数据以及初始化
	AlgInitSt *algInitSt = new AlgInitSt;
	algInitSt->gpuId = 0;
	algInitSt->modelsPath = const_cast<char*>(modelPath);

	void* buMingWu = InitialBuMingWuFunct(algInitSt);

	cv::VideoCapture cap;
	// "c-8.mp4"
	// rtsp://admin:1q2w#E$R@192.168.0.241:1554/Streaming/Channels/301  rtsp://admin:1q2w#E$R@192.168.0.243:554
	cap.open("rtsp://admin:1q2w#E$R@192.168.0.243:554");

	Mat frame;
	cap >> frame;
	for (int i = 0; i < 20; i++)
		cap >> frame;
	// 0:人员出现,1:越界侦测,2:超时滞留/徘徊,3:超时独处,4:离床检测,5:XXXX检测,6:睡岗检测,7:攀高,8:制服检测,
	// 9:头盔检测,10:车辆检测,11:不明物体检测,12:人员快速聚集,13:人员打架检测
	InDataSt* inDataSt = getInDataSt(frame, 11);

	while (true) {
		cap >> frame;
		if (frame.empty()) break;

		inDataSt->imgInform->dataPtr = frame.data;
		inDataSt->imgInform->width = frame.cols;
		inDataSt->imgInform->height = frame.rows;
		inDataSt->imgInform->chns = frame.channels();

		double start_time = (double)cv::getTickCount();
		ObjectBoxSurvSt* objBoxeList = new ObjectBoxSurvSt[inDataSt->numPoly];
		DoBuMingWuObjectFunct(inDataSt, buMingWu, objBoxeList);
		for (int i = 0; i < inDataSt->numPoly; i++) {
			if (objBoxeList[i].numObjs > 0) {
				for (int j = 0; j < objBoxeList[i].numObjs; j++) {
					cv::Rect rect = Rect();
					rect.x = objBoxeList[i].objsPtr[j].x*inDataSt->imgInform->width;
					rect.y = objBoxeList[i].objsPtr[j].y*inDataSt->imgInform->height;
					rect.width = objBoxeList[i].objsPtr[j].w*inDataSt->imgInform->width;
					rect.height = objBoxeList[i].objsPtr[j].h*inDataSt->imgInform->height;
					rectangle(frame, rect, Scalar(0, 0, 255), 2);
					//cout << objBoxeList[i].objsPtr[j].x*inDataSt->imgInform.width << endl;
				}
				//DrawBox(frame, pBoxResult);
				//imwrite("tmp.jpg", frame);
				std::string imgName;
				switch (objBoxeList[i].taskType)
				{
				case 3:
					std::cout << "检测到独处..." << std::endl;
					imgName = "03_" + getLocNameTime() + ".png";
					cv::imwrite(imgName, frame);
					break;
				case 4:
					std::cout << "检测到离床..." << std::endl;
					imgName = "04_" + getLocNameTime() + ".png";
					cv::imwrite(imgName, frame);
					break;
				case 7:
					std::cout << "检测到攀高..." << std::endl;
					imgName = "07_" + getLocNameTime() + ".png";
					cv::imwrite(imgName, frame);
					break;
				case 11:
					std::cout << "检测到不明物..." << std::endl;
					imgName = "11_" + getLocNameTime() + ".png";
					cv::imwrite(imgName, frame);
					break;
				case 13:
					std::cout << "检测到打架..." << std::endl;
					imgName = "13_" + getLocNameTime() + ".png";
					cv::imwrite(imgName, frame);
					break;
				default:
					std::cout << "检测到其他..." << std::endl;
					break;
				}
			}
		}
		delete[] objBoxeList;

		double end_time = (double)cv::getTickCount();
		double fps = cv::getTickFrequency() / (end_time - start_time);
		string label = "FPS:" + cv::format("%.2f", fps);
		putText(frame, label, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
		imshow("frame", frame);
		if (waitKey(1) == 27) {
			cap.release();
			destroyAllWindows();
			break;
		}
	}
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

	cv::cvtColor(frame, gray, COLOR_BGR2GRAY);
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
		cv::cvtColor(frame, gray, COLOR_BGR2GRAY);

		vibe.findFgMask(gray);
		ret = vibe.getFGMask(false);
		vt_rect.clear();
		vt_rect = vibe.getRect();


		UMat grad = getImgGradient(gray);
		double meanGrad = 0;
		meanGrad = getPartImgMeanGradient_2(grad, ret, Rect(points[0].x, points[0].y, (points[2].x - points[0].x), (points[2].y - points[0].y)));

		string gm = "GM:" + format("%.2f", meanGrad);
		cv::putText(frame, gm, Point(points[0].x, points[0].y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);

		for (int i = 0; i < points.size(); i++) {
			drawingLine(frame, points);
		}

		rectangle(frame, Rect(points[0].x, points[0].y, (points[2].x - points[0].x), (points[2].y - points[0].y)), Scalar(255, 0, 0), 2);

		cv::imshow("zone", zone_mat);
		cv::imshow("ret", ret);
		cv::imshow("frame", frame);
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
	Remnamts remn = Remnamts(0, 0.8, Size(20, 20), Size(600, 400), 123, 0.9, 5, true, 12, 200);
	remn.initSamples(gray);

	int i = 50;
	UMat mask;
	unsigned long fgNum;
	//unsigned long frameNum = 0;
	vector<SuspFgMask> vt_ret;
	string label;
	int reconnection = 3;
	while (i) {
		if (!cap.read(frame)) {
			if (reconnection == 0) {
				int sz[2] = { 200, 1000 };
				UMat end_img = UMat::zeros(2, sz, CV_8UC1);
				string end = "the video is over! Press any key to end";
				putText(end_img, end, cv::Point(5, 60), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
				imshow("end_img", end_img);
				waitKey();
				destroyAllWindows();
				return;
			}
			else {
				if (c > 1) {
					cap.open(path[1]);
				}
				else {
					cap.open(0);
				}
				cout << "重连......" << endl;
				reconnection -= 1;
				continue;
			}
		}
		reconnection = 3;
		//frameNum++;
		double start_time = (double)getTickCount();
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		remn.findFgMask(gray, zone);
		mask = remn.getFGMask();
		fgNum = remn.getFgNum();
		vt_ret.clear();
		vt_ret = remn.getResults();

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

InDataSt* getInDataSt(Mat frame, int taskType) {
	unsigned char *dataPtr = new unsigned char;
	ImageInformSurvSt* imageInformSurvSt = new ImageInformSurvSt;
	imageInformSurvSt->cameraID = 0;
	imageInformSurvSt->imgFormat = 1;
	imageInformSurvSt->codeformat = 1;
	imageInformSurvSt->dataPtr = dataPtr;
	InDataSt* inDataSt = new InDataSt;
	inDataSt->imgInform = imageInformSurvSt;

	// 添加监控区域
	float width = frame.size().width, height = frame.size().height;
	vector<cv::Point> points = getPoints(frame);
	inDataSt->numPoly = 1;
	PolySurvSt *polyPtr = new PolySurvSt[inDataSt->numPoly];
	inDataSt->polyPtr = polyPtr;
	for (int i = 0; i < inDataSt->numPoly; i++) {
		polyPtr[i].numPts = 4;
		polyPtr[i].taskID = 1;
		polyPtr[i].taskType = taskType;
		polyPtr[i].triggerLinesCount = 1;
		polyPtr[i].timeThr = 5;
		polyPtr[i].triggerLines[0].p1.x = points[0].x / width;
		polyPtr[i].triggerLines[0].p1.y = points[0].y / height;
		polyPtr[i].triggerLines[0].p2.x = points[1].x / width;
		polyPtr[i].triggerLines[0].p2.y = points[1].y / height;
		for (int j = 0; j < polyPtr[i].numPts; j++) {
			polyPtr[i].polyPts[j].x = points[j].x * 1.0 / frame.size().width;
			polyPtr[i].polyPts[j].y = points[j].y * 1.0 / frame.size().height;
		}
	}


	inDataSt->imgInform->dataPtr = frame.data;
	inDataSt->imgInform->width = frame.cols;
	inDataSt->imgInform->height = frame.rows;
	inDataSt->imgInform->chns = frame.channels();

	return inDataSt;
}