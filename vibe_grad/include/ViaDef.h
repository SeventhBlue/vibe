#ifndef __VIA_DEF_H__
#define __VIA_DEF_H__

/**********************************************************************/
/********************************V0.1**********************************/
/**********************************************************************/

#include <stdint.h>
#include <vector>
#define NUM_SURVEIL 20
#define NUM_MAX_PTS 20
#define MAX_NUM_OBJS 100

extern "C"
{
	/*---------------------------*/
	//目标框结构体
	typedef struct    //真实坐标
	{
		int x, y, w, h;  //目标的box左上角坐标及宽高
		int obj_id;      //类别,越线计数结果:0正向,1反向
		float prob;		 //置信度
	}BoxSurvSt;

	typedef struct  //相对坐标(0~1)
	{
		float x, y, w, h;  //目标的box左上角坐标及宽高
		int obj_id;        //类别,越线计数结果:0正向,1反向
		float prob;		   //置信度
	}FBoxSurvSt;
	/*---------------------------*/

	/*---------------------------*/
	//坐标点结构体
	typedef struct
	{
		float x;
		float y;
	}FPointSurvSt;

	//线结构体
	typedef struct
	{
		FPointSurvSt p1;
		FPointSurvSt p2;
	}FLineSurvSt;
	/*---------------------------*/

	/*---------------------------*/
	//长方形或目标大小结构体
	typedef struct   //（x,y）表示坐上顶点；表示目标框大小时x,y无效
	{
		int x, y;
		int width, height;
	}RectSurvSt;

	//长方形结构体
	typedef struct   //相对坐标
	{
		float x, y;
		float width, height;
	}FRectSurvSt;
	/*---------------------------*/

	/*---------------------------*/
	//图片信息结构体
	typedef struct
	{
		int cameraID;	  //摄像头id
		int imgIndex;     //图片序号
		__int64 dataTime; // 图片采集时间 ms
		bool isAllKeyFrame;//是否全为关键帧，true表示分析服务器只解关键帧。

		int imgFormat;  //0=yuv, 1=rgb, 2=二进制, 3=图片路径
		int codeformat; // 对应数据格式的编码方式,yuv:0=YUV420P, 1=NV12, 2=NV21; rgb:0=rgbrgbrgb,1=bgrbgr,2=rrrgggbbb

		int width, height, chns; //图像宽度、高度、通道数量
		unsigned char *dataPtr;	 //图片数据

	} ImageInformSurvSt;


	//多边形布控点结构体
	typedef struct
	{
		int taskID;   //任务id
		int taskType; //0:人员出现,1:越界侦测,2:超时滞留/徘徊,3:超时独处,4:离床检测,5:脱岗检测,6:睡岗检测,7:攀高,8:制服检测,
					  //9:头盔检测,10:车辆检测,11:不明物体检测,12:人员快速聚集,13:人员打架检测

		int numPts;		 //每块布控区域所包含的点数，大于2
		FPointSurvSt polyPts[NUM_MAX_PTS]; //布控区域点系列

		int triggerLinesCount;  //触发线数量
		FLineSurvSt triggerLines[5]; //触发线(攀高线、翻墙线), 越线检测取第一条线

		int numFalseBx;	//人形:误检框数量
		FBoxSurvSt *falseBoxes; //误检框,实时传输所有的误检框

		float  threshold;  //	算法阈值(0 - 1)
		float	timeThr;  //	时间阈值，超过则告警(单位：ms)

	} PolySurvSt;
	/*---------------------------*/

	/*******************************************************************/
	//初始化所需要的参数
	typedef struct //初始化单一算法接口，算法内部接口
	{
		int gpuId;//0,1,2,3

		char *modelsPath;  //模型路径

		int maxW, maxH;   //输入图像的最大宽和高

		int frameInterval;

	}AlgInitSt;


	typedef struct  //初始化多算法接口，分析服务接口
	{
		//initial num of algorithm
		int taskListType[NUM_SURVEIL]; //每种算法占用一个元素，0=该算法无效，1=该算法有效;0:人员出现,1:越界侦测,2:超时滞留/徘徊,3:超时独处,4:离床检测,5:脱岗检测,6:睡岗检测,
									   //7:攀高,8:制服检测,9:头盔检测,10:车辆检测,11:不明物体检测，12:人员快速聚集,13:人员打架检测
		int gpuId[NUM_SURVEIL];		   //每种算法占用一个元素，对应的值表示gpu索引号

		char *modelsPath;  //模型路径

		int maxW, maxH;    //输入图像的最大宽和高

		int frameInterval; //帧间隔(ms, 需要设置成40的倍数), 默认为0, 表示每帧都发

	} ConfigParamSurvSt;
	/*******************************************************************/


	/*******************************************************************/
	//调用检测函数所用的实时数据
	typedef struct   //单一算法检测函数接口，算法内部接口
	{
		ImageInformSurvSt *imgInform;//输入图像信息

		FRectSurvSt detRoi; // 最大矩形

		int numPoly;   //布控区域数量
		PolySurvSt *polyPtr;

		FRectSurvSt minObj, maxObj;  //物体最小/最大尺寸

		std::vector<int> index;//使用该算法的多边形区域索引号
	}InDataSt;

	typedef struct  //多算法检测函数接口，分析服务接口
	{
		ImageInformSurvSt imgInform; //输入图像信息

		int numPoly; //布控区域数量
		PolySurvSt polyPtr[NUM_MAX_PTS];

		//检测阈值参数信息

		FRectSurvSt minObj, maxObj; //物体最小/最大尺寸

		//float threshold[NUM_MAX_PTS];
		//float timeThr[NUM_MAX_PTS];
	} InDataSurvSt;
	/*******************************************************************/

	/*******************************************************************/
	//每个任务(与布控区域对应)所获得的目标结果
	typedef struct   //现对坐标
	{
		int cameraID; //摄像头id   从输入直接拷贝
		int taskID;   //任务id，从输入直接拷贝
		int taskType;
		int numObjs;					   //目标的个数（0表示无事件）
		FBoxSurvSt objsPtr[MAX_NUM_OBJS];  //目标的boxes，无目标为空指针，
	} ObjectBoxSurvSt;

	typedef struct   	//获得识别结构，真实坐标（绝对坐标）
	{
		BoxSurvSt objsPtr[MAX_NUM_OBJS]; //目标的Boxes
		int numObjs;                           //目标的个数（0表示无事件）

	}SurvResBoxSt;   //输出时，需要转换为相对坐标



  //识别结果结构体，一张图片可能有多个任务(与布控区域对应)
	typedef struct
	{
		int numPoly; //布控区域个数，一个布控区域对应一个任务
		int hasUpdateNum;//已经跟新区域
		ObjectBoxSurvSt objBoxeList[NUM_SURVEIL];

		ImageInformSurvSt *imgOutform; //输出图像信息

	} RecResultSurvSt;
	/*******************************************************************/
}

#endif