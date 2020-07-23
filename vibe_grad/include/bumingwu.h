#ifndef __BUMINGWU_DETECT_HEADER_H__
#define __BUMINGWU_DETECT_HEADER_H__

#define BUMINGWU_HAS_DLL
#if defined (BUMINGWU_HAS_DLL)
#    if defined (BUMINGWU_BUILD_DLL)
#      define BUMINGWU_API __declspec (dllexport)
#    else
#      define BUMINGWU_API  __declspec (dllimport)
#	endif
#else
#  define BUMINGWU_API
#endif

#include "ViaDef.h"

extern "C"
{
	/*-----------------------初始化--------------------------
	输入参数：
	ParamSt *paramPtr：初始化所需要的一些参数，包括算法类型、gpu和模型路径等
	输出结果：
	void *handle：返回一个空指针类型数据，用于保存算法所需要的一些内存
	-------------------------------------------------------------*/
	BUMINGWU_API void *InitialBuMingWuFunct(AlgInitSt *paramPtr);


	/*------------------------------内存释放------------------------------
	输入参数：
	void *handle：指向算法所申请的内存
	输出结果：
	返回为空
	-----------------------------------------------------------------------------*/
	BUMINGWU_API void ReleaseBuMingWuFunct(void *handle);


	/*------------------------------检测/识别------------------------------
	输入参数：
	InDataSt *inDataPtr：实时输入数据，包括图片信息、布控区域和一些阈值信息
	void *handle：指向算法所分配的内存
	OutDataSt *outDataPtr：图像经过算法识别后所得到的结果
	输出参数：
	int flag：返回算法运行的状态，是否异常
	---------------------------------------------------------------------------*/
	BUMINGWU_API int DoBuMingWuObjectFunct(InDataSt *inDataPtr, void *handle, ObjectBoxSurvSt *objBoxeList);


	/*-----------------------------删除任务ID所对应的资源---------------------------
	输入参数：
	InDataSt *inDataPtr：输入实时信息，主要是传入任务id
	void *handle：指向算法所分配的内存
	输出结果：
	空
	-------------------------------------------------------------------------------------------*/
	BUMINGWU_API void DeleteBuMingWuTaskFunct(InDataSt *inDataPtr, void *handle);
}
#endif