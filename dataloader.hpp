#ifndef _DATALOADER_H_
#define _DATALOADER_H_
#include <iostream>

#include <opencv2/core/version.hpp>
#ifdef _DEBUG
#define LIBEXT CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR) CVAUX_STR(CV_VERSION_REVISION) CV_VERSION_STATUS "d.lib"
#else
#define LIBEXT CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR) CVAUX_STR(CV_VERSION_REVISION) CV_VERSION_STATUS ".lib"
#endif

#include <opencv2/opencv.hpp>
#include <filesystem>
using namespace cv;
using namespace std;
namespace fs = filesystem;


class DataLoader {
public:
	void test(string dirPath = "DataSet/") {
		namespace fs = filesystem;
		Mat img;
		for (const auto& f : fs::directory_iterator(dirPath)) {
			img = imread(f.path().string(), 1);
			imshow("img", img);
			waitKey(100);
		}
	}
};

#endif // !_DATALOADER_H_
