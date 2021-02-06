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

#ifndef DATAMAX
#define DATAMAX		100
#endif // !DATAMAX
#ifndef CHANNEL
#define CHANNEL		1
#endif // !CHANNEL
#ifndef IMG_HEIGHT
#define IMG_HEIGHT	160
#endif // !IMG_HEIGHT
#ifndef IMG_WIDTH
#define IMG_WIDTH	160
#endif // !IMG_WIDTH
#ifndef INPUT_SIZE
#define INPUT_SIZE  CHANNEL*IMG_HEIGHT*IMG_WIDTH
#endif // !INPUT_SIZE

void dataloader(string dirPath, int dataSize, int channelSize, int img_height, int img_width, double vecImg[DATAMAX][INPUT_SIZE]) {
	Mat img;
	int v = 0, i = 0;
	for (const auto& f : fs::directory_iterator(dirPath)) {
		img = imread(f.path().string(), 0);
		resize(img, img, Size(), (double)img_width / img.cols, (double)img_height / img.rows);
		v = 0;
		for (int j = 0; j < img_height; ++j) {
			for (int k = 0; k < img_width; ++k) {
				for (int l = 0; l < channelSize; ++l) {
					vecImg[i][v] = (double)img.ptr<Vec3b>(j)[k][l] / 255;
					v++;
				}
			}
		}
		i++;
	}
}

#endif // !_DATALOADER_H_