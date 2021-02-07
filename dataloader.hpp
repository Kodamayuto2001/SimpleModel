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

void dataloader(string dirPath, double vecImg[DATAMAX][INPUT_SIZE]) {
	Mat img;
	int v = 0, i = 0;
	for (const auto& f : fs::directory_iterator(dirPath)) {
		img = imread(f.path().string(), 0);
		resize(img, img, Size(), (double)IMG_WIDTH / img.cols, (double)IMG_HEIGHT / img.rows);
		v = 0;
#pragma omp parallel for
		for (int j = 0; j < IMG_HEIGHT; ++j) {
			for (int k = 0; k < IMG_WIDTH; ++k) {
				for (int l = 0; l < CHANNEL; ++l) {
					vecImg[i][v] = (double)img.ptr<Vec3b>(j)[k][l] / 255;
					v++;
				}
			}
		}
		i++;
	}
}

#endif // !_DATALOADER_H_