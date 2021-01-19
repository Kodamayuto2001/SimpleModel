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
	double**** imgList;
	DataLoader(
		string dirPath,
		int dataSize,
		int channelSize,
		int height,
		int width
	) {
		DataLoader::dirPath = dirPath;
		DataLoader::dataSize = dataSize;
		DataLoader::channelSize = channelSize;
		DataLoader::height = height;
		DataLoader::width = width;

		imgList = new double*** [dataSize];
		int i, j, k, l;
		for (i = 0; i < dataSize; i++) {
			imgList[i] = new double** [height];
			for (j = 0; j < height; j++) {
				imgList[i][j] = new double* [width];
				for (k = 0; k < width; k++) {
					imgList[i][j][k] = new double[channelSize];
					for (l = 0; l < channelSize; l++) {
						imgList[i][j][k][l] = 0.0;
					}
				}
			}
		}
	}

	double**** load() {
		Mat img,ch[3];
		Vec3b pix;
		
		int i = 0, j, k, l;
		for (const auto& f : fs::directory_iterator(dirPath)) {
			img = imread(f.path().string(), 0);

			resize(img, img, Size(), (double)width / img.cols, (double)height / img.rows);

			for (j = 0; j < height; j++) {
				for (k = 0; k < width; k++) {
					for (l = 0; l < channelSize; l++) {
						imgList[i][j][k][l] = (double)img.ptr<Vec3b>(j)[k][l];
					}
				}
			}
			i++;
		}
		return imgList;
	}

	void del_loadImgList(double**** used) {
		int i, j, k;
		for (i = 0; i < dataSize; i++) {
			for (j = 0; j < height; j++) {
				for (k = 0; k < width; k++) {
					delete[] used[i][j][k];
				}
				delete[] used[i][j];
			}
			delete[] used[i];
		}
		delete[] used;
	}
private:
	string dirPath;
	int dataSize;
	int channelSize;
	int height;
	int width;
};

#endif // !_DATALOADER_H_