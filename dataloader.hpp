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
	double** Vimages;
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
		Vimages = new double* [dataSize];
		int i, j, k, l, vectorSize = height * width * channelSize;
		for (i = 0; i < dataSize; i++) {
			Vimages[i] = new double[vectorSize];
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
		Mat img;

		int x;
		if (channelSize == 1) { x = 0; }
		else { x = 1; }

		int i = 0, j, k, l, v;
		for (const auto& f : fs::directory_iterator(dirPath)) {
			img = imread(f.path().string(), x);

			resize(img, img, Size(), (double)width / img.cols, (double)height / img.rows);

			v = 0;
			for (j = 0; j < height; j++) {
				for (k = 0; k < width; k++) {
					for (l = 0; l < img.channels(); l++) {
						imgList[i][j][k][l] = (double)img.ptr<Vec3b>(j)[k][l];
						imgList[i][j][k][l] /= 255;
						Vimages[i][v] = imgList[i][j][k][l];
						// printf("%d %d %lf\n", i, v, (double)img.ptr<Vec3b>(j)[k][l]);
						v++;
					}
				}
			}
			i++;
			//	エラーが発生するためいまはこうしている
			if (i == dataSize) { break; }
		}
		return imgList;
	}

	double** vecImg() {
		return Vimages;
	}

	void del() {
		int i, j, k;
		for (i = 0; i < dataSize; i++) {
			for (j = 0; j < height; j++) {
				for (k = 0; k < width; k++) {
					delete[] imgList[i][j][k];
				}
				delete[] imgList[i][j];
			}
			delete[] imgList[i];
			delete[] Vimages[i];
		}
		delete[] imgList;
		delete[] Vimages;
		cout << "正常に解放しました（DataLoader）" << endl;
	}

private:
	string dirPath;
	int dataSize;
	int channelSize;
	int height;
	int width;
};

#endif // !_DATALOADER_H_
