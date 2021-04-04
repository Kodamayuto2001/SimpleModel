#ifndef _DATALOADER_HPP_
#define _DATALOADER_HPP_
#include <iostream>
#include <opencv2/core/version.hpp>
#ifdef _DEBUG
#define LIBEXT CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR) CVAUX_STR(CV_VERSION_REVISION) CV_VERSION_STATUS "d.lib"
#else
#define LIBEXT CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR) CVAUX_STR(CV_VERSION_REVISION) CV_VERSION_STATUS ".lib"
#endif
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <random>

#ifndef DATA_MAX
#define DATA_MAX 29000
#endif // !DATA_MAX
#ifndef HEIGHT
#define HEIGHT 160
#endif // !HEIGHT
#ifndef WIDTH
#define WIDTH 160
#endif // !WIDTH
#ifndef INPUT
#define INPUT HEIGHT * WIDTH
#endif // !INPUT


using namespace cv;
using namespace std;
namespace fs = filesystem;	//	C++17

typedef struct dataset {
	string path;
	int label;
};

void path_shuffle(string root, dataset train_data[DATA_MAX]) {
	int i = 0, j = 0;
	for (const auto& class_name : fs::directory_iterator(root)) {
		for (const auto& f_name : fs::directory_iterator(class_name)) {
			train_data[j].path = f_name.path().string();
			train_data[j].label = i;
			j++;
		}
		i++;
	}

	//	shuffle
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dist(0, DATA_MAX - 1);
	dataset tmp;
	int x = 0;
	for (int i = 0; i < DATA_MAX; ++i) {
		x = dist(gen);
		tmp = train_data[i];
		train_data[i] = train_data[x];
		train_data[x] = tmp;
	}
}

void test_1(string path, double vecImg[INPUT]) {
	Mat img = imread(path, 1);
	resize(img, img, Size(), (double)WIDTH / img.cols, (double)HEIGHT / img.rows);
	cvtColor(img, img, COLOR_BGR2HSV);
	Mat test_1((int)HEIGHT, (int)WIDTH, CV_8UC3);
	Mat test_2((int)HEIGHT, (int)WIDTH, CV_8UC3);
	Mat test_3((int)HEIGHT, (int)WIDTH, CV_8UC3);
	Mat test_4((int)HEIGHT, (int)WIDTH, CV_8UC3);

	for (int i = 0; i < HEIGHT; ++i) {
		for (int j = 0; j < WIDTH; ++j) {
			
			int tmp = (int)img.ptr<Vec3b>(i)[j][2];
			test_1.ptr<Vec3b>(i)[j] = Vec3b(tmp, 0, 0);
			test_2.ptr<Vec3b>(i)[j] = Vec3b(0, tmp, 0);
			test_3.ptr<Vec3b>(i)[j] = Vec3b(0,0,tmp);
			test_4.ptr<Vec3b>(i)[j] = Vec3b(tmp, tmp, tmp);
		}
	}

	imshow("1", test_1);
	imshow("2", test_2);
	imshow("3", test_3);
	imshow("4", test_4);
	waitKey(1);
}

void dataloader(string path, float vectorImg[INPUT]) {
	Mat img = imread(path, 1);

	resize(
		img, 
		img,
		Size(),
		(double)WIDTH / img.cols,
		(double)HEIGHT / img.rows
	);

	cvtColor(img, img, COLOR_BGR2HSV);

	int v = 0;
	for (int i = 0; i < HEIGHT; ++i) {
		for (int j = 0; j < WIDTH; ++j) {
			//	HSV色空間のうちV（明度）を使用
			vectorImg[v] = (float)img.ptr<Vec3b>(i)[j][2];
			//	正規化
			vectorImg[v] /= 255;
		}
	}
}

#endif // !_DATALOADER_HPP_
