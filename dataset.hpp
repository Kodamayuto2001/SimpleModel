#ifndef _DATASET_H_
#define _DATASET_H_
#include <opencv2/core/version.hpp>
#ifdef _DEBUG
#define LIBEXT CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR) CVAUX_STR(CV_VERSION_REVISION) CV_VERSION_STATUS "d.lib"
#else
#define LIBEXT CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR) CVAUX_STR(CV_VERSION_REVISION) CV_VERSION_STATUS ".lib"
#endif
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <filesystem>
using namespace std;
using namespace cv;

/****************************************************************************************
	DataSet	class	データセットを作成します。

	コンストラクタの引数
	int dataMax			データセットの枚数
	string cascadePath	カスケード特徴分類器のパス
	char*  saveDir		データセットを保存するディレクトリ
****************************************************************************************/
class DataSet {
public:
	void MakeDataSet(
		int dataMax = 100,
		string cascadePath = "haarcascades\\haarcascade_frontalface_alt.xml",
		char* saveDir = (char*)"DataSet"
	) {
		// C++17
		namespace fs = std::filesystem;
		bool isExist = fs::create_directory(saveDir);
		if (isExist == 0) {
			cout << "ディレクトリは作成していません。" << endl;
		}
		else {
			cout << "ディレクトリを作成しました。" << endl;
		}
		CascadeClassifier cascade;
		if (!cascade.load(cascadePath)) {
			cout << "カスケード分類器を読み込めませんでした" << endl;
		}

		vector<Rect> faces;
		VideoCapture cap(0);
#ifndef CV_CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_WIDTH 3
#endif // !CV_CAP_PROP_FRAME_WIDTH
#ifndef CV_CAP_PROP_FRAME_HEIGHT
#define CV_CAP_PROP_FRAME_HEIGHT 4
#endif // !CV_CAP_PROP_FRAME_HEIGHT
#ifndef CV_RGB2GRAY
#define CV_RGB2GRAY 7
#endif // !CV_RGB2GRAY
#ifndef CV_AA
#define CV_AA 16
#endif // !CV_AA
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
		Mat img, imgGray;
		string strCnt;
		int i, cnt = 0;

		while (1)
		{
			cap >> img;
			cvtColor(img, imgGray, CV_RGB2GRAY);
			cascade.detectMultiScale(imgGray, faces, 1.1, 3, 0, Size(100, 100));
			for (i = 0; i < faces.size(); i++) {
				rectangle(img,
					Point(faces[i].x, faces[i].y),
					Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
					Scalar(0, 0, 255),
					3,
					CV_AA
				);
				cnt++;
				Mat tri(imgGray, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height));
				imgGray = tri;
			}
			if (cnt > dataMax) { break; }
			strCnt = cntFunc(dataMax, cnt);
			imshow("img", img);
			imwrite(saveDir + (string)"\\" + strCnt + ".jpg", imgGray);

			const int key = waitKey(1);
			if (key == 'q') { break; }
		}
	}

private:
	size_t z = 1;
	bool isFirst = true;

	string cntFunc(int dataMax, int cnt) {
		string result = "";
		if (isFirst == true) {
			isFirst = false;
			int a = dataMax;
			while (1)
			{
				if (a < 10) {
					break;
				}
				a /= 10;
				z++;
			}
		}
		for (int i = 0; i < z; i++) {
			if ((cnt / (dataMax / (int)(pow(10, i))) >= 10)) {
				cnt = cnt % 10;
			}
			result += (char)((cnt / (dataMax / (int)(pow(10, i)))) + 48);
		}
		return result;
	}
};

#endif // !_DATASET_H_