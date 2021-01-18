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
//class TEST {
//public:
//	void test0() {
//		const char* imagename = "lena.jpg";
//		Mat img = imread(imagename);
//		imshow("image test", img);
//		waitKey(1000);
//	}
//	void test1() {
//		Mat img = imread("lena.jpg");
//		imshow("Original img", img);
//		waitKey(1000);
//		resize(img, img, Size(), 0.5, 0.5);
//		imshow("Resized img", img);
//		waitKey(1000);
//	}
//	void test2() {
//		Mat img = imread("lena.jpg", IMREAD_UNCHANGED);
//		Mat imgGray;
//		cvtColor(img, imgGray, CV_RGB2GRAY);
//		string cascadeName = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
//		CascadeClassifier cascade;
//		cascade.load(cascadeName);
//		vector<Rect> faces;
//		cascade.detectMultiScale(imgGray, faces, 1.1, 3, 0, Size(20, 20));
//		
//		for (int i = 0; i < faces.size(); i++) //���o������̌�"faces.size()"�����[�v���s��
//		{
//			rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 0, 255), 3, CV_AA); //���o�������ԐF��`�ň͂�
//		}
//		cout << "ok!!!" << endl;
//	}
//	void test3() {
//		Mat img = imread("lena.jpg");
//		// 0�s��
//		Vec3b* src = img.ptr<Vec3b>(0);
//		// 0�Ԗ�
//		src[0];
//		cout << src[0] << endl;
//
//		printf("img height: %d\n", img.rows);
//		printf("img width : %d\n", img.cols);
//	}
//	void test4() {
//		Mat img = imread("lena.jpg");
//		Vec3b* src;
//		for (int i = 0; i < img.rows; i++) {
//			src = img.ptr<Vec3b>(i);
//			for (int j = 0; j < img.cols; j++) {
//				cout << src[j] << endl;
//			}
//		}
//	}
//	void test5() {
//		// filesystem ���g���ɂ́AVisualC++�̌����C++17�Ƃ���K�v������
//		namespace fs = std::filesystem;
//		fs::path path = fs::current_path();
//		// cout << path << endl;
//		cout << path.string() << endl;
//	}
//	void test6() {
//		VideoCapture cap(0);
//		cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
//		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
//		while (1)
//		{
//			Mat frame;
//			cap >> frame;
//			imshow("img", frame);
//			const int key = waitKey(1);
//			if (key == 'q') {
//				break;
//			}
//		}
//	}
//	void test7() {
//		VideoCapture cap(0);
//		cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
//		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
//		const string CASCADE_PATH = "haarcascades/haarcascade_frontalface_default.xml";
//		CascadeClassifier cascade;
//		if (!cascade.load(CASCADE_PATH))
//			cout << "�ǂݍ��߂܂���ł����B" << endl;
//		
//		Mat img,imgGray;
//		vector<Rect> facerect;
//		while (1)
//		{
//			cap >> img;
//			cvtColor(img, imgGray, CV_RGB2GRAY);
//			// cascade.detectMultiScale(imgGray, facerect, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(200, 200));
//
//			imshow("img", img);
//
//			const int key = waitKey(1);
//			if (key == 'q') {
//				break;
//			}
//		}
//	}
//	void test8() {
//		// �猟�o�Ώۂ̉摜�f�[�^�p
//		IplImage* tarImg;
//
//		// ���o�Ώۂ̉摜�t�@�C���p�X
//		char tarFilePath[] = "lena.jpg";
//
//		// �摜�f�[�^�̓ǂݍ���
//		tarImg = cvLoadImage(tarFilePath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
//
//		// ���ʊ猟�o��̓ǂݍ���
//		CvHaarClassifierCascade* cvHCC = (CvHaarClassifierCascade*)cvLoad("haarcascade_frontalface_default.xml");
//
//		// ���o�ɕK�v�ȃ������X�g���[�W��p�ӂ���
//		CvMemStorage* cvMStr = cvCreateMemStorage(0);
//
//		// ���o�����󂯎�邽�߂̃V�[�P���X��p�ӂ���
//		CvSeq* face;
//
//		// �摜�����猟�o�Ώۂ̏����擾����
//		face = cvHaarDetectObjects(tarImg, cvHCC, cvMStr);
//
//		for (int i = 0; i < face->total; i++) {
//			CvRect* faceRect = (CvRect*)cvGetSeqElem(face, i);
//
//			// �擾������̈ʒu���Ɋ�Â��A��`�`����s��
//			cvRectangle(tarImg,
//				cvPoint(faceRect->x, faceRect->y),
//				cvPoint(faceRect->x + faceRect->width, faceRect->y + faceRect->height),
//				CV_RGB(255, 0, 0),
//				3, CV_AA);
//		}
//
//		// ��ʒu�ɋ�`�`����{�����摜��\��
//		cvNamedWindow("face_detect");
//		cvShowImage("face_detect", tarImg);
//
//		// �L�[���͑҂�
//		cvWaitKey(0);
//
//		// �E�B���h�E�̔j��
//		cvDestroyWindow("face_detect");
//
//		// �p�ӂ����������X�g���[�W�����
//		cvReleaseMemStorage(&cvMStr);
//
//		// �J�X�P�[�h���ʊ�̉��
//		cvReleaseHaarClassifierCascade(&cvHCC);
//
//		// �C���[�W�̉��
//		cvReleaseImage(&tarImg);
//	}
//	void test9() {
//		const char* imagename = "lena.jpg";
//		Mat img = imread(imagename);
//		if (img.empty()) 
//			cout << "�摜��ǂݍ��߂܂���ł���" << endl;
//
//		double scale = 4.0;
//		Mat gray, smallImg(saturate_cast<int>(img.rows / scale), saturate_cast<int>(img.cols / scale), CV_8UC1);
//		// �O���[�X�P�[���摜�ɕϊ�
//		cvtColor(img, gray, CV_BGR2GRAY);
//		// �������ԒZ�k�̂��߂ɉ摜���k��
//		resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
//		equalizeHist(smallImg, smallImg);
//
//		// ���ފ�̓ǂݍ���
//		std::string cascadeName = "haarcascades/haarcascade_frontalface_default.xml";
//		//std::string cascadeName = "./lbpcascade_frontalface.xml"; // LBP
//		CascadeClassifier cascade;
//		if (!cascade.load(cascadeName))
//			cout << "�ǂݍ��߂܂���ł���" << endl;
//
//		std::vector<Rect> faces;
//		/// �}���`�X�P�[���i��j�T��xo
//		// �摜�C�o�͋�`�C�k���X�P�[���C�Œ��`���C�i�t���O�j�C�ŏ���`
//		cascade.detectMultiScale(smallImg, faces,
//			1.1, 2,
//			CV_HAAR_SCALE_IMAGE,
//			Size(30, 30));
//
//		// ���ʂ̕`��
//		std::vector<Rect>::const_iterator r = faces.begin();
//		for (; r != faces.end(); ++r) {
//			Point center;
//			int radius;
//			center.x = saturate_cast<int>((r->x + r->width * 0.5) * scale);
//			center.y = saturate_cast<int>((r->y + r->height * 0.5) * scale);
//			radius = saturate_cast<int>((r->width + r->height) * 0.25 * scale);
//			circle(img, center, radius, Scalar(80, 80, 255), 3, 8, 0);
//		}
//
//		namedWindow("result", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
//		imshow("result", img);
//		waitKey(0);
//	}
//	void test10() {
//		Mat img = imread("lena.jpg");
//		string cascadeFile = "haarcascades/haarcascade_frontalface_default.xml";
//
//		Mat detectFaceImage = detectFaceInImage(img, cascadeFile);
//		imshow("detect face", detectFaceImage);
//		waitKey(1000);
//	}
//	Mat detectFaceInImage(Mat& image, string& cascade_file) {
//		CascadeClassifier cascade;
//		cascade.load(cascade_file);
//
//		vector<Rect> faces;
//		cascade.detectMultiScale(image, faces, 1.1, 3, 0, Size(20, 20));
//
//		for (int i = 0; i < faces.size(); i++) {
//			rectangle(image, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 200, 0), 3, CV_AA);
//		}
//		return image;
//	}
//	void test11() {
//		Mat img = imread("lena.jpg", IMREAD_UNCHANGED);
//		Mat imgGray;
//		cvtColor(img, imgGray, CV_RGB2GRAY);
//		string cascadeName = "haarcascades\\haarcascade_frontalface_alt.xml";
//		CascadeClassifier cascade;
//		if (!cascade.load(cascadeName)) {
//			cout << "�ǂݍ��߂܂���ł���" << endl;
//		}
//		vector<Rect> faces;
//		cascade.detectMultiScale(imgGray, faces, 1.1, 3, 0, Size(20, 20));
//
//		cout << faces[0] << endl;		// [76 x 76 from (175, 72)]
//		cout << faces[0].size() << endl;// [76 x 76]
//		cout << faces[0].x << endl;		// 175
//		cout << faces[0].y << endl;		// 72
//		cout << faces[1] << endl;		// [57 x 57 from (188, 86)]
//		cout << faces[2] << endl;		// [84 x 84 from (171, 75)]
//		
//		// [���]���o������̌��������Ă���
//		cout << faces.size() << endl;	// 18446743881536058925
//
//		int i = 0;
//		rectangle(img,
//			Point(faces[i].x, faces[i].y),
//			Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
//			Scalar(0, 0, 255),
//			3,
//			CV_AA
//		);
//		imshow("detect face", img);
//		waitKey(1000);
//		cout << "ok!!!" << endl;
//	}
//	void test12() {
//		VideoCapture cap(0);
//		cap.set(CV_CAP_PROP_FRAME_WIDTH, 192);
//		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 108);
//		const string cascadeName = "haarcascades\\haarcascade_frontalface_alt.xml";
//		CascadeClassifier cascade;
//		if (!cascade.load(cascadeName)) {
//			cout << "�ǂݍ��߂܂���ł���" << endl;
//		}
//		Mat img, imgGray;
//		vector<Rect> facerect;
//		int i = 0;
//		while (1)
//		{
//			cap >> img;
//			cvtColor(img, imgGray, CV_RGB2GRAY);
//			cascade.detectMultiScale(imgGray, facerect, 1.1, 3, 0, Size(20, 20));
//			rectangle(img,
//				Point(facerect[i].x, facerect[i].y),
//				Point(facerect[i].x + facerect[i].width, facerect[i].y + facerect[i].height),
//				Scalar(0, 0, 255),
//				3,
//				CV_AA
//			);
//			imshow("detect face", img);
//			const int key = waitKey(1);
//			if (key == 'q') {
//				break;
//			}
//		}
//	}
//	// https://dixq.net/forum/viewtopic.php?t=10263
//	// �Ȃ��J�X�P�[�h���ފ킪�����Ȃ������̂�
//	// Debug���[�h�ŃR���p�C������Ƃ��܂��������A
//	// Release���[�h���Ƃ��܂�����
//	void test13() {
//		VideoCapture cap(0);
//		cap.set(CV_CAP_PROP_FRAME_WIDTH, 192);
//		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 108);
//		/*const string cascadeName = "haarcascades\\haarcascade_frontalface_alt.xml";
//		CascadeClassifier cascade;
//		if (!cascade.load(cascadeName)) {
//			cout << "�ǂݍ��߂܂���ł���" << endl;
//		}*/
//		Mat img, imgGray;
//		/*vector<Rect> facerect;
//		int i = 0;*/
//		while (1)
//		{
//			cap >> img;
//			cvtColor(img, imgGray, CV_RGB2GRAY);
//			//cascade.detectMultiScale(imgGray, facerect, 1.1, 3, 0, Size(20, 20));
//
//			//cout << facerect.size() << endl;
//			//for (int i = 0; i < facerect.size(); i++) {
//			//	rectangle(img,
//			//		Point(facerect[i].x, facerect[i].y),
//			//		Point(facerect[i].x + facerect[i].width, facerect[i].y + facerect[i].height),
//			//		Scalar(0, 0, 255),
//			//		3,
//			//		CV_AA
//			//	);
//			//}
//			
//			//imshow("Result", img);
//			const int key = waitKey(1);
//			if (key == 'q') {
//				break;
//			}
//		}
//	}
//	void test14() {
//		string cascadeName = "haarcascades\\haarcascade_frontalface_alt.xml";
//		CascadeClassifier cascade;
//		if (!cascade.load(cascadeName)) {
//			cout << "�ǂݍ��߂܂���ł���" << endl;
//		}
//		vector<Rect> faces;
//
//		VideoCapture cap(0);
//		cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
//		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
//		Mat img, imgGray;
//		int i;
//		while (1)
//		{
//			cap >> img;
//
//			cvtColor(img, imgGray, CV_RGB2GRAY);
//
//			
//			cascade.detectMultiScale(imgGray, faces, 1.1, 3, 0, Size(100, 100));
//
//			for (i = 0; i < faces.size(); i++) {
//				rectangle(img,
//					Point(faces[i].x, faces[i].y),
//					Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
//					Scalar(0, 0, 255),
//					3,
//					CV_AA
//				);
//			}
//
//			imshow("img", img);
//			const int key = waitKey(1);
//			if (key == 'q') {
//				break;
//			}
//		}
//	}
//	void test15() {
//		// VisualC++ = C++17
//		namespace fs = std::filesystem;
//		bool result = fs::create_directory("saveDir");
//		if (result == 0) {
//			cout << "�f�B���N�g����V���ɍ쐬�ł��܂���ł���" << endl;
//		}
//		else {
//			cout << "�V�����f�B���N�g�����쐬�ł��܂���" << endl;
//		}
//	}
//	void test16() {
//		// VisualC++ = C++17
//		const char* saveDir = "DataSet";
//		namespace fs = std::filesystem;
//		bool result = fs::create_directory(saveDir);
//		if (result == 0) {
//			cout << "�f�B���N�g����V���ɍ쐬�ł��܂���ł���" << endl;
//		}
//		else {
//			cout << "�V�����f�B���N�g�����쐬�ł��܂���" << endl;
//		}
//
//		string cascadeName = "haarcascades\\haarcascade_frontalface_alt.xml";
//		CascadeClassifier cascade;
//		if (!cascade.load(cascadeName)) {
//			cout << "�ǂݍ��߂܂���ł���" << endl;
//		}
//		vector<Rect> faces;
//
//		VideoCapture cap(0);
//		cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
//		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
//		Mat img, imgGray;
//		int i;
//		while (1)
//		{
//			cap >> img;
//
//			cvtColor(img, imgGray, CV_RGB2GRAY);
//
//
//			cascade.detectMultiScale(imgGray, faces, 1.1, 3, 0, Size(100, 100));
//
//			for (i = 0; i < faces.size(); i++) {
//				rectangle(img,
//					Point(faces[i].x, faces[i].y),
//					Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
//					Scalar(0, 0, 255),
//					3,
//					CV_AA
//				);
//			}
//
//			imshow("img", img);
//			const int key = waitKey(1);
//			if (key == 'q') {
//				break;
//			}
//		}
//	}
//	void test17() {
//		//int i = 0;
//		//char cnt = '0';
//		//cout << cnt << endl;
//		//cout << (char)(i + 1 + 48) << endl;
//		//cout << (char)(i + 2 + 48) << endl;
//		//cout << (char)(i + 3 + 48) << endl;
//		//cout << (char)(i + 4 + 48) << endl;
//		//cout << (char)(i + 5 + 48) << endl;
//		//cout << (char)(i + 6 + 48) << endl;
//		//cout << (char)(i + 7 + 48) << endl;
//		//cout << (char)(i + 8 + 48) << endl;
//		//cout << (char)(i + 9 + 48) << endl;
//		//cout << (char)(i + 10 + 48) << endl;
//		//char a, b, c;
//		//int i = 0;
//		//int j = 0;
//		//string result;
//		//for (i = 0; i < 100; i++) {
//		//	if (i < 10) {
//		//		a = (char)(i + 48);
//		//		b = (char)(0);
//		//	}
//		//	else if (i >= 10 && i < 100) {
//		//		if ((i % 10) == 0) {
//		//			j++;
//		//		}
//		//		a = (char)((i % 10) + 48);
//		//		b = (char)(j + 48);
//		//	}
//		//	
//		//	result = b;
//		//	result += a;
//		//	cout << result << endl;
//		//}
//		int dataMax = 1000;
//		int cnt = 13;
//		size_t z = 1;
//		int tmp = dataMax;
//		while (1)
//		{
//			if (tmp < 10) {
//				break;
//			}
//			tmp /= 10;
//			z++;
//		}
//		string result = "";
//		int x = 0;
//
//
//		for (int i = 0; i < z; i++) {
//			if ((cnt / (dataMax / (int)(pow(10, i))) >= 10)) {
//				cnt = cnt % 10;
//			}
//			result += (char)((cnt / (dataMax / (int)(pow(10, i)))) + 48);
//		}
//		cout << result << endl;
//	}
//	void test18() {
//		// VisualC++ = C++17
//		const char* saveDir = "DataSet";
//		namespace fs = std::filesystem;
//		bool result = fs::create_directory(saveDir);
//		if (result == 0) {
//			cout << "�f�B���N�g����V���ɍ쐬�ł��܂���ł���" << endl;
//		}
//		else {
//			cout << "�V�����f�B���N�g�����쐬�ł��܂���" << endl;
//		}
//
//		string cascadeName = "haarcascades\\haarcascade_frontalface_alt.xml";
//		CascadeClassifier cascade;
//		if (!cascade.load(cascadeName)) {
//			cout << "�ǂݍ��߂܂���ł���" << endl;
//		}
//		vector<Rect> faces;
//
//		VideoCapture cap(0);
//		cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
//		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
//		Mat img, imgGray;
//		int i;
//		int cnt = 0;
//		int dataMax = 100;
//		string strCnt;
//		while (1)
//		{
//			cap >> img;
//
//			cvtColor(img, imgGray, CV_RGB2GRAY);
//
//
//			cascade.detectMultiScale(imgGray, faces, 1.1, 3, 0, Size(100, 100));
//
//			for (i = 0; i < faces.size(); i++) {
//				rectangle(img,
//					Point(faces[i].x, faces[i].y),
//					Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
//					Scalar(0, 0, 255),
//					3,
//					CV_AA
//				);
//			}
//			
//			imshow("img", img);
//			strCnt = cntFunc(dataMax, cnt);
//			cout << strCnt << endl;
//			imwrite(saveDir + (string)"\\" + strCnt + ".jpg", img);
//
//			if (cnt < dataMax) {
//				cnt++;
//			}
//			else {
//				break;
//			}
//
//			const int key = waitKey(1);
//			if (key == 'q') {
//				break;
//			}
//		}
//	}
//	string cntFunc(int dataMax, int cnt) {
//		string result = "";
//		size_t z = 1;
//		int a = dataMax;
//		while (1)
//		{
//			if (a < 10) {
//				break;
//			}
//			a /= 10;
//			z++;
//		}
//		for (int i = 0; i < z; i++) {
//			if ((cnt / (dataMax / (int)(pow(10, i))) >= 10)) {
//				cnt = cnt % 10;
//			}
//			result += (char)((cnt / (dataMax / (int)(pow(10, i)))) + 48);
//		}
//		return result;
//	}
//};


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
			cout << "�f�B���N�g���͍쐬���Ă��܂���B" << endl;
		}
		else {
			cout << "�f�B���N�g�����쐬���܂����B" << endl;
		}
		CascadeClassifier cascade;
		if (!cascade.load(cascadePath)) {
			cout << "�J�X�P�[�h���ފ��ǂݍ��߂܂���ł���" << endl;
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
		Mat img, imgGray,resultImg;
		string strCnt;
		int i, cnt = 0;
		
		while (1)
		{
			cap >> img;
			cap >> resultImg;
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
				if (cnt <= dataMax) { cnt++; }
				Mat tri(resultImg, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height));
				resultImg = tri;
			}
			if (cnt > dataMax) { break; }

			strCnt = cntFunc(dataMax,cnt);
			imshow("img", img);
			imwrite(saveDir + (string)"\\" + strCnt + ".jpg", resultImg);

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
