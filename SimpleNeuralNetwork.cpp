#include "SimpleNeuralNetwork.hpp"
#include "dataset.hpp"
#include "dataloader.hpp"

void trainSGD() {
	typedef SimpleNet<ReLU, Softmax, CrossEntropyError> Net;
	Net model(2, 3, 2);
	double x[2] = { 2.35,4.97 };
	double t[2] = { 1,0 };
	int epoch = 1000;
	double loss;
	SGD<Net> optimizer;
	for (int e = 0; e < epoch; e++) {
		loss = model.forward(x, t);
		model.backward();
		optimizer.step(model);
		printf("loss = %lf\n", loss);
	}
	model.save("model.txt");
	model.del();
}

void trainMomentum() {
	typedef SimpleNet<ReLU, Softmax, CrossEntropyError> Net;
	Net model(2, 3, 2);
	double x[2] = { 2.35,4.97 };
	double t[2] = { 1,0 };
	int epoch = 1000;
	double loss;
	Momentum<Net> optimizer;
	for (int e = 0; e < epoch; e++) {
		loss = model.forward(x, t);
		model.backward();
		optimizer.step(model);
		printf("loss = %lf\n", loss);
	}
	model.save("model.txt");
	model.del();
}

void trainAdaGrad() {
	typedef SimpleNet<ReLU, Softmax, CrossEntropyError> Net;
	Net model(2, 3, 2);
	double x[2] = { 2.35,4.97 };
	double t[2] = { 1,0 };
	int epoch = 1000;
	double loss;
	AdaGrad<Net> optimizer;
	for (int e = 0; e < epoch; e++) {
		loss = model.forward(x, t);
		model.backward();
		optimizer.step(model);
		printf("loss = %lf\n", loss);
	}
	model.save("model.txt");
	model.del();
}

void trainRMSProp() {
	typedef SimpleNet<ReLU, Softmax, CrossEntropyError> Net;
	Net model(2, 3, 2);
	double x[2] = { 2.35,4.97 };
	double t[2] = { 1,0 };
	int epoch = 1000;
	double loss;
	RMSProp<Net> optimizer;
	for (int e = 0; e < epoch; e++) {
		loss = model.forward(x, t);
		model.backward();
		optimizer.step(model);
		printf("loss = %lf\n", loss);
	}
	model.save("model.txt");
	model.del();
}

void trainAdam() {
	typedef SimpleNet<ReLU, Softmax, CrossEntropyError> Net;
	Net model(2, 3, 2);
	double x[2] = { 2.35,4.97 };
	double t[2] = { 1,0 };
	int epoch = 1000;
	double loss;
	Adam<Net> optimizer;
	for (int e = 0; e < epoch; e++) {
		loss = model.forward(x, t);
		model.backward();
		optimizer.step(model);
		printf("loss = %lf\n", loss);
	}
	model.save("model.txt");
	model.del();
}

void test() {
	typedef SimpleNet<ReLU, Softmax, CrossEntropyError> Net;
	Net ai(2, 3, 2);

	// �ۑ��������f�������[�h
	ai.load("model.txt");

	// ���؃f�[�^
	double x[2] = { 2.5,5.0 };

	// ���`��
	double* y = ai.predict(x);

	// �\���l
	printf("y[0] = %lf ��\n", y[0] * 100);

	// �������̊J��
	ai.del();
}

void MakeDataFunc() {
	DataSet ds;
	ds.MakeDataSet(100);
}

void LoadDataSet() {
	DataLoader dl("DataSet/",100,3,160,160);
	double**** tmp = dl.load();
	cout << "�摜��ǂݍ��߂܂���" << endl;
	dl.del_loadImgList(tmp);
}


int main() {
	LoadDataSet();
	return 0;
}