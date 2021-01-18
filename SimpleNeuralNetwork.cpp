#include "SimpleNeuralNetwork.hpp"
#include "dataset.hpp"
#include "dataloader.hpp"

typedef SimpleNet<ReLU, Softmax, CrossEntropyError> Net;

Net model(2, 3, 2);
double x[2] = { 2.35,4.97 };
double t[2] = { 1,0 };
int epoch = 1000;
double loss;

void trainSGD() {
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
	Net ai(2, 3, 2);

	// 保存したモデルをロード
	ai.load("model.txt");

	// 検証データ
	double x[2] = { 2.5,5.0 };

	// 順伝搬
	double* y = ai.predict(x);

	// 予測値
	printf("y[0] = %lf ％\n", y[0] * 100);

	// メモリの開放
	ai.del();
}

void MakeDataFunc() {
	DataSet ds;
	ds.MakeDataSet(100);
}

void LoadDataSet() {
	DataLoader dl;
	dl.test();
}

int main() {
	LoadDataSet();
	return 0;
}