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
	DataLoader dl("DataSet/", 100, 3, 160, 160);
	double**** tmp = dl.load();
	double** vectorImg = dl.vecImg();
	cout << "画像を読み込めました" << endl;
	dl.del();
}

void trainFromDataSet() {
	DataLoader dl("DataSet/", 100, 3, 160, 160);
	double** x = dl.vecImg();
	double t[2] = { 1,0 };

	typedef SimpleNet<Sigmoid, Softmax, CrossEntropyError> Net;
	Net model(3 * 160 * 160, 320, 2);

	int epoch = 40;
	double loss;
	Adam<Net> optimizer;
	for (int e = 0; e < epoch; e++) {
		for (int i = 0; i < 100; i++) {
			loss = model.forward(x[i], t);
			model.backward();
			optimizer.step(model);
			printf("loss = %lf\n", loss);
		}
	}
	model.save("model.txt");
	model.del();
}

int main() {
	trainFromDataSet();
	return 0;
}