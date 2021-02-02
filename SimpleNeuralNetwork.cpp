#include "dataset.hpp"
#include "dataloader.hpp"
#include "FastSimpleNeuralNetwork.hpp"

void MakeDataFunc() {
	DataSet ds;
	ds.MakeDataSet(100);
}

void NetTest() {
	typedef SimpleNeuralNetwork<FastSigmoid, FastSoftmaxWithLoss> Net;
	Net model(2, 3, 2);
	double x[2] = { 2.35,4.97 };
	double t[2] = { 1,0 };
	FastAdam<Net> optimizer;
	for (int e = 0; e < 10000; ++e) {
		model.forward(x, t);
		model.backward();
		optimizer.step(&model);
		cout << model.loss << endl;
	}
	model.save("model.kodamayuto");

	model.load("model.kodamayuto");
	double z[2] = { 2.5,5.0 };
	model.predict(z);
	cout << model.node[1][0] * 100 << "%" << endl;

	model.del();
}

void trainSGD() {
	int dataMax = 100;
	int channel = 1;
	int imgHeight = 160;
	int imgWidth = 160;
	int hiddenNeuron = 320;
	int outSize = 10;
	double lr = 0.05;

	typedef SimpleNeuralNetwork<FastSigmoid, FastSoftmaxWithLoss> Net;
	DataLoader dl("DataSet/", dataMax, channel, imgHeight, imgWidth);
	Net model(
		move(imgHeight*imgWidth*channel),
		move(hiddenNeuron),
		move(outSize)
	);
	FastSGD<Net> optimizer(lr);

	dl.load();
	double** x = dl.vecImg();
	double t[10] = { 0,0,0,0,0,0,0,1,0,0 };

	for (int e = 0; e < 10; ++e) {
		for (int i = 0; i < 80; ++i) {
			model.forward(x[i], t);
			model.backward();
			optimizer.step(&model);
			cout << model.loss << endl;
		}
	}
	model.save();
	model.del();
	dl.del();
}

void trainMomentum() {
	int dataMax = 100;
	int channel = 1;
	int imgHeight = 160;
	int imgWidth = 160;
	int hiddenNeuron = 320;
	int outSize = 10;
	double lr = 0.05;
	double momentum = 0.9;

	typedef SimpleNeuralNetwork<FastSigmoid, FastSoftmaxWithLoss> Net;
	DataLoader dl("DataSet/", dataMax, channel, imgHeight, imgWidth);
	Net model(
		move(imgHeight * imgWidth * channel),
		move(hiddenNeuron),
		move(outSize)
	);
	FastMomentum<Net> optimizer(lr, momentum);

	dl.load();
	double** x = dl.vecImg();
	double t[10] = { 0,0,0,0,0,0,0,1,0,0 };

	for (int e = 0; e < 10; ++e) {
		for (int i = 0; i < 80; ++i) {
			model.forward(x[i], t);
			model.backward();
			optimizer.step(&model);
			cout << model.loss << endl;
		}
	}
	model.save();
	model.del();
	dl.del();
}

void trainAdaGrad() {
	int dataMax = 100;
	int channel = 1;
	int imgHeight = 160;
	int imgWidth = 160;
	int hiddenNeuron = 320;
	int outSize = 10;
	double lr = 0.05;

	typedef SimpleNeuralNetwork<FastSigmoid, FastSoftmaxWithLoss> Net;
	DataLoader dl("DataSet/", dataMax, channel, imgHeight, imgWidth);
	Net model(
		move(imgHeight * imgWidth * channel),
		move(hiddenNeuron),
		move(outSize)
	);
	FastAdaGrad<Net> optimizer(lr);

	dl.load();
	double** x = dl.vecImg();
	double t[10] = { 0,0,0,0,0,0,0,1,0,0 };

	for (int e = 0; e < 10; ++e) {
		for (int i = 0; i < 80; ++i) {
			model.forward(x[i], t);
			model.backward();
			optimizer.step(&model);
			cout << model.loss << endl;
		}
	}
	model.save();
	model.del();
	dl.del();
}

void trainRMSProp() {
	int dataMax = 100;
	int channel = 1;
	int imgHeight = 160;
	int imgWidth = 160;
	int hiddenNeuron = 320;
	int outSize = 10;

	typedef SimpleNeuralNetwork<FastSigmoid, FastSoftmaxWithLoss> Net;
	DataLoader dl("DataSet/", dataMax, channel, imgHeight, imgWidth);
	Net model(
		move(imgHeight * imgWidth * channel),
		move(hiddenNeuron),
		move(outSize)
	);
	FastRMSProp<Net> optimizer;

	dl.load();
	double** x = dl.vecImg();
	double t[10] = { 0,0,0,0,0,0,0,1,0,0 };

	for (int e = 0; e < 10; ++e) {
		for (int i = 0; i < 80; ++i) {
			model.forward(x[i], t);
			model.backward();
			optimizer.step(&model);
			cout << model.loss << endl;
		}
	}
	model.save();
	model.del();
	dl.del();
}

void trainAdam() {
	int dataMax = 100;
	int channel = 1;
	int imgHeight = 160;
	int imgWidth = 160;
	double lr = 0.0001;
	double beta1 = 0.9;
	double beta2 = 0.999;
	int hiddenNeuron = 320;
	int outSize = 10;

	typedef SimpleNeuralNetwork<FastReLU, FastSoftmaxWithLoss> Net;
	DataLoader dl("DataSet/", dataMax, channel, imgHeight, imgWidth);
	dl.load();
	double** x = dl.vecImg();
	double t[10] = { 0,0,0,0,0,0,0,1,0,0 };

	Net model(
		move(imgHeight * imgWidth * channel),
		move(hiddenNeuron),
		move(outSize)
	);
	FastAdam<Net> optimizer(lr, beta1, beta2);

	for (int e = 0; e < 10; ++e) {
		for (int i = 0; i < 80; ++i) {
			model.forward(x[i], t);
			model.backward();
			optimizer.step(&model);
			cout << model.loss << endl;
		}
	}
	model.save("model.kodamayuto");

	model.load("model.kodamayuto");
	model.predict(x[96]);
	cout << model.node[1][7] * 100 << "%" << endl;
	model.del();
	dl.del();
}

void testSigmoid() {
	int dataMax = 100;
	int channel = 1;
	int imgHeight = 160;
	int imgWidth = 160;
	int hiddenNeuron = 320;
	int outSize = 10;
	typedef SimpleNeuralNetwork<FastSigmoid, FastSoftmaxWithLoss> Net;
	DataLoader dl("DataSet/", dataMax, channel, imgHeight, imgWidth);
	Net model(
		move(imgHeight * imgWidth * channel),
		move(hiddenNeuron),
		move(outSize)
	);
	dl.load();
	double** x = dl.vecImg();
	model.load("model.kodamayuto");
	model.predict(x[83]);
	for (int i = 0; i < 10; ++i) {
		cout << model.node[1][i] * 100 << "%" << endl;
	}
	model.del();
	dl.del();
}

void testReLU() {
	int dataMax = 100;
	int channel = 1;
	int imgHeight = 160;
	int imgWidth = 160;
	int hiddenNeuron = 320;
	int outSize = 10;
	typedef SimpleNeuralNetwork<FastReLU, FastSoftmaxWithLoss> Net;
	DataLoader dl("DataSet/", dataMax, channel, imgHeight, imgWidth);
	dl.load();
	double** x = dl.vecImg();
	double t[10] = { 0,0,0,0,0,0,0,1,0,0 };
	Net model(
		move(imgHeight * imgWidth * channel),
		move(hiddenNeuron),
		move(outSize)
	);
	
	model.load("model.kodamayuto");
	model.predict(x[83]);
	for (int i = 0; i < 10; ++i) {
		cout << model.node[1][i] * 100 << "%" << endl;
	}
	model.del();
	dl.del();
}

void flattenTest(void) {
	Flatten f;
	int* y = f.vec(5, 30);
	int* y2 = f.vec(10, 30);
	for (int i = 0; i < 30; ++i) {
		cout << y[i] << "   " << y2[i] << endl;
	}
}

void test() {
	int x[2] = { 1,2 };
	int w[2][3] = { {1,2,3},{2,3,4} };
	int b[3] = { 0,0,0 };
	int c[3] = { 0,0,0 };

	//x[0] * w[0][0];
	//x[1] * w[1][0];
	//x[0] * w[0][1];
	//x[1] * w[1][1];
	//x[0] * w[0][2];
	//x[1] * w[1][2];
	
	//c[0] = x[0] * w[0][0] + x[1] * w[1][0];
	//c[1] = x[0] * w[0][1] + x[1] * w[1][1];
	//c[2] = x[0] * w[0][2] + x[1] * w[1][2];

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 3; ++j) {
			c[j] += x[i] * w[i][j];
		}
		c[i] += b[i];
	}
}

int main() {
	trainRMSProp();
	return 0;
}