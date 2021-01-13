#include "SimpleNeuralNetwork.hpp"

void train() {
	SimpleNet<ReLU, Softmax, CrossEntropyError> model(2, 3, 2);
	SGD<SimpleNet<ReLU, Softmax, CrossEntropyError>> optimizer(0.01);
	
	double x[2] = { 2.35,4.97 };	// 教師データ
	double t[2] = { 1,0 };			// 正解ラベル

	// 学習回数
	int epoch = 1000;

	double loss;
	for (int e = 0; e < epoch; e++) {
		// 順伝播
		loss = model.forward(x, t);

		// 逆伝播
		model.backward(1.0);

		// 更新
		optimizer.step(model);

		// 損失値
		printf("loss = %lf\n", loss);
	}

	// 学習したモデルのパラメータを保存
	model.save("model.txt");

	// メモリの解放
	model.del();
}

void test() {
	SimpleNet<ReLU, Softmax, CrossEntropyError> ai(2, 3, 2);

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

int main() {
	train();
	test();
	return 0;
}