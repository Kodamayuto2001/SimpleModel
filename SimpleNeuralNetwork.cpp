#include "SimpleNeuralNetwork.hpp"

void train() {
	// モデルのインスタンス化（入力層、中間層、出力層）
	SimpleNet model(2, 3, 2);

	// 最適化アルゴリズムのインスタンス化（学習率）
	SGD optimizer(0.01);

	double x[2] = { 2.35,4.97 };	// 教師データ
	double t[2] = { 1,0 };			// 正解ラベル

	// 学習回数
	int epoch = 1000;

	double loss;
	for (int e = 0; e < epoch; e++) {
		// 順伝搬
		loss = model.Loss(x, t);

		// 逆伝搬
		model.backward();

		// 更新
		optimizer.step(model);

		// 損失値
		printf("loss = %lf\n", loss);
	}

	// 学習したモデルのパラメータを保存
	model.save("model.txt");

	// メモリの開放
	model.del();
}

void test() {
	// モデルのインスタンス化
	SimpleNet model(2, 3, 2);

	// 保存したモデルをロード
	model.load("model.txt");

	// 検証データ
	double x[2] = { 2.5,5.0 };

	// 順伝搬
	double* y = model.predict(x);
	
	// 予測値
	printf("y[0] = %lf ％\n",y[0]*100);
	
	// メモリの開放
	model.del();
}

int main(void) {
	train();
	test();
	return 0;
}