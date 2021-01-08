#include <iostream>
#include <float.h>
#include <random>
#include <cmath>
using namespace std;
class Add {
public:
	double* result;
	Add() {}
	~Add() {
		delete[] result;
	}
	double forward(double a, double b) {
		return a + b;
	}
	double* backward(double dout) {
		result = new double[2];
		result[0] = dout * 1.0;
		result[1] = dout * 1.0;
		return result;
	}
private:
};
class Mul {
public:
	double* result;
	double x;
	double y;
	Mul() {}
	~Mul() {
		delete[] result;
	}
	double forward(double a, double b) {
		x = a;
		y = b;
		return a * b;
	}
	double* backward(double dout) {
		result = new double[2];
		result[0] = dout * y;
		result[1] = dout * x;
		return result;
	}
private:
};
class Div {
public:
	double y;
	Div() {
		y = NULL;
	}
	~Div() {}
	double forward(double x) {
		// ゼロ除算回避
		if (x > DBL_MIN / C) {
			y = 1 / x;
		}
		else if (x < -DBL_MIN / C) {
			y = 1 / x;
		}
		else {
			isInf = true;
			y = DBL_MAX / C;
		}
		return y;
	}
	double backward(double dout) {
		if (isInf == true) {
			return dout * (-DBL_MAX / C);
		}
		return  dout * (-1 * y * y);
	}
private:
	bool isInf = false;
	const double C = 1.0e+100;
};
class Exp {
public:
	double out;
	Exp() {
		out = NULL;
	}
	~Exp() {}
	double forward(double a) {
		out = exp(a);
		return out;
	}
	double backward(double dout) {
		return dout * out;
	}
private:
};
class Log {
public:
	double x;
	Log() {
		x = NULL;
	}
	~Log() {}
	double forward(double x) {
		if (x <= DBL_MIN) {
			__isInf = true;
			x = DBL_MIN;
		}
		Log::x = x;
		return log(x);
	}
	double backward(double dout) {
		if (__isInf == true) {
			return dout * (DBL_MAX / C);
		}
		return dout * (1 / x);
	}
private:
	bool __isInf = false;
	const double C = 1.0e+100;
};
class Sigmoid {
public:
	Sigmoid() {}
	~Sigmoid() {}
	double forward(double x) {
		if (x <= -SIGMOID_RANGE) {
			return DBL_MIN;
		}
		if (x >= SIGMOID_RANGE) {
			return 1.0 - DBL_MIN;
		}
		double a = mul.forward(x, -1);
		double b = exp.forward(a);
		double c = add.forward(b, 1);
		double d = div.forward(c);

		return d;
	}
	double backward(double dout) {
		double ddx = div.backward(dout);
		double* dc = add.backward(ddx);
		double dbx = exp.backward(dc[0]);
		double* da = mul.backward(dbx);

		return da[0];
	}
private:
	const double SIGMOID_RANGE = DBL_MAX;
	Mul mul;
	Exp exp;
	Add add;
	Div div;
};
class Softmax {
public:
	double* y;
	double* result;

	Softmax() {}
	~Softmax() {
		// 動的にメモリを作成したので配列の先頭ポインタを削除
		delete[] exps;
		delete[] adds;
		delete[] muls;
		delete[] y;
		delete[] result;
	}
	double* forward(double* x, size_t size) {
		exps = new Exp[size];
		adds = new Add[size];
		muls = new Mul[size];
		y = new double[size];
		double* exp_a = new double[size];
		double exp_sum = 0.0;
		double exp_div;

		// オーバーフロー対策
		double Cmax = x[0];
		for (int i = 0; i < (int)size; i++) {
			if (Cmax < x[i]) {
				Cmax = x[i];
			}
		}

		// ソフトマックス関数の分子分母
		for (int i = 0; i < (int)size; i++) {
			exp_a[i] = exps[i].forward(x[i] - Cmax);
			exp_sum = adds[i].forward(exp_sum, exp_a[i]);
		}

		// ソフトマックス関数の分母
		exp_div = div.forward(exp_sum);

		for (int i = 0; i < (int)size; i++) {
			y[i] = muls[i].forward(exp_a[i], exp_div);
		}
		delete[] exp_a;
		return y;
	}

	double* backward(double* dout, size_t size) {
		double** dexp_as = new double* [size];
		double** dexp_asdiv = new double* [size];
		double* dexp_a = new double[size];
		result = new double[size];
		double dexp_sum = 0.0;
		double dexp_div;
		double tmp;

		for (int i = 0; i < (int)size; i++) {
			dexp_as[i] = muls[i].backward(dout[i]);
			dexp_a[i] = dexp_as[i][0];
			dexp_sum += dexp_as[i][1];
			delete[] dexp_as[i];
		}
		delete[] dexp_as;
		dexp_div = div.backward(dexp_sum);
		for (int i = 0; i < (int)size; i++) {
			dexp_asdiv[i] = adds[i].backward(dexp_div);
			tmp = dexp_asdiv[i][1] + dexp_a[i];
			tmp = exps[i].backward(tmp);
			result[i] = tmp;
			delete[] dexp_asdiv[i];
		}
		delete[] dexp_asdiv;
		delete[] dexp_a;

		return result;
	}

private:
	Exp* exps;
	Add* adds;
	Div div;
	Mul* muls;
};
class CrossEntropyError {
public:
	double* result;
	double E = 0.0;

	CrossEntropyError() {}
	~CrossEntropyError() {
		delete[] logs;
		delete[] muls;
		delete[] adds;
		delete[] result;
	}

	double forward(double* x, double* t, size_t x_size) {
		size = x_size;
		logs = new Log[size];
		muls = new Mul[size];
		adds = new Add[size];
		double log_x;
		double m;
		E = 0.0;
		for (int i = 0; i < (int)size; i++) {
			// アンダーフロー回避
			if (x[i] <= delta && x[i] >= -delta) {
				x[i] = delta;
			}
			log_x = logs[i].forward(x[i]);
			// cout << log_x << endl;
			m = muls[i].forward(log_x,t[i]);
			// cout << m << endl;
			E = adds[i].forward(E, m);
			// cout << E << endl;
		}
		E = mul.forward(-1, E);
		// cout << E << endl;
		return E;
	}
	double* backward(double dout) {
		result = new double[size];
		double* tmp;
		double dlog_x;

		double* de = mul.backward(dout);
		for (int i = 0; i < size; i++) {
			tmp = adds[i].backward(de[1]);
			tmp = muls[i].backward(tmp[1]);
			dlog_x = logs[i].backward(tmp[0]);
			result[i] = dlog_x;
		}
		return result;
	}

private:
	Log* logs;
	Mul* muls;
	Add* adds;
	Mul mul;
	int size;
	const double delta = 1e-10;
};
class SimpleNet {
public:
	double* y;
	double _loss;
	double** dweight2;
	double** dweight1;
	double* dbias2;
	double* dbias1;
	double*** params;

	SimpleNet(
		int input_size,
		int hidden_size,
		int output_size) {
		_input_size = input_size;
		_hidden_size = hidden_size;
		_output_size = output_size;

		// 重み配列の作成（入力層と中間層の間）
		weight1 = new double* [hidden_size];
		for (int i = 0; i < hidden_size; i++) {
			weight1[i] = new double[input_size];
		}

		// 重み配列の作成（中間層と出力層の間）
		weight2 = new double* [output_size];
		for (int i = 0; i < output_size; i++) {
			weight2[i] = new double[hidden_size];
		}

		// 中間層の入力ノードに対するバイアス
		bias1 = new double[hidden_size];

		// 出力層の入力ノードに対するバイアス
		bias2 = new double[output_size];

		// 中間層の出力ノード
		hidden_out = new double[hidden_size];

		// 出力層の出力ノード
		output_out = new double[output_size];

		// 最終的な出力
		y = new double[output_size];

		// ランダムに初期化
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<double> dist(-1, 1);

		// 重み1をランダムに初期化
		for (int i = 0; i < hidden_size; i++) {
			for (int j = 0; j < input_size; j++) {
				weight1[i][j] = dist(gen);
				// printf("weight1[%d][%d] = %lf\n", i, j, weight1[i][j]);
			}
		}

		// 重み2をランダムに初期化
		for (int i = 0; i < output_size; i++) {
			for (int j = 0; j < hidden_size; j++) {
				weight2[i][j] = dist(gen);
				// printf("weight2[%d][%d] = %lf\n", i, j, weight2[i][j]);
			}
		}

		// バイアス1を0で初期化
		for (int i = 0; i < hidden_size; i++) {
			bias1[i] = 0.0;
			// printf("bias1[%d] = %lf\n", i, bias1[i]);
		}

		// バイアス2を0で初期化
		for (int i = 0; i < output_size; i++) {
			bias2[i] = 0.0;
			// printf("bias2[%d] = %lf\n", i, bias2[i]);
		}
		cout << "初期化終了" << endl;
	}
	~SimpleNet() {
		// 動的に割り当てたメモリを開放する
		for (int i = 0; i < _hidden_size; i++) {
			delete weight1[i];
			delete[] muls_fc1[i];
			delete[] adds1_fc1[i];
			delete[] dweight1[i];
		}

		for (int i = 0; i < _output_size; i++) {
			delete weight2[i];
			delete[] muls_fc2[i];
			delete[] adds1_fc2[i];
			delete[] dweight2[i];
		}
		delete[] weight1;
		delete[] weight2;
		delete[] bias1;
		delete[] bias2;
		delete[] hidden_out;
		delete[] output_out;
		delete[] muls_fc1;
		delete[] adds1_fc1;
		delete[] adds2_fc1;
		delete[] sigmoids_fc1;
		delete[] muls_fc2;
		delete[] adds1_fc2;
		delete[] y;
		delete[] dweight2;
		delete[] dbias2;
		delete[] dhidden_out;
		delete[] dweight1;
		delete[] dbias1;
		
		//delete params[0][0];
		//delete params[0][1];
		//delete params[0][2];
		//delete params[1][0];
		//delete params[1][1];
		//delete params[2][0];
		//delete params[2][1];

		//delete[] params[0];
		//delete[] params[1];
		//delete[] params[2];

		//delete[] params;
		cout << "ごめん開放できてない。" << endl;
	}
	double* predict(double* x) {
		fc1(x);
		fc2();
		y = softmax.forward(output_out, _output_size);
		return y;
	}

	double loss(double* x, double* t) {
		double* y = predict(x);
		_loss = cee.forward(y, t,_output_size);
		cout << "順伝搬完了" << endl;
		return _loss;
	}

	void backward(double dout) {
		// 損失関数の逆伝搬
		double* result_arr = cee.backward(dout);
		// ソフトマックス関数の逆伝搬
		result_arr = softmax.backward(result_arr,_output_size);
		// fc2の逆伝搬
		dfc2(result_arr);
		// fc1の逆伝搬
		dfc1();
	}

	double*** get_dW1_dW2_db1_db2(void) {
		params = new double** [3];
		params[2] = new double* [2];
		params[0] = dweight1;
		params[1] = dweight2;
		params[2][0] = dbias1;
		params[2][1] = dbias2;
		return params;
	}

private:
	double** weight1;
	double** weight2;
	double* bias1;
	double* bias2;
	double* hidden_out;
	double* output_out;
	int _input_size;
	int _hidden_size;
	int _output_size;
	Mul** muls_fc1;
	Add** adds1_fc1;
	Add* adds2_fc1;
	Sigmoid* sigmoids_fc1;
	Mul** muls_fc2;
	Add** adds1_fc2;
	Add* adds2_fc2;
	Softmax softmax;
	CrossEntropyError cee;
	double* dhidden_out;

	void fc1(double* x) {

		muls_fc1 = new Mul * [_hidden_size];
		adds1_fc1 = new Add * [_hidden_size];

		for (int i = 0; i < _hidden_size; i++) {
			muls_fc1[i] = new Mul[_input_size];
			adds1_fc1[i] = new Add[_input_size];
		}

		adds2_fc1 = new Add[_hidden_size];
		sigmoids_fc1 = new Sigmoid[_hidden_size];

		double tmp;
		double a;
		for (int i = 0; i < _hidden_size; i++) {
			tmp = 0.0;
			for (int j = 0; j < _input_size; j++) {
				a = muls_fc1[i][j].forward(x[j], weight1[i][j]);
				tmp = adds1_fc1[i][j].forward(tmp, a);
			}
			hidden_out[i] = adds2_fc1[i].forward(tmp, bias1[i]);
			hidden_out[i] = sigmoids_fc1[i].forward(hidden_out[i]);
		}
	}

	void fc2(void) {
		muls_fc2 = new Mul * [_output_size];
		adds1_fc2 = new Add * [_output_size];
		for (int i = 0; i < _output_size; i++) {
			muls_fc2[i] = new Mul[_hidden_size];
			adds1_fc2[i] = new Add[_hidden_size];
		}
		adds2_fc2 = new Add[_output_size];

		double tmp;
		double a;
		for (int i = 0; i < _output_size; i++) {
			tmp = 0.0;
			for (int j = 0; j < _hidden_size; j++) {
				a = muls_fc2[i][j].forward(hidden_out[j], weight2[i][j]);
				tmp = adds1_fc2[i][j].forward(tmp, a);
			}
			output_out[i] = adds2_fc2[i].forward(tmp, bias2[i]);
		}
	}

	void dfc2(double* dout) {
		// 動的にメモリを割り当てる
		dbias2 = new double[_output_size];
		dweight2 = new double* [_output_size];
		for (int i = 0; i < _output_size; i++) {
			dweight2[i] = new double[_hidden_size];
		}
		dhidden_out = new double[_hidden_size];
		double* dtmp_bias2;
		double* dtmp_da;
		double* dhidden_out_dweight2;
		for (int i = 0; i < _hidden_size; i++) {
			dhidden_out[i] = 0.0;
		}
		for (int i = 0; i < _output_size; i++) {
			dtmp_bias2 = adds2_fc2[i].backward(dout[i]);
			dbias2[i] = dtmp_bias2[1];
			for (int j = 0; j < _hidden_size; j++) {
				dtmp_da = adds1_fc2[i][j].backward(dtmp_bias2[0]);
				dhidden_out_dweight2 = muls_fc2[i][j].backward(dtmp_da[1]);
				// printf("%lf   %lf\n", dhidden_out_dweight2[0], dhidden_out_dweight2[1]);
				dweight2[i][j] = dhidden_out_dweight2[1];
				dhidden_out[j] += dhidden_out_dweight2[0];
				// printf("dweight2[%d][%d] = %lf\n", i, j, dweight2[i][j]);
			}
		}
	}

	void dfc1(void) {
		dbias1 = new double[_hidden_size];
		dweight1 = new double* [_hidden_size];
		for (int i = 0; i < _hidden_size; i++) {
			dweight1[i] = new double[_input_size];
		}
		double tmp;
		double* dtmp_bias1;
		double* dtmp_da;
		double* dx_dweight1;
		for (int i = 0; i < _hidden_size; i++) {
			tmp = sigmoids_fc1[i].backward(dhidden_out[i]);
			dtmp_bias1 = adds2_fc1[i].backward(tmp);
			dbias1[i] = dtmp_bias1[1];
			for (int j = 0; j < _input_size; j++) {
				dtmp_da = adds1_fc1[i][j].backward(dtmp_bias1[0]);
				// Runtime Library
				// Debug Assetion Failed!
				// Expression: _CrtlsVaildHeapPointer(block)
				// メモリ破壊によるエラー
				// コードのどこかでメモリ破壊をしているプログラムがある
				dx_dweight1 = muls_fc1[i][j].backward(dtmp_da[1]);
				dweight1[i][j] = dx_dweight1[1];
				//printf("muls_fc1[%d][%d] = %lf\n", i, j, dweight1[i][j]);
			}
		}
		cout << "逆伝搬終了" << endl;
	}
};
#pragma region TEST

void testAdd(void) {
	Add add;
	double y = add.forward(1, 2);
	double* douts = add.backward(1);
	printf("%lf\n", y);
	printf("%lf  %lf\n", douts[0], douts[1]);
}

void testMul(void) {
	Mul mul;
	double y = mul.forward(2, 3);
	double* douts = mul.backward(1);
	printf("%lf\n", y);
	printf("%lf  %lf\n", douts[0], douts[1]);
}

void testDiv(void) {
	//cout << "doubleの最大値 : " << DBL_MAX << endl;
	//cout << "doubleの最小値 : " << -DBL_MAX << endl;
	//cout << "doubleの浮動小数点で表現できる最小値" << DBL_MIN << endl;
	//cout << 1 / DBL_MIN << endl;
	Div div;
	double y = div.forward(0);
	double dx = div.backward(1);
	cout << "y = " << y << endl;
	cout << "dx = " << dx << endl;
}

void testExp(void) {
	Exp myexp;
	double y = myexp.forward(1);
	double dx = myexp.backward(1);
	cout << y << endl;
	cout << dx << endl;
}

void testLog(void) {
	Log log;
	double y = log.forward(0);
	double dx = log.backward(10);
	cout << y << endl;
	cout << dx << endl;
}

void testSigmoid(void) {
	Sigmoid sigmoid;
	double y = sigmoid.forward(10);
	double dx = sigmoid.backward(1);
	cout << y << endl;
	cout << dx << endl;
}

void testSoftmax(void) {
	Softmax softmax;
	double arr[10] = {
		1.0,
		1.5,
		2.1,
		9.1,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		8.8
	};
	double darr[10] = {
		-1.5,
		-1.4,
		-1.3,
		-1.2,
		-1.1,
		-0.5,
		-0.0,
		-0.2,
		0.98,
		15.3
	};
	double* y = softmax.forward(arr, 10);
	double* dx = softmax.backward(darr, 10);

	for (int i = 0; i < 10; i++) {
		printf("y[%d] = %lf\n", i, y[i]);
		//printf("dx[%d] = %lf\n", i, dx[i]);
	}
}

void testCrossEntropyError(void) {
	CrossEntropyError cee;
	double x[2] = { 1.23,3.0 };
	double t[2] = { 1,0 };

	double E = cee.forward(x,t,2);
	double* result = cee.backward(1);
	//cout << E << endl;
}

void testPointer(void) {
	/*double* p = new double[1000];
	cout << "ok" << endl;
	delete[] p;*/
	double** p;
	p = new double* [10];
	for (int i = 0; i < 10; i++) {
		p[i] = new double[20];
	}
	cout << "ok" << endl;
}

void testSimpleNet(void) {
	SimpleNet model(2, 3, 2);
	double x[2] = { 1.0,2.0 };
	double t[2] = { 1,0 };

	// 順伝搬そして損失値
	double loss = model.loss(x, t);

	// 逆伝搬
	model.backward(1.0);

	// 勾配減少
	double*** params = model.get_dW1_dW2_db1_db2();
	//// 重み１の勾配
	//cout << params[0][0][0] << endl;
	//cout << params[0][0][1] << endl;
	//cout << params[0][1][0] << endl;
	//cout << params[0][1][1] << endl;
	//cout << params[0][2][0] << endl;
	//cout << params[0][2][1] << endl;
	//// 重み２の勾配
	//cout << params[1][0][0] << endl;
	//cout << params[1][0][1] << endl;
	//cout << params[1][0][2] << endl;
	//cout << params[1][1][0] << endl;
	//cout << params[1][1][1] << endl;
	//cout << params[1][1][2] << endl;
	//// バイアス１の勾配
	//cout << params[2][0][0] << endl;
	//cout << params[2][0][1] << endl;
	//cout << params[2][0][2] << endl;
	//// バイアス２の勾配
	//cout << params[2][1][0] << endl;
	//cout << params[2][1][1] << endl;

}

void testMul2(void) {
	//Mul** mul_test;
	//mul_test = new Mul * [10000000];
	//for (int i = 0; i < 10000000; i++) {
	//	mul_test[i] = new Mul[500000];
	//}
	cout << "ok!" << endl;
	// 問題はアドレスを開放するときに発生している
	//for (int i = 0; i < 10; i++) {
	//	delete[] mul_test[i];
	//}
	//delete[] mul_test;
	cout << "testMul2" << endl;
}

#pragma endregion



int main(void) {
	// モデルのインスタンス化
	SimpleNet model(2,3,2);

	// 教師データ
	double x[2] = { 1.0,2.0 };

	// 正解ラベル
	double t[2] = { 1,0 };

	// 損失値
	double loss = model.loss(x, t);

	// 誤差逆伝搬
	model.backward(1.0);

	// パラメータ
	double*** params = model.get_dW1_dW2_db1_db2();
	
	return 0;
}







