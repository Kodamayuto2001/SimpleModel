#ifndef _H_
#define _H_
#include<iostream>
#include<fstream>
#include<float.h>
#include<string>
#include<random>
#include<cmath>
using namespace std;
const double C = 1.0e+100;
const double D = 1.0e+200;

/*-----どんなに難しい数式も合成関数でできている-----*/
/*-----基礎関数クラス-----*/
class Add {
public:
	double result[2];
	double forward(double a, double b) {
		return a + b;
	}
	double* backward(double dout) {
		result[0] = dout * 1.0;
		result[1] = dout * 1.0;
		return result;
	}
};
class Mul {
public:
	double result[2];
	double forward(double a, double b) {
		x = a;
		y = b;
		if (a * b == float(INFINITY)) {
			return DBL_MAX / C;
		}
		return a * b;
	}
	double* backward(double dout) {
		result[0] = dout * y;
		result[1] = dout * x;
		return result;
	}
private:
	double x = 0.0;
	double y = 0.0;
};
class Div {
public:
	double y = 0.0;
	double forward(double x) {
		y = 1 / x;
		// 0を除算することはできない
		if (y == float(INFINITY)) {
			y = DBL_MAX / D;
		}
		return y;
	}
	double backward(double dout) {
		return dout * (-1 * y * y);
	}
};
class Exp {
public:
	double out = 0.0;
	double forward(double a) {
		out = exp(a);
		// ネイピアの数の二乗なのでオーバーフローが発生しやすい
		// オーバーフロー対策
		if (out == float(INFINITY)) {
			out = DBL_MAX / C;
		}
		return out;
	}
	double backward(double dout) {
		return dout * out;
	}
};
class Log {
public:
	double x = 1.0;
	double forward(double x) {
		// 極小値のときアンダーフローが発生する
		// アンダーフロー対策
		if (x <= DBL_MIN) {
			isInf = true;
			x = DBL_MIN;
		}
		Log::x = x;
		return log(x);
	}
	double backward(double dout) {
		if (isInf == true) {
			return dout * (DBL_MAX / C);
		}
		return dout * (1 / x);
	}
private:
	bool isInf = false;
};


/*-----活性化関数-----*/
/*-----基礎関数クラスを利用-----*/
class Sigmoid {
public:
	double forward(double x) {
		double f;
		f = mul.forward(x, -1.0);
		f = exp.forward(f);
		f = add.forward(f, 1.0);
		f = div.forward(f);
		return f;
	}
	double backward(double dout) {
		double ddx = div.backward(dout);
		double* dc = add.backward(ddx);
		double dbx = exp.backward(dc[0]);
		double* da = mul.backward(dbx);
		return da[0];
	}
	double forward2(double x) {
		y = 1.0 / (1.0 + std::exp(-x));
		return y;
	}
	double backward2(double dout) {
		return dout * (1.0 - y) * y;
	}
private:
	Mul mul;
	Exp exp;
	Add add;
	Div div;
	double y;
};
class ReLU {
public:
	double forward(double x) {
		ReLU::x = x;
		if (x > 0) { return x; }
		else { return 0; }
	}
	double backward(double dout) {
		if (x > 0) { return dout; }
		else { return 0; }
	}
private:
	double x;
};
class Softmax {
public:
	double* y;
	double* dr;

	double* forward(double* x, size_t size) {
		exps = new Exp[size];
		adds = new Add[size];
		muls = new Mul[size];
		y = new double[size];
		Softmax::size = size;
		double* exp_a = new double[size];
		double exp_sum = 0.0;
		double exp_div;

		double Cmax = x[0];
		for (int i = 0; i < (int)size; i++) {
			if (Cmax < x[i]) {
				Cmax = x[i];
			}
		}

		for (int i = 0; i < (int)size; i++) {
			exp_a[i] = exps[i].forward(x[i] - Cmax);
			exp_sum = adds[i].forward(exp_sum, exp_a[i]);
		}

		exp_div = div.forward(exp_sum);

		for (int i = 0; i < (int)size; i++) {
			y[i] = muls[i].forward(exp_a[i], exp_div);
		}
		delete[] exp_a;
		return y;
	}

	double* backward(double* dout) {
		double* dexp_a = new double[size];
		double dexp_sum = 0.0;

		for (int i = 0; i < (int)size; i++) {
			dexp_a[i] = *(muls[i].backward(dout[i]) + 0);
			dexp_sum += *(muls[i].backward(dout[i]) + 1);
		}

		double dexp_div = div.backward(dexp_sum);

		double tmp;
		dr = new double[size];
		for (int i = 0; i < (int)size; i++) {
			tmp = *(adds[i].backward(dexp_div) + 1) + dexp_a[i];
			tmp = exps[i].backward(tmp);
			dr[i] = tmp;
		}

		delete[] dexp_a;

		return dr;
	}

	void del() {
		delete[] exps;
		delete[] adds;
		delete[] muls;
	}
private:
	Exp* exps;
	Add* adds;
	Div div;
	Mul* muls;
	size_t size;
};
class CrossEntropyError {
public:
	double* dx;

	double forward(double* x, double* t, size_t size) {
		logs = new Log[size];
		muls = new Mul[size];
		adds = new Add[size];
		CrossEntropyError::size = size;
		double log_x;
		double E = 0.0;
		for (int i = 0; i < (int)size; i++) {
			log_x = logs[i].forward(x[i]);
			E = adds[i].forward(
				E,
				muls[i].forward(log_x,t[i])
			);
		}
		E = mul.forward(-1, E);
		return E;
	}

	double* backward(double dout = 1.0) {
		dx = new double[size];
		double de = *(mul.backward(dout) + 1);
		double tmp;
		for (int i = 0; i < (int)size; i++) {
			tmp = *(adds[i].backward(de) + 1);
			tmp = *(muls[i].backward(tmp) + 0);
			dx[i] = logs[i].backward(tmp);
		}
		return dx;
	}

	void del() {
		delete[] logs;
		delete[] muls;
		delete[] adds;
		delete[] dx;
	}
private:
	Log* logs;
	Mul* muls;
	Add* adds;
	Mul mul;
	size_t size;
};


/*-----3層のモデルクラス-----*/
template<class A1, class A2, class LOSS>
class SimpleNet {
public:
#pragma region 変数宣言
	class Forward {
	public:
		double** weight;
		double* bias;
		double* node_out;
		Mul** muls_two;
		Add** adds_two;
		Add* adds_one;
	};

	class Backward {
	public:
		double** dweight;
		double* dbias;
		double* dnode_out;
	};

	size_t input_size;
	size_t hidden_size;
	size_t output_size;
	Forward fc1, fc2;
	Backward dfc1, dfc2;

	double* y;	// 予測値
	double loss;// 損失値
#pragma endregion

	SimpleNet(size_t input_size,size_t hidden_size,size_t output_size) 
	{
		SimpleNet::input_size = input_size;
		SimpleNet::hidden_size = hidden_size;
		SimpleNet::output_size = output_size;

		fc1.weight = new double* [hidden_size];
		fc2.weight = new double* [output_size];
		dfc1.dweight = new double* [hidden_size];
		dfc2.dweight = new double* [output_size];

		fc1.bias = new double[hidden_size];
		fc2.bias = new double[output_size];
		dfc1.dbias = new double[hidden_size];
		dfc2.dbias = new double[output_size];

		fc1.node_out = new double[hidden_size];
		fc2.node_out = new double[output_size];
		dfc2.dnode_out = new double[hidden_size];

		fc1.muls_two = new Mul * [hidden_size];
		fc1.adds_two = new Add * [hidden_size];
		fc1.adds_one = new Add[hidden_size];

		fc2.muls_two = new Mul * [output_size];
		fc2.adds_two = new Add * [output_size];
		fc2.adds_one = new Add[output_size];

		// 中間層の活性化関数
		activations = new A1[hidden_size];
		// 予測値
		y = new double[output_size];
	
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<double> dist(-1, 1);

		int i, j;
		for (i = 0; i < (int)hidden_size; i++) {
			fc1.weight[i] = new double[input_size];
			fc1.muls_two[i] = new Mul[input_size];
			fc1.adds_two[i] = new Add[input_size];
			dfc1.dweight[i] = new double[input_size];
			fc1.bias[i] = 0.0;
			dfc1.dbias[i] = 0.0;
			// 逆伝搬中間層のノードを初期化
			dfc2.dnode_out[i] = 0.0;
			for (j = 0; j < (int)input_size; j++) {
				fc1.weight[i][j] = dist(gen);
				dfc1.dweight[i][j] = 0.0;
			}
		}
		for (i = 0; i < (int)output_size; i++) {
			fc2.weight[i] = new double[hidden_size];
			fc2.muls_two[i] = new Mul[hidden_size];
			fc2.adds_two[i] = new Add[hidden_size];
			dfc2.dweight[i] = new double[hidden_size];
			fc2.bias[i] = 0.0;
			dfc2.dbias[i] = 0.0;
			for (j = 0; j < (int)hidden_size; j++) {
				fc2.weight[i][j] = dist(gen);
				dfc2.dweight[i][j] = 0.0;
			}
		}
	}

	void del() {
		int i;
		for (i = 0; i < (int)hidden_size; i++) {
			delete[] fc1.weight[i];
			delete[] fc1.muls_two[i];
			delete[] fc1.adds_two[i];
			delete[] dfc1.dweight[i];
		}
		for (i = 0; i < (int)output_size; i++) {
			delete[] fc2.weight[i];
			delete[] fc2.muls_two[i];
			delete[] fc2.adds_two[i];
			delete[] dfc2.dweight[i];
		}
		delete[] fc1.weight;
		delete[] fc1.bias;
		delete[] fc1.node_out;
		delete[] fc1.muls_two;
		delete[] fc1.adds_two;
		delete[] fc1.adds_one;
		delete[] dfc1.dweight;
		delete[] dfc1.dbias;
		delete[] dfc1.dnode_out;

		delete[] fc2.weight;
		delete[] fc2.bias;
		delete[] fc2.node_out;
		delete[] fc2.muls_two;
		delete[] fc2.adds_two;
		delete[] fc2.adds_one;
		delete[] dfc2.dweight;
		delete[] dfc2.dbias;
		delete[] dfc2.dnode_out;

		delete[] activations;
		delete[] y;

		activationOut.del();
		lossFunc.del();

		cout << "正常に解放されました！" << endl;
	}

private:
	A1* activations;
	A2 activationOut;
	LOSS lossFunc;
};

/*-----未実装-----*/
class LogSoftmax {};
class Nll_Loss {};
class Momentum {};
class AdaGrad {};	// <- RMSProp
class Adam {};
#endif // !_H_
