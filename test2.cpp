#include<iostream>
#include<float.h>
#include<random>
#include<cmath>

using namespace std;

const double C = 1.0e+100;
const double D = 1.0e+200;

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

		// ソフトマックス関数の分子分母
		for (int i = 0; i < (int)size; i++) {
			exp_a[i] = exps[i].forward(x[i] - Cmax);
			exp_sum = adds[i].forward(exp_sum, exp_a[i]);
		}

		// ソフトマックス関数の分子分母
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
	double forward(double* x, double* t, size_t size) {
		logs = new Log[size];
		muls = new Mul[size];
		adds = new Add[size];
		CrossEntropyError::size = size;
		double log_x;
		double m;
		double E = 0.0;
		for (int i = 0; i < (int)size; i++) {
			log_x = logs[i].forward(x[i]);
			m = muls[i].forward(log_x, t[i]);
			E = adds[i].forward(E, m);
		}
		E = mul.forward(-1, E);
		return E;
	}
	double* dx;
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
class SoftmaxWithLossLayer {
public:
	double forward(double* x, double* t, size_t size) {
		Softmax softmax;
		CrossEntropyError cee;
		SoftmaxWithLossLayer::t = t;
		SoftmaxWithLossLayer::size = size;
		y = new double[size];
		for (int i = 0; i < (int)size; i++) {
			y[i] = *(softmax.forward(x, size) + i);
		}
		double loss = cee.forward(y, t, size);
		softmax.del();
		cee.del();
		return loss;
	}
	double* dx;
	double* backward(double dout = 1.0) {
		dx = new double[size];
		for (int i = 0; i < (int)size; i++) {
			dx[i] = y[i] - t[i];
		}
		return dx;
	}
	void del() {
		delete[] y;
		delete[] dx;
	}
private:
	double* y;
	double* t;
	size_t size;
};
class SimpleNet {
public:
	class DF {
	public:
		double** dweight;
		double* dbias;
		double* dnode_out;
	};
	DF dfc1, dfc2;
	double* y;
	double loss;
	
	SimpleNet(size_t input_size,size_t hidden_size,size_t output_size) {
		SimpleNet::input_size = input_size;
		SimpleNet::hidden_size = hidden_size;
		SimpleNet::output_size = output_size;

		fc1.weight = new double* [hidden_size];
		fc2.weight = new double* [output_size];

		fc1.bias = new double[hidden_size];
		fc2.bias = new double[output_size];

		fc1.node_out = new double[hidden_size];
		fc2.node_out = new double[output_size];

		fc1.muls_two = new Mul * [hidden_size];
		fc1.adds_two = new Add * [hidden_size];
		fc1.adds_one = new Add[hidden_size];

		fc2.muls_two = new Mul * [output_size];
		fc2.adds_two = new Add * [output_size];
		fc2.adds_one = new Add[output_size];
		
		sigmoid = new Sigmoid[hidden_size];

		y = new double[output_size];

		dfc1.dweight = new double* [hidden_size];
		dfc2.dweight = new double* [output_size];

		dfc1.dbias = new double[hidden_size];
		dfc2.dbias = new double[output_size];

		dfc2.dnode_out = new double[hidden_size];

		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<double> dist(-1, 1);

		for (int i = 0; i < (int)hidden_size; i++) {
			fc1.weight[i] = new double[input_size];
			fc1.muls_two[i] = new Mul[input_size];
			fc1.adds_two[i] = new Add[input_size];
			fc1.bias[i] = 0.0;
			dfc2.dnode_out[i] = 0.0;
			dfc1.dweight[i] = new double[input_size];
			dfc1.dbias[i] = 0.0;
			for (int j = 0; j < (int)input_size; j++) {
				fc1.weight[i][j] = dist(gen);
				dfc1.dweight[i][j] = 0.0;
			}
		}

		for (int i = 0; i < (int)output_size; i++) {
			fc2.weight[i] = new double[hidden_size];
			fc2.muls_two[i] = new Mul[hidden_size];
			fc2.adds_two[i] = new Add[hidden_size];
			fc2.bias[i] = 0.0;
			dfc2.dweight[i] = new double[hidden_size];
			dfc2.dbias[i] = 0.0;
			dfc2.dnode_out[i] = 0.0;
			for (int j = 0; j < (int)hidden_size; j++) {
				fc2.weight[i][j] = dist(gen);
				dfc2.dweight[i][j] = 0.0;
			}
		}
		cout << "初期化完了！" << endl;
	}
	~SimpleNet() {
		for (int i = 0; i < (int)hidden_size; i++) {
			delete[] fc1.weight[i];
			delete[] fc1.muls_two[i];
			delete[] fc1.adds_two[i];
			delete[] dfc1.dweight[i];
		}
		for (int i = 0; i < (int)output_size; i++) {
			delete[] fc2.weight[i];
			delete[] fc2.muls_two[i];
			delete[] fc2.adds_two[i];
			delete[] dfc2.dweight[i];
		}
		delete[] fc1.weight;
		delete[] fc1.muls_two;
		delete[] fc1.adds_two;
		delete[] dfc1.dweight;
		delete[] fc1.bias;
		delete[] fc1.node_out;

		delete[] fc2.weight;
		delete[] fc2.muls_two;
		delete[] fc2.adds_two;
		delete[] dfc2.dweight;
		delete[] fc2.bias;
		delete[] fc2.node_out;

		delete[] sigmoid;
		delete[] y;

		delete[] dfc2.dnode_out;
		cee.del();
		softmax.del();
		
		cout << "正常に解放されました！" << endl;
	}

	double* predict(double* x) {
		_fc1(x);
		_fc2();
		y = softmax.forward(fc2.node_out, output_size);
		return y;
	}

	double Loss(double* x, double* t) {
		double* y = predict(x);
		loss = cee.forward(y, t, output_size);
		cout << "順伝播完了！" << endl;
		return loss;
	}

	void backward(double dout = 1.0) {
		// 損失関数の逆伝播
		double* result_arr = cee.backward(dout);
		// ソフトマックス関数の逆伝播
		result_arr = softmax.backward(result_arr);
		// fc2の逆伝播
		_dfc2(result_arr);
		// fc1の逆伝播
		_dfc1();
	}

	double*** params;
	double*** get_dW1_dW2_db1_db2(void) {
		// 数字を格納
		params = new double** [3];
		params[2] = new double* [2];
		params[0] = new double* [hidden_size];
		params[1] = new double* [output_size];
		params[2][0] = new double[hidden_size];
		params[2][1] = new double[output_size];
		for (int i = 0; i < hidden_size; i++) {
			params[0][i] = new double[input_size];
			params[2][0][i] = dfc1.dbias[i];
			for (int j = 0; j < input_size; j++) {
				params[0][i][j] = dfc1.dweight[i][j];
			}
		}
		for (int i = 0; i < output_size; i++) {
			params[1][i] = new double[hidden_size];
			params[2][1][i] = dfc2.dbias[i];
			for (int j = 0; j < hidden_size; j++) {
				params[1][i][j] = dfc2.dweight[i][j];
			}
		}
		return params;
	}

private:
	class V {
	public:
		double** weight;
		double* bias;
		double* node_out;
		Mul** muls_two;
		Add** adds_two;
		Add* adds_one;
	};

	V fc1, fc2;
	size_t input_size;
	size_t hidden_size;
	size_t output_size;
	Sigmoid* sigmoid;
	Softmax softmax;
	CrossEntropyError cee;

	void _fc1(double* x) {
		double tmp, a, s;
		for (int i = 0; i < hidden_size; i++) {
			tmp = 0.0;
			for (int j = 0; j < input_size; j++) {
				a = fc1.muls_two[i][j].forward(
					x[j], fc1.weight[i][j]
				);
				tmp = fc1.adds_two[i][j].forward(tmp, a);
			}
			s = fc1.adds_one[i].forward(
				tmp, 
				fc1.bias[i]
			);
			fc1.node_out[i] = sigmoid[i].forward(s);
		}
	}

	void _fc2(void) {
		double tmp, a, s;
		for (int i = 0; i < output_size; i++) {
			tmp = 0.0;
			for (int j = 0; j < hidden_size; j++) {
				a = fc2.muls_two[i][j].forward(
					fc1.node_out[j], 
					fc2.weight[i][j]
				);
				tmp = fc2.adds_two[i][j].forward(tmp, a);
			}
			fc2.node_out[i] = fc2.adds_one[i].forward(
				tmp, 
				fc2.bias[i]
			);
		}
	}

	void _dfc2(double* dout) {
		double* dtmp_bias2;
		double* dtmp_da;
		double* dhidden_out_dweight2;
		for (int i = 0; i < output_size; i++) {
			dtmp_bias2 = fc2.adds_one[i].backward(dout[i]);
			dfc2.dbias[i] = dtmp_bias2[1];
			for (int j = 0; j < hidden_size; j++) {
				dtmp_da = fc2.adds_two[i][j].backward(dtmp_bias2[0]);
				dhidden_out_dweight2 = fc2.muls_two[i][j].backward(dtmp_da[1]);
				dfc2.dweight[i][j] = dhidden_out_dweight2[1];
				dfc2.dnode_out[j] += dhidden_out_dweight2[0];
			}
		}
	}

	void _dfc1(void) {
		double tmp;
		double* dtmp_bias1;
		double* dtmp_da;
		double* dx_dweight1;
		for (int i = 0; i < hidden_size; i++) {
			tmp = sigmoid[i].backward(dfc2.dnode_out[i]);
			dtmp_bias1 = fc1.adds_one[i].backward(tmp);
			dfc1.dbias[i] = dtmp_bias1[1];
			for (int j = 0; j < input_size; j++) {
				dtmp_da = fc1.adds_two[i][j].backward(dtmp_bias1[0]);
				dx_dweight1 = fc1.muls_two[i][j].backward(dtmp_da[1]);
				dfc1.dweight[i][j] = dx_dweight1[1];
			}
		}
	}
};

#pragma region TEST
void test1() {
	Add add;
	cout << add.forward(1, 2) << endl;
	cout << *(add.backward(1.0) + 0) << endl;
	cout << *(add.backward(1.0) + 1) << endl;
}
void test2() {
	Mul mul;
	cout << mul.forward(2, 3) << endl;
	cout << *((mul.backward(1)) + 0) << endl;
	cout << *((mul.backward(1)) + 1) << endl;
}
void test3() {
	Div div;
	cout << div.forward(0) << endl;
	cout << div.backward(1) << endl;
}
void test4() {
	Exp exp;
	cout << exp.forward(100000) << endl;
	cout << exp.backward(1) << endl;
}
void test5() {
	Log log;
	cout << log.forward(2.7) << endl;
	cout << log.backward(1.0) << endl;
}
void test6() {
	Sigmoid sigmoid;
	cout << sigmoid.forward(12) << endl;
	cout << sigmoid.forward2(12) << endl;

	cout << sigmoid.backward(1.0) << endl;
	cout << sigmoid.backward2(1.0) << endl;

	// 精度に違いなし
}
void test7() {
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
	softmax.forward(arr, 10);
	for (int i = 0; i < 10; i++) {
		printf("dy[%d] = %lf\n", i, *(softmax.backward(darr) + i));
	}
	softmax.del();
}
void test8() {
	CrossEntropyError cee;
	double x[2] = { 0.8,0.2 };
	double t[2] = { 1,0 };
	double E = cee.forward(x, t, 2);
	double* p= cee.backward(1.0);
	cee.del();
	cout << E << endl;
	cout << p[0] << endl;
	cout << p[1] << endl;
}
void test9() {
	SoftmaxWithLossLayer swll;
	double x[2] = { 0.97,-1.56 };
	double t[2] = { 1,0 };
	cout << swll.forward(x, t, 2) << endl;
	cout << *(swll.backward() + 0) << endl;
	cout << *(swll.backward() + 1) << endl;
}
void test10() {
	double x[2] = { 0.63,-0.225 };
	double t[2] = { 1,0 };
	Softmax softmax;
	CrossEntropyError cee;
	double* y = softmax.forward(x, 2);
	double loss = cee.forward(y, t, 2);

	printf("Softmax->CrossEntropy :%lf\n", loss);

	SoftmaxWithLossLayer swll;
	loss = swll.forward(x, t, 2);

	printf("SoftmaxWithLossLayer : %lf\n", loss);

	double* dx;
	dx = cee.backward();
	dx = softmax.backward(dx);

	double* dx2;
	dx2 = swll.backward();

	printf("Softmax->CrossEntropy(backward_0) :%lf\n", dx[0]);
	printf("Softmax->CrossEntropy(backward_1) :%lf\n", dx[1]);
	printf("SoftmaxWithLossLayer(backward_0)  :%lf\n", dx2[0]);
	printf("SoftmaxWithLossLayer(backward_1)  :%lf\n", dx2[1]);

	cee.del();
	softmax.del();
	swll.del();
}
void test11() {
	SimpleNet model(2, 3, 2);
	double x[2] = { 0.63,-0.225 };
	double t[2] = { 1,0 };
	double loss = model.Loss(x, t);
	model.backward();
	double*** params = model.get_dW1_dW2_db1_db2();
	cout << params[0][0][0] << endl;
	cout << params[0][0][1] << endl;
	cout << params[0][1][0] << endl;
	cout << params[0][1][1] << endl;
	cout << params[0][2][0] << endl;
	cout << params[0][2][1] << endl;
	cout << params[1][0][0] << endl;
	cout << params[1][0][1] << endl;
	cout << params[1][0][2] << endl;
	cout << params[1][1][0] << endl;
	cout << params[1][1][1] << endl;
	cout << params[1][1][2] << endl;
	cout << params[2][0][0] << endl;
	cout << params[2][0][1] << endl;
	cout << params[2][0][2] << endl;
	cout << params[2][1][0] << endl;
	cout << params[2][1][1] << endl;
}
#pragma endregion


int main(void) {
	test11();
	return 0;
}