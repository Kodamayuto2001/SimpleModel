#ifndef _SimpleNeuralNetwork_H_
#define _SimpleNeuralNetwork_H_
#include<iostream>
#include<fstream>
#include<float.h>
#include<string>
#include<random>
#include<cmath>
using namespace std;
constexpr double C = 1.0e+100;
constexpr double D = 1.0e+200;
constexpr double Delta = 1.0e-300;

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
		if ((y = 1 / x) == float(INFINITY)) {
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
		if (x > 0.0) { return x; }
		return 0.0;
	}
	double backward(double dout) {
		if (x > 0.0) { return dout; }
		return 0;
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
		int i;
		for (i = 0; i < (int)size; ++i) {
			if (Cmax < x[i]) {
				Cmax = x[i];
			}
		}

		for (i = 0; i < (int)size; ++i) {
			exp_a[i] = exps[i].forward(x[i] - Cmax);
			exp_sum = adds[i].forward(exp_sum, exp_a[i]);
		}

		exp_div = div.forward(exp_sum);

		for (i = 0; i < (int)size; ++i) {
			y[i] = muls[i].forward(exp_a[i], exp_div);
		}
		delete[] exp_a;
		return y;
	}

	double* backward(double* dout) {
		double* dexp_a = new double[size];
		double dexp_sum = 0.0;
		int i;
		for (i = 0; i < (int)size; ++i) {
			dexp_a[i] = *(muls[i].backward(dout[i]) + 0);
			dexp_sum += *(muls[i].backward(dout[i]) + 1);
		}

		double dexp_div = div.backward(dexp_sum);

		double tmp;
		dr = new double[size];
		for (i = 0; i < (int)size; ++i) {
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
		for (int i = 0; i < (int)size; ++i) {
			log_x = logs[i].forward(x[i]);
			E = adds[i].forward(
				E,
				muls[i].forward(log_x, t[i])
			);
		}
		E = mul.forward(-1, E);
		return E;
	}

	double* backward(double dout = 1.0) {
		dx = new double[size];
		double de = *(mul.backward(dout) + 1);
		double tmp;
		for (int i = 0; i < (int)size; ++i) {
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

#pragma endregion

	SimpleNet(size_t input_size, size_t hidden_size, size_t output_size)
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

		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<double> dist(-1, 1);

		int i, j;
		for (i = 0; i < (int)hidden_size; ++i) {
			fc1.weight[i] = new double[input_size];
			fc1.muls_two[i] = new Mul[input_size];
			fc1.adds_two[i] = new Add[input_size];
			dfc1.dweight[i] = new double[input_size];
			fc1.bias[i] = 0.0;
			dfc1.dbias[i] = 0.0;
			// 逆伝搬中間層のノードを初期化
			dfc2.dnode_out[i] = 0.0;
			for (j = 0; j < (int)input_size; ++j) {
				fc1.weight[i][j] = dist(gen);
				dfc1.dweight[i][j] = 0.0;
			}
		}
		for (i = 0; i < (int)output_size; ++i) {
			fc2.weight[i] = new double[hidden_size];
			fc2.muls_two[i] = new Mul[hidden_size];
			fc2.adds_two[i] = new Add[hidden_size];
			dfc2.dweight[i] = new double[hidden_size];
			fc2.bias[i] = 0.0;
			dfc2.dbias[i] = 0.0;
			for (j = 0; j < (int)hidden_size; ++j) {
				fc2.weight[i][j] = dist(gen);
				dfc2.dweight[i][j] = 0.0;
			}
		}
		cout << "初期化完了" << endl;
	}

	double* predict(double* x) {
		_fc1(x);
		_fc2();
		return activationOut.forward(fc2.node_out, output_size);
	}

	double forward(double* x, double* t) {
		double* y = predict(x);
		return lossFunc.forward(y, t, output_size);
	}

	void backward(double dout = 1.0) {
		double* result_arr = lossFunc.backward(dout);
		result_arr = activationOut.backward(result_arr);
		_dfc2(result_arr);
		_dfc1();
	}

	void save(const char* fileName = "model.txt") {
		ofstream ofs(fileName);

		if (!ofs) {
			cout << "ファイルが開けませんでした。" << endl;
			cin.get();
		}

		for (int i = 0; i < hidden_size; ++i) {
			for (int j = 0; j < input_size; ++j) {

				ofs << "W1_" << i << j << fc1.weight[i][j] << "_";
			}
		}

		for (int i = 0; i < output_size; ++i) {
			for (int j = 0; j < hidden_size; ++j) {
				ofs << "W2_" << i << j << fc2.weight[i][j] << "_";
			}
		}

		for (int i = 0; i < hidden_size; ++i) {
			ofs << "b1_" << i << fc1.bias[i] << "_";
		}

		for (int i = 0; i < output_size; ++i) {
			ofs << "b2_" << i << fc2.bias[i] << "_";
		}
	}

	void load(const char* fileName = "model.txt") {
		string data;
		{
			ifstream ifs(fileName);
			if (!ifs) {
				cout << "ファイルが開けませんでした。" << endl;
				cin.get();
			}
			string buf;
			while (!ifs.eof())
			{
				getline(ifs, buf);
				data += buf + "\n";
			}
		}

		int a = 0;
		int cnt = 0;
		string tmp = "W1_W2_b1_b2_";
		while (1)
		{
			if (data[a + 2] == NULL) { break; }
			tmp = data[a + 0];
			tmp += data[a + 1];
			tmp += data[a + 2];

			if (tmp == "W1_") {
				cnt = a + 5;
				tmp = data[cnt];
				while (true)
				{
					cnt++;
					if (data[cnt] == '_') { break; }
					tmp += data[cnt];
				}
				fc1.weight[data[a + 3] - 48][data[a + 4] - 48] = stod(tmp);
			}
			if (tmp == "W2_") {
				cnt = a + 5;
				tmp = data[cnt];
				while (true)
				{
					cnt++;
					if (data[cnt] == '_') { break; }
					tmp += data[cnt];
				}
				fc2.weight[data[a + 3] - 48][data[a + 4] - 48] = stod(tmp);
			}
			if (tmp == "b1_") {
				cnt = a + 4;
				tmp = data[cnt];
				while (true)
				{
					cnt++;
					if (data[cnt] == '_') { break; }
					tmp += data[cnt];
				}
				fc1.bias[data[a + 3] - 48] = stod(tmp);
			}
			if (tmp == "b2_") {
				cnt = a + 4;
				tmp = data[cnt];
				while (true)
				{
					cnt++;
					if (data[cnt] == '_') { break; }
					tmp += data[cnt];
				}
				fc2.bias[data[a + 3] - 48] = stod(tmp);
			}
			a++;
		}
	}

	void del() {
		int i;
		for (i = 0; i < (int)hidden_size; ++i) {
			delete[] fc1.weight[i];
			delete[] fc1.muls_two[i];
			delete[] fc1.adds_two[i];
			delete[] dfc1.dweight[i];
		}
		for (i = 0; i < (int)output_size; ++i) {
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

		activationOut.del();
		lossFunc.del();

		cout << "正常に解放しました！" << endl;
	}

private:
	A1* activations;
	A2 activationOut;
	LOSS lossFunc;

	void _fc1(double* x) {
		double tmp, a, s;
		int i, j;
		for (i = 0; i < hidden_size; ++i) {
			tmp = 0.0;
			for (j = 0; j < input_size; ++j) {
				a = fc1.muls_two[i][j].forward(x[j], fc1.weight[i][j]);
				tmp = fc1.adds_two[i][j].forward(tmp, a);
			}
			s = fc1.adds_one[i].forward(tmp, fc1.bias[i]);
			fc1.node_out[i] = activations[i].forward(s);
			dfc2.dnode_out[i] = 0.0;
		}
	}

	void _fc2(void) {
		double tmp, a;
		int i, j;
		for (i = 0; i < output_size; ++i) {
			tmp = 0.0;
			for (j = 0; j < hidden_size; ++j) {
				a = fc2.muls_two[i][j].forward(
					fc1.node_out[j],
					fc2.weight[i][j]
				);
				tmp = fc2.adds_two[i][j].forward(tmp, a);
			}
			fc2.node_out[i] = fc2.adds_one[i].forward(tmp, fc2.bias[i]);
		}
	}

	void _dfc2(double* dout) {
		double* dtmp_bias2;
		double* dtmp_da;
		double* dhidden_out_dweight2;
		int i, j;
		for (i = 0; i < output_size; ++i) {
			dtmp_bias2 = fc2.adds_one[i].backward(dout[i]);
			dfc2.dbias[i] = dtmp_bias2[1];
			for (j = 0; j < hidden_size; ++j) {
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
		int i, j;
		for (i = 0; i < hidden_size; ++i) {
			tmp = activations[i].backward(dfc2.dnode_out[i]);
			dtmp_bias1 = fc1.adds_one[i].backward(tmp);
			dfc1.dbias[i] = dtmp_bias1[1];
			for (j = 0; j < input_size; ++j) {
				dtmp_da = fc1.adds_two[i][j].backward(dtmp_bias1[0]);
				dx_dweight1 = fc1.muls_two[i][j].backward(dtmp_da[1]);
				dfc1.dweight[i][j] = dx_dweight1[1];
			}
		}
	}
};

class FastMul {
public:
	void forward(double* a, double* b, double* y) {
		tmp[0] = a;
		tmp[1] = b;
		if (((*y) = (*a) * (*b)) == float(INFINITY)) {
			(*y) = DBL_MAX / C;
		}
	}
	void backward(double* dout, double dx[2]) {
		dx[0] = (*dout) * (*tmp[1]);
		dx[1] = (*dout) * (*tmp[0]);
	}
private:
	double* tmp[2];	//	アドレスを格納する配列
};

class FastDiv {
public:
	void forward(double* x, double* y) {
		if (((*y) = 1 / (*x)) == float(INFINITY)) {
			(*y) = DBL_MAX / D;
		}
		FastDiv::y = y;
	}
	void backward(double* dout, double* dx) {
		(*dx) = (*dout) * (-1 * (*y) * (*y));
	}
private:
	double* y;
};



//template<class ActFunc, class SoftmaxWithLoss>
//class FastSimpleNet {
//public:
//	class Forward {
//	public:
//		double** weight;
//		double* bias;
//		double* node_out;
//		FastMul** muls_two;
//	};
//	class Backward {
//	public:
//		double** dweight;
//		double* dbias;
//		double* dnode_out;
//	};
//
//	size_t input_size;
//	size_t hidden_size;
//	size_t output_size;
//	Forward fc1, fc2;
//	Backward dfc1, dfc2;
//	double* y;
//	double loss;
//
//	FastSimpleNet(
//		size_t&& _inputSize,
//		size_t&& _hiddenSize,
//		size_t&& _outputSize) {
//
//		i = j = 0;
//		a = b = 0.0;
//
//		input_size = _inputSize;
//		hidden_size = _hiddenSize;
//		output_size = _outputSize;
//
//		fc1.weight = new double* [hidden_size];
//		fc1.bias = new double[hidden_size];
//		fc1.node_out = new double[hidden_size];
//		fc1.muls_two = new FastMul * [hidden_size];
//
//		fc2.weight = new double* [output_size];
//		fc2.bias = new double[output_size];
//		fc2.node_out = new double[output_size];
//		fc2.muls_two = new FastMul * [output_size];
//
//		aFunc = new ActFunc[hidden_size];
//
//		dfc2.dweight = new double* [output_size];
//		dfc2.dbias = new double[output_size];
//		dfc2.dnode_out = new double[output_size];
//
//		dfc1.dweight = new double* [hidden_size];
//		dfc1.dbias = new double[hidden_size];
//		dfc1.dnode_out = new double[hidden_size];
//
//		swl.__init__(&output_size);
//		y = new double[output_size];
//
//		random_device rd;
//		mt19937 gen(rd());
//		uniform_real_distribution<double> dist(-1, 1);
//
//		for (i = 0; i < (int)hidden_size; ++i) {
//			fc1.muls_two[i] = new FastMul[input_size];
//			fc1.weight[i] = new double[input_size];
//			fc1.bias[i] = 0.0;
//			fc1.node_out[i] = 0.0;
//			dfc1.dweight[i] = new double[input_size];
//			dfc1.dbias[i] = 0.0;
//			dfc1.dnode_out[i] = 0.0;
//			for (j = 0; j < (int)input_size; ++j) {
//				fc1.weight[i][j] = dist(gen);
//				dfc1.dweight[i][j] = 0.0;
//			}
//		}
//
//		for (i = 0; i < (int)output_size; ++i) {
//			fc2.muls_two[i] = new FastMul[hidden_size];
//			fc2.weight[i] = new double[hidden_size];
//			fc2.bias[i] = 0.0;
//			fc2.node_out[i] = 0.0;
//			dfc2.dweight[i] = new double[hidden_size];
//			dfc2.dbias[i] = 0.0;
//			dfc2.dnode_out[i] = 0.0;
//			y[i] = 0.0;
//			for (j = 0; j < (int)hidden_size; ++j) {
//				fc2.weight[i][j] = dist(gen);
//				dfc2.dweight[i][j] = 0.0;
//			}
//		}
//	}
//
//	void predict(double* x) {
//		_fc1(x);
//		_fc2();
//		swl.SoftmaxForward(fc2.node_out, y);
//	}
//
//	void forward(double* x, double* t) {
//		predict(x);
//		swl.CrossEntropyErrorForward(t, &loss);
//	}
//
//	void backward() {
//		swl.backward(dfc2.dnode_out);
//		_dfc2(dfc2.dnode_out);
//		_dfc1();
//	}
//
//	void save(const char* fileName = "model.kodamayuto") {
//		ofstream ofs;
//		ofs.open(fileName, ios::out | ios::binary | ios::trunc);
//		if (!ofs) {
//			cout << "ファイルが開けませんでした。" << endl;
//		}
//		for (i = 0; i < hidden_size; ++i) {
//			ofs.write((char*)&fc1.bias[i], sizeof(double));
//			for (j = 0; j < input_size; ++j) {
//				ofs.write((char*)&fc1.weight[i][j], sizeof(double));
//			}
//		}
//		for (i = 0; i < output_size; ++i) {
//			ofs.write((char*)&fc2.bias[i], sizeof(double));
//			for (j = 0; j < hidden_size; ++j) {
//				ofs.write((char*)&fc2.weight[i][j], sizeof(double));
//			}
//		}
//		ofs.close();
//	}
//
//	void load(const char* fileName = "model.kodamayuto") {
//		ifstream ifs(fileName, ios::in | ios::binary);
//		if (!ifs) {
//			cout << "ファイルが開けませんでした。" << endl;
//		}
//		for (i = 0; i < hidden_size; ++i) {
//			ifs.read((char*)&fc1.bias[i], sizeof(double));
//			for (j = 0; j < input_size; ++j) {
//				ifs.read((char*)&fc1.weight[i][j], sizeof(double));
//			}
//		}
//		for (i = 0; i < output_size; ++i) {
//			ifs.read((char*)&fc2.bias[i], sizeof(double));
//			for (j = 0; j < hidden_size; ++j) {
//				ifs.read((char*)&fc2.weight[i][j], sizeof(double));
//			}
//		}
//		ifs.close();
//	}
//
//	void del() {
//		for (i = 0; i < (int)hidden_size; ++i) {
//			delete[] fc1.weight[i];
//			delete[] fc1.muls_two[i];
//			delete[] dfc1.dweight[i];
//		}
//		for (i = 0; i < (int)output_size; ++i) {
//			delete[] fc2.weight[i];
//			delete[] fc2.muls_two[i];
//			delete[] dfc2.dweight[i];
//		}
//		delete[] fc1.weight;
//		delete[] fc1.muls_two;
//		delete[] fc1.node_out;
//		delete[] fc1.bias;
//
//		delete[] fc2.weight;
//		delete[] fc2.muls_two;
//		delete[] fc2.bias;
//		delete[] fc2.node_out;
//
//		delete[] dfc2.dweight;
//		delete[] dfc2.dbias;
//		//	dfc2.dbiasにdfc2.dnode_outのアドレスを格納したので、dfc2.dnode_outをdelete[]できない
//		//	delete[] dfc2.dnode_out;
//
//		delete[] dfc1.dweight;
//		delete[] dfc1.dbias;
//		delete[] dfc1.dnode_out;
//
//		delete[] y;
//		swl.del();
//
//		cout << "正常に解放しました（FastSimpleNet）" << endl;
//	}
//private:
//	ActFunc* aFunc;
//	SoftmaxWithLoss swl;
//	int i, j;
//	double a, b;
//	double dx_dweight[2];
//
//	void _fc1(double* x) {
//		for (i = 0; i < hidden_size; ++i) {
//			b = 0.0;
//			for (j = 0; j < input_size; ++j) {
//				fc1.muls_two[i][j].forward(&(x[j]), &(fc1.weight[i][j]), &a);
//				b += a;
//			}
//			a = b + fc1.bias[i];
//			aFunc[i].forward(&a, &fc1.node_out[i]);
//			dfc1.dnode_out[i] = 0.0;
//		}
//	}
//
//	void _fc2(void) {
//		for (i = 0; i < output_size; ++i) {
//			b = 0.0;
//			for (j = 0; j < hidden_size; ++j) {
//				fc2.muls_two[i][j].forward(
//					&fc1.node_out[j], &fc2.weight[i][j], &a);
//				b += a;
//			}
//			fc2.node_out[i] = b + fc2.bias[i];
//			dfc2.dnode_out[i] = 0.0;
//		}
//	}
//
//	void _dfc2(double* dout) {
//		dfc2.dbias = dout;
//		for (i = 0; i < output_size; ++i) {
//			for (j = 0; j < hidden_size; ++j) {
//				fc2.muls_two[i][j].backward(&dfc2.dbias[i], dx_dweight);
//				dfc1.dnode_out[j] += dx_dweight[0];
//				dfc2.dweight[i][j] = dx_dweight[1];
//			}
//		}
//	}
//
//	void _dfc1(void) {
//		for (i = 0; i < hidden_size; ++i) {
//			aFunc[i].backward(&dfc1.dnode_out[i], &a);
//			dfc1.dbias[i] = a;
//			for (j = 0; j < input_size; ++j) {
//				fc1.muls_two[i][j].backward(&a, dx_dweight);
//				dfc1.dweight[i][j] = dx_dweight[1];
//			}
//		}
//	}
//};

template<class Net>
class SGD {
public:
	SGD(double lr = 0.01) {
		SGD::lr = lr;
	}

	void step(Net model) {
		int i, j;
		for (i = 0; i < model.hidden_size; ++i) {
			// バイアス1の更新
			model.fc1.bias[i] -= lr * model.dfc1.dbias[i];
			for (j = 0; j < model.input_size; ++j) {
				// 重み1の更新
				model.fc1.weight[i][j] -= lr * model.dfc1.dweight[i][j];
			}
		}
		for (i = 0; i < model.output_size; ++i) {
			// バイアス2の更新
			model.fc2.bias[i] -= lr * model.dfc2.dbias[i];
			for (j = 0; j < model.hidden_size; ++j) {
				// 重み2の更新
				model.fc2.weight[i][j] -= lr * model.dfc2.dweight[i][j];
			}
		}
	}
private:
	double lr;
};

template <class Net>
class Momentum {
public:
	Momentum(double lr = 0.01, double momentum = 0.9) {
		Momentum::lr = lr;
		Momentum::momentum = momentum;
	}

	void step(Net model) {
		input_size = model.input_size;
		hidden_size = model.hidden_size;
		output_size = model.output_size;
		int i, j;
		if (v_w1 == NULL || v_w2 == NULL || v_b1 == NULL || v_b2 == NULL) {
			v_w1 = new double* [hidden_size];
			v_w2 = new double* [output_size];
			v_b1 = new double[hidden_size];
			v_b2 = new double[output_size];
			for (i = 0; i < hidden_size; ++i) {
				v_w1[i] = new double[input_size];
				v_b1[i] = 0.0;
				for (j = 0; j < input_size; ++j) {
					v_w1[i][j] = 0.0;
				}
			}
			for (i = 0; i < output_size; ++i) {
				v_w2[i] = new double[hidden_size];
				v_b2[i] = 0.0;
				for (j = 0; j < hidden_size; ++j) {
					v_w2[i][j] = 0.0;
				}
			}
		}
		for (i = 0; i < hidden_size; ++i) {
			v_b1[i] = momentum * v_b1[i] - lr * model.dfc1.dbias[i];
			model.fc1.bias[i] += v_b1[i];
			for (j = 0; j < input_size; ++j) {
				v_w1[i][j] =
					momentum * v_w1[i][j]
					- lr * model.dfc1.dweight[i][j];
				model.fc1.weight[i][j] += v_w1[i][j];
			}
		}
		for (i = 0; i < output_size; ++i) {
			v_b2[i] = momentum * v_b2[i] - lr * model.dfc2.dbias[i];
			model.fc1.bias[i] += v_b2[i];
			for (j = 0; j < hidden_size; ++j) {
				v_w2[i][j] =
					momentum * v_w2[i][j]
					- lr * model.dfc2.dweight[i][j];
				model.fc2.weight[i][j] += v_w2[i][j];
			}
		}
	}

	~Momentum() {
		int i;
		for (i = 0; i < hidden_size; ++i) {
			delete[] v_w1[i];
		}
		for (i = 0; i < output_size; ++i) {
			delete[] v_w2[i];
		}
		delete[] v_b1;
		delete[] v_b2;
		delete[] v_w1;
		delete[] v_w2;
		cout << "正常に解放しました（Momentum）" << endl;
	}
private:
	double lr;
	double momentum;
	double** v_w1;
	double** v_w2;
	double* v_b1;
	double* v_b2;
	size_t input_size;
	size_t hidden_size;
	size_t output_size;
};

template <class Net>
class AdaGrad {
public:
	AdaGrad(double lr = 0.01) {
		AdaGrad::lr = lr;
	}

	void step(Net model) {
		input_size = model.input_size;
		hidden_size = model.hidden_size;
		output_size = model.output_size;
		int i, j;
		if (h_w1 == NULL || h_w2 == NULL || h_b1 == NULL || h_b2 == NULL) {
			h_w1 = new double* [hidden_size];
			h_w2 = new double* [output_size];
			h_b1 = new double[hidden_size];
			h_b2 = new double[output_size];
			for (i = 0; i < hidden_size; ++i) {
				h_w1[i] = new double[input_size];
				h_b1[i] = 0.0;
				for (j = 0; j < input_size; ++j) {
					h_w1[i][j] = 0.0;
				}
			}
			for (i = 0; i < output_size; ++i) {
				h_w2[i] = new double[hidden_size];
				h_b2[i] = 0.0;
				for (j = 0; j < hidden_size; ++j) {
					h_w2[i][j] = 0.0;
				}
			}
		}
		for (i = 0; i < hidden_size; ++i) {
			h_b1[i] += model.dfc1.dbias[i] * model.dfc1.dbias[i];
			// 0で除算することを防ぐためにDeltaを足している
			model.fc1.bias[i] -=
				lr / (sqrt(h_b1[i]) + Delta)
				* model.dfc1.dbias[i];
			for (j = 0; j < input_size; ++j) {
				h_w1[i][j] += model.dfc1.dweight[i][j] * model.dfc1.dweight[i][j];
				model.fc1.weight[i][j] -=
					lr / (sqrt(h_w1[i][j]) + Delta)
					* model.dfc1.dweight[i][j];
			}
		}
		for (i = 0; i < output_size; ++i) {
			h_b2[i] += model.dfc2.dbias[i] * model.dfc2.dbias[i];
			model.fc2.bias[i] -=
				lr / (sqrt(h_b2[i]) + Delta)
				* model.dfc2.dbias[i];
			for (j = 0; j < hidden_size; ++j) {
				h_w2[i][j] += model.dfc2.dweight[i][j] * model.dfc2.dweight[i][j];
				model.fc2.weight[i][j] -=
					lr / (sqrt(h_w2[i][j]) + Delta)
					* model.dfc2.dweight[i][j];
			}
		}
	}
	~AdaGrad() {
		int i;
		for (i = 0; i < hidden_size; ++i) {
			delete[] h_w1[i];
		}
		for (i = 0; i < output_size; ++i) {
			delete[] h_w2[i];
		}
		delete[] h_b1;
		delete[] h_b2;
		delete[] h_w1;
		delete[] h_w2;
		cout << "正常に解放しました（AdaGrad）" << endl;
	}
private:
	double lr;
	double** h_w1;
	double** h_w2;
	double* h_b1;
	double* h_b2;
	size_t input_size;
	size_t hidden_size;
	size_t output_size;
};

template<class Net>
class RMSProp {
public:
	RMSProp(double lr = 0.01, double decay_rate = 0.99) {
		RMSProp::lr = lr;
		RMSProp::decay_rate = decay_rate;
	}
	void step(Net model) {
		input_size = model.input_size;
		hidden_size = model.hidden_size;
		output_size = model.output_size;
		int i, j;
		if (h_w1 == NULL || h_w2 == NULL || h_b1 == NULL || h_b2 == NULL) {
			h_w1 = new double* [hidden_size];
			h_w2 = new double* [output_size];
			h_b1 = new double[hidden_size];
			h_b2 = new double[output_size];
			for (i = 0; i < hidden_size; ++i) {
				h_w1[i] = new double[input_size];
				h_b1[i] = 0.0;
				for (j = 0; j < input_size; ++j) {
					h_w1[i][j] = 0.0;
				}
			}
			for (i = 0; i < output_size; ++i) {
				h_w2[i] = new double[hidden_size];
				h_b2[i] = 0.0;
				for (j = 0; j < hidden_size; ++j) {
					h_w2[i][j] = 0.0;
				}
			}
		}
		for (i = 0; i < hidden_size; ++i) {
			h_b1[i] *= decay_rate;
			h_b1[i] += (1 - decay_rate)
				* model.dfc1.dbias[i] * model.dfc1.dbias[i];
			model.fc1.bias[i] -= lr * model.dfc1.dbias[i]
				/ (sqrt(h_b1[i]) + Delta);
			for (j = 0; j < input_size; ++j) {
				h_w1[i][j] *= decay_rate;
				h_w1[i][j] += (1 - decay_rate)
					* model.dfc1.dweight[i][j] * model.dfc1.dweight[i][j];
				model.fc1.weight[i][j] -= lr * model.dfc1.dweight[i][j]
					/ (sqrt(h_w1[i][j]) + Delta);
			}
		}
		for (i = 0; i < output_size; ++i) {
			h_b2[i] *= decay_rate;
			h_b2[i] += (1 - decay_rate)
				* model.dfc2.dbias[i] * model.dfc2.dbias[i];
			model.fc2.bias[i] -= lr * model.dfc2.dbias[i]
				/ (sqrt(h_b2[i]) + Delta);
			for (j = 0; j < hidden_size; ++j) {
				h_w2[i][j] *= decay_rate;
				h_w2[i][j] += (1 - decay_rate)
					* model.dfc2.dweight[i][j] * model.dfc2.dweight[i][j];
				model.fc2.weight[i][j] -= lr * model.dfc2.dweight[i][j]
					/ (sqrt(h_w2[i][j]) + Delta);
			}
		}
	}
	~RMSProp() {
		int i;
		for (i = 0; i < hidden_size; ++i) {
			delete[] h_w1[i];
		}
		for (i = 0; i < output_size; ++i) {
			delete[] h_w2[i];
		}
		delete[] h_w1;
		delete[] h_w2;
		delete[] h_b1;
		delete[] h_b2;
		cout << "正常に解放しました（RMSProp）" << endl;
	}
private:
	double lr;
	double decay_rate;
	double** h_w1;
	double** h_w2;
	double* h_b1;
	double* h_b2;
	size_t input_size;
	size_t hidden_size;
	size_t output_size;
};

template<class Net>
class Adam {
public:
	Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999) {
		Adam::lr = lr;
		Adam::beta1 = beta1;
		Adam::beta2 = beta2;
		lr_t = 0.0;
		i = j = 0;
		isFirst = 0;
	}

	void step(Net model) {
		if (isFirst == 0) {
			isFirst = 1;
			input_size = model.input_size;
			hidden_size = model.hidden_size;
			output_size = model.output_size;

			m_w1 = new double* [hidden_size];
			m_w2 = new double* [output_size];
			m_b1 = new double[hidden_size];
			m_b2 = new double[output_size];
			v_w1 = new double* [hidden_size];
			v_w2 = new double* [output_size];
			v_b1 = new double[hidden_size];
			v_b2 = new double[output_size];
			for (i = 0; i < hidden_size; ++i) {
				m_w1[i] = new double[input_size];
				m_b1[i] = 0.0;
				v_w1[i] = new double[input_size];
				v_b1[i] = 0.0;
				for (j = 0; j < input_size; ++j) {
					m_w1[i][j] = 0.0;
					v_w1[i][j] = 0.0;
				}
			}
			for (i = 0; i < output_size; ++i) {
				m_w2[i] = new double[hidden_size];
				m_b2[i] = 0.0;
				v_w2[i] = new double[hidden_size];
				v_b2[i] = 0.0;
				for (j = 0; j < hidden_size; ++j) {
					m_w2[i][j] = 0.0;
					v_w2[i][j] = 0.0;
				}
			}
		}
		iter += 1.0;
		lr_t = lr * sqrt(1.0 - pow(beta2, iter)) / (1.0 - pow(beta1, iter));
		for (i = 0; i < hidden_size; ++i) {
			m_b1[i] += (1 - beta1) * (model.dfc1.dbias[i] - m_b1[i]);
			v_b1[i] += (1 - beta2) * (model.dfc1.dbias[i] * model.dfc1.dbias[i] - v_b1[i]);
			model.fc1.bias[i] -= lr_t * m_b1[i] / (sqrt(v_b1[i]) + Delta);
			for (j = 0; j < input_size; ++j) {
				m_w1[i][j] += (1 - beta1) * (model.dfc1.dweight[i][j] - m_w1[i][j]);
				v_w1[i][j] += (1 - beta2) * (model.dfc1.dweight[i][j] * model.dfc1.dweight[i][j] - v_w1[i][j]);
				model.fc1.weight[i][j] -= lr_t * m_w1[i][j] / (sqrt(v_w1[i][j]) + Delta);
			}
		}
		for (i = 0; i < output_size; ++i) {
			m_b2[i] += (1 - beta1) * (model.dfc2.dbias[i] - m_b2[i]);
			v_b2[i] += (1 - beta2) * (model.dfc2.dbias[i] * model.dfc2.dbias[i] - v_b2[i]);
			model.fc2.bias[i] -= lr_t * m_b2[i] / (sqrt(v_b2[i]) + Delta);
			for (j = 0; j < hidden_size; ++j) {
				m_w2[i][j] += (1 - beta1) * (model.dfc2.dweight[i][j] - m_w2[i][j]);
				v_w2[i][j] += (1 - beta2) * (model.dfc2.dweight[i][j] * model.dfc2.dweight[i][j] - v_w2[i][j]);
				model.fc2.weight[i][j] -= lr_t * m_w2[i][j] / (sqrt(v_w2[i][j]) + Delta);
			}
		}
	}

	~Adam() {
		int i;
		for (i = 0; i < hidden_size; ++i) {
			delete[] m_w1[i];
			delete[] v_w1[i];
		}
		for (i = 0; i < output_size; ++i) {
			delete[] m_w2[i];
			delete[] v_w2[i];
		}
		delete[] m_w1;
		delete[] v_w1;
		delete[] m_w2;
		delete[] v_w2;
		delete[] m_b1;
		delete[] m_b2;
		delete[] v_b1;
		delete[] v_b2;
		cout << "正常に解放しました（Adam）" << endl;
	}
private:
	int i, j, isFirst;
	double lr;
	double beta1;
	double beta2;
	double** m_w1;
	double** m_w2;
	double* m_b1;
	double* m_b2;
	double** v_w1;
	double** v_w2;
	double* v_b1;
	double* v_b2;
	size_t input_size;
	size_t hidden_size;
	size_t output_size;
	double iter = 0.0;
	double lr_t;
};

//template<class Net>
//class FastAdam {
//public:
//	FastAdam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999) {
//		FastAdam::lr = lr;
//		FastAdam::beta1 = beta1;
//		FastAdam::beta2 = beta2;
//		lr_t = 0.0;
//		i = j = 0;
//		isFirst = 0;
//	}
//
//	void step(Net* model) {
//		if (isFirst == 0) {
//			isFirst = 1;
//			input_size = model->input_size;
//			hidden_size = model->hidden_size;
//			output_size = model->output_size;
//			m_w1 = new double* [hidden_size];
//			m_w2 = new double* [output_size];
//			m_b1 = new double[hidden_size];
//			m_b2 = new double[output_size];
//			v_w1 = new double* [hidden_size];
//			v_w2 = new double* [output_size];
//			v_b1 = new double[hidden_size];
//			v_b2 = new double[output_size];
//			for (i = 0; i < hidden_size; ++i) {
//				m_w1[i] = new double[input_size];
//				m_b1[i] = 0.0;
//				v_w1[i] = new double[input_size];
//				v_b1[i] = 0.0;
//				for (j = 0; j < input_size; ++j) {
//					m_w1[i][j] = 0.0;
//					v_w1[i][j] = 0.0;
//				}
//			}
//			for (i = 0; i < output_size; ++i) {
//				m_w2[i] = new double[hidden_size];
//				m_b2[i] = 0.0;
//				v_w2[i] = new double[hidden_size];
//				v_b2[i] = 0.0;
//				for (j = 0; j < hidden_size; ++j) {
//					m_w2[i][j] = 0.0;
//					v_w2[i][j] = 0.0;
//				}
//			}
//		}
//		iter += 1.0;
//		lr_t = lr * sqrt(1.0 - pow(beta2, iter)) / (1.0 - pow(beta1, iter));
//		for (i = 0; i < hidden_size; ++i) {
//			m_b1[i] += (1 - beta1) * (model->dfc1.dbias[i] - m_b1[i]);
//			v_b1[i] += (1 - beta2) * (model->dfc1.dbias[i] * model->dfc1.dbias[i] - v_b1[i]);
//			model->fc1.bias[i] -= lr_t * m_b1[i] / (sqrt(v_b1[i]) + Delta);
//			for (j = 0; j < input_size; ++j) {
//				m_w1[i][j] += (1 - beta1) * (model->dfc1.dweight[i][j] - m_w1[i][j]);
//				v_w1[i][j] += (1 - beta2) * (model->dfc1.dweight[i][j] * model->dfc1.dweight[i][j] - v_w1[i][j]);
//				model->fc1.weight[i][j] -= lr_t * m_w1[i][j] / (sqrt(v_w1[i][j]) + Delta);
//			}
//		}
//		for (i = 0; i < output_size; ++i) {
//			m_b2[i] += (1 - beta1) * (model->dfc2.dbias[i] - m_b2[i]);
//			v_b2[i] += (1 - beta2) * (model->dfc2.dbias[i] * model->dfc2.dbias[i] - v_b2[i]);
//			model->fc2.bias[i] -= lr_t * m_b2[i] / (sqrt(v_b2[i]) + Delta);
//			for (j = 0; j < hidden_size; ++j) {
//				m_w2[i][j] += (1 - beta1) * (model->dfc2.dweight[i][j] - m_w2[i][j]);
//				v_w2[i][j] += (1 - beta2) * (model->dfc2.dweight[i][j] * model->dfc2.dweight[i][j] - v_w2[i][j]);
//				model->fc2.weight[i][j] -= lr_t * m_w2[i][j] / (sqrt(v_w2[i][j]) + Delta);
//			}
//		}
//	}
//	~FastAdam() {
//		for (i = 0; i < hidden_size; ++i) {
//			delete[] m_w1[i];
//			delete[] v_w1[i];
//		}
//		for (i = 0; i < output_size; ++i) {
//			delete[] m_w2[i];
//			delete[] v_w2[i];
//		}
//		delete[] m_w1;
//		delete[] v_w1;
//		delete[] m_w2;
//		delete[] v_w2;
//		delete[] m_b1;
//		delete[] m_b2;
//		delete[] v_b1;
//		delete[] v_b2;
//		cout << "正常に解放しました（Adam）" << endl;
//	}
//private:
//	int i, j, isFirst;
//	double lr;
//	double beta1;
//	double beta2;
//	double** m_w1;
//	double** m_w2;
//	double* m_b1;
//	double* m_b2;
//	double** v_w1;
//	double** v_w2;
//	double* v_b1;
//	double* v_b2;
//	size_t input_size;
//	size_t hidden_size;
//	size_t output_size;
//	double iter = 0.0;
//	double lr_t;
//};

#endif // !_SimpleNeuralNetwork_H_