#include <iostream>
#include <float.h>
#include <random>
#include <cmath>

using namespace std;

class Add {
public:
	Add() {}
	~Add() {}
	double forward(double a, double b) {
		return a + b;
	}
	double* backward(double dout) {
		static double result[2];
		result[0] = dout * 1.0;
		result[1] = dout * 1.0;
		return result;
	}
};

class Mul {
public:
	Mul() {
		x = NULL;
		y = NULL;
	}
	~Mul(){}
	double forward(double a, double b) {
		x = a;
		y = b;
		return a * b;
	}
	double* backward(double dout) {
		static double result[2];
		result[0] = dout * y;
		result[1] = dout * x;
		return result;
	}
private:
	double x;
	double y;
};

class Div {
public:
	Div() {
		y = NULL;
	}
	~Div() {}
	double forward(double x) {
		// ゼロ除算回避
		if (x > DBL_MIN/C) {
			y = 1 / x;
		}
		else if (x < -DBL_MIN/C) {
			y = 1 / x;
		}
		else {
			isInf = true;
			y = DBL_MAX/C;
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
	double y;
	bool isInf=false;
	const double C=1.0e+100;
};

class Exp {
public:
	Exp() {
		out = NULL;
	}
	~Exp(){}
	double forward(double a) {
		out = exp(a);
		return out;
	}
	double backward(double dout) {
		return dout * out;
	}
private:
	double out;
};

class Log {
public:
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
	double x;
	bool __isInf = false;
	const double C = 1.0e+100;
};

class Sigmoid {
public:
	Sigmoid()	{}
	~Sigmoid()	{}
	double forward(double x) {
		if (x <= -SIGMOID_RANGE) {
			return DBL_MIN;
		}
		if (x >= SIGMOID_RANGE) {
			return 1.0 - DBL_MIN;
		}

		static double a = mul.forward(x, -1);
		static double b = exp.forward(a);
		static double c = add.forward(b, 1);
		static double d = div.forward(c);

		return d;
	}
	double backward(double dout) {
		static double ddx = div.backward(dout);
		static double* dc = add.backward(ddx);
		static double dbx = exp.backward(dc[0]);
		static double* da = mul.backward(dbx);

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
	Softmax() {}
	~Softmax(){
		// 動的にメモリを作成したので配列の先頭ポインタを削除
		delete[] exps;
		delete[] adds;
		delete[] muls;
	}
	template <class TYPE,size_t SIZE> 
	TYPE* forward(const TYPE (&x)[SIZE]) {
		// 配列の動的メモリ割り当て
		exps = new Exp[SIZE];
		adds = new Add[SIZE];
		muls = new Mul[SIZE];
		static TYPE y[SIZE];
		static TYPE exp_a[SIZE];
		static TYPE exp_sum = 0.0;
		static TYPE exp_div;

		// オーバーフロー対策
		static TYPE Cmax = x[0];
		for (int i = 0; i < (int)SIZE; i++) {
			if (Cmax < x[i]) {
				Cmax = x[i];
			}
		}

		// ソフトマックス関数の分子分母
		for (int i = 0; i < (int)SIZE; i++) {
			exp_a[i] = exps[i].forward(x[i]- Cmax);
			exp_sum = adds[i].forward(exp_sum,exp_a[i]);
			//printf("exp_a[%d] = %lf   ", i, exp_a[i]);
			//printf("exp_sum = %lf\n", exp_sum);
		}

		// ソフトマックス関数の分母
		exp_div = div.forward(exp_sum);
		//printf("exp_div = %lf\n", exp_div);

		for (int i = 0; i < (int)SIZE; i++) {
			y[i] = muls[i].forward(exp_a[i], exp_div);
			//printf("y[%d] = %lf\n", i, y[i]);
		}

		return y;
	}
	template <class TYPE,size_t SIZE>
	TYPE* backward(const TYPE(&dout)[SIZE]) {
		static TYPE* dexp_as[SIZE];
		static TYPE* dexp_asdiv[SIZE];
		static TYPE dexp_a[SIZE];
		static TYPE result[SIZE];
		static TYPE dexp_sum = 0.0;
		static TYPE dexp_div;
		static TYPE tmp;

		for (int i = 0; i < (int)SIZE; i++) {
			dexp_as[i] = muls[i].backward(dout[i]);
			//printf("dexp_a[%d][0] = %lf,dexp_a[%d][1] = %lf\n", i, dexp_as[i][0], i, dexp_as[i][1]);
			// 分子の値（乗算の逆伝播）
			dexp_a[i] = dexp_as[i][0];
			// 分母の値（乗算の逆伝播）
			dexp_sum += dexp_as[i][1];
		}
		// 分母の値（除算の逆伝播）
		dexp_div = div.backward(dexp_sum);
		
		for (int i = 0; i < (int)SIZE; i++) {
			dexp_asdiv[i] = adds[i].backward(dexp_div);
			tmp = dexp_asdiv[i][1] + dexp_a[i];
			//cout << tmp << endl;
			tmp = exps[i].backward(tmp);
			result[i] = tmp;
		}
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
	CrossEntropyError(){}
	~CrossEntropyError(){
		delete[] logs;
		delete[] muls;
		delete[] adds;
		delete[] result;
	}
	template <
		class TYPE1,
		size_t SIZE1,
		class TYPE2,
		size_t SIZE2
	> double forward(
		TYPE1 (&x)[SIZE1],
		TYPE2 (&t)[SIZE2]
	) {
		if (SIZE1 != SIZE2) {
			cout << "正解ラベルのサイズと";
			cout << "出力レイヤのサイズが";
			cout << "異なります。" << endl;
			return 0.0;
		}

		size = SIZE1;

		logs = new Log[SIZE1];
		muls = new Mul[SIZE1];
		adds = new Add[SIZE1];
		static double log_x;
		static double m;
		static double E = 0.0;

		for (int i = 0; i < (int)SIZE1; i++) {
			// アンダーフロー回避
			if (x[i] <= delta && x[i] >= -delta) {
				x[i] = delta;
			}
			log_x = logs[i].forward(x[i]);
			//cout << log_x << endl;
			m = muls[i].forward(log_x, t[i]);
			//cout << m << endl;
			E = adds[i].forward(E, m);
			//cout << E << endl;
		}
		E = mul.forward(-1, E);
		//cout << E << endl;
		return E;
	}
	double* backward(double dout) {
		result = new double[size];
		static double* tmp;
		static double dlog_x;

		static double* de = mul.backward(dout);
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
	double* result;
	int size;
	const double delta = 1e-10;
};

class SimpleNet {
public:
	SimpleNet(
		double input_size,
		double hidden_size,
		double output_size) {
		cout << "初期化スタート" << endl;
		
	}
	~SimpleNet() {

	}
private:
	double** weight1;
	double** weight2;
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
	double* y = softmax.forward(arr);
	double* dx = softmax.backward(darr);

	for (int i = 0; i < 10; i++) {
		//printf("y[%d] = %lf\n", i, y[i]);
		printf("dx[%d] = %lf\n", i, dx[i]);
	}
}

void testCrossEntropyError(void) {
	CrossEntropyError cee;
	double x[2] = { 1.23,3.0 };
	double t[2] = { 1,0 };

	double E = cee.forward(x, t);
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
	SimpleNet model(2,3,2);
}

#pragma endregion

int main(void) {
	testPointer();
	return 0;
}