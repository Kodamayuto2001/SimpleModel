#ifndef _FastSimpleNeuralNetwork_H_
#define _FastSimpleNeuralNetwork_H_
#ifndef DATA_MAX
#define DATA_MAX 29000
#endif // !DATA_MAX
#ifndef HEIGHT
#define HEIGHT 160
#endif // !HEIGHT
#ifndef WIDTH
#define WIDTH 160
#endif // !WIDTH
#ifndef INPUT
#define INPUT HEIGHT * WIDTH
#endif // !INPUT
#ifndef HIDDEN
#define HIDDEN 320
#endif // !HIDDEN
#ifndef OUTPUT
#define OUTPUT 29
#endif // !OUTPUT


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
	正解ラベルをone_hot表現にする

	引数
		１．正解ラベル（整数）のアドレス
		２．one_hot変換後の格納用配列の先頭アドレス
		
 */
void Flatten(int* label, float one_hot[]) {
	for (int i = 0; i < OUTPUT; ++i) {
		one_hot[i] = 0;
	}
	one_hot[(*label)] = 1;
}

/*
	シグモイド関数の順伝播

	引数
		１．活性化関数を通す前の数列の和の変数のアドレス
		２．活性化後格納用変数のアドレス
*/
void Sigmoid_forward(float* a, float* y) {
	*y = 1.0 / (1.0 + exp(-(*a)));
}
/*
	シグモイド関数の逆伝播

	引数
		１．前ノードに逆伝播値を格納するための変数のアドレス
		２．順伝播時活性化処理後の値格納用変数のアドレス
		３．出力層側から逆伝播してきた値格納用変数のアドレス
*/
void Sigmoid_backward(float* da, float* y, float* dout) {
	*da = (*dout) * (1.0 - (*y)) * (*y);
}
/*
	ReLU関数の順伝播

	引数
		１．活性化関数を通す前の数列の和の変数のアドレス
		２．活性化後格納用変数のアドレス
*/
void ReLU_forward(float* a, float* y) {
	if ((*a) > 0.0) { (*y) = (*a); }
	else { (*y) = (float)0; }
}

/*
	ReLU関数の逆伝播

	引数
		１．前ノードに逆伝播値を格納するための変数のアドレス
		２．順伝播時スイッチ用変数（順伝搬時活性化処理後変数）のアドレス
		３．出力側から逆伝播してきた値格納用変数のアドレス
*/
void ReLU_backward(float* da, float* y, float* dout) {
	if ((*y) > 0.0) { (*da) = (*dout); }
	else { (*da) = 0.0; }
}
/*
	Softmax関数の順伝播

	引数
		１．関数に通す前の出力層のサイズの配列の先頭アドレス
		２．関数で求まった確率を格納するための配列の先頭アドレス
*/
void Softmax_forward(float score[OUTPUT], float rate[OUTPUT]) {
	float a = score[0];
	float tmp[OUTPUT];
	float b = 0.0;
	for (int i = 0; i < OUTPUT; ++i) {
		if (a < score[i]) { a = score[i]; }	
	}
	for (int i = 0; i < OUTPUT; ++i) {
		tmp[i] = exp((double)score[i] - a);
		b += tmp[i];
	}
	a = 1 / b;
	for (int i = 0; i < OUTPUT; ++i) {
		rate[i] = tmp[i] * a;
	}
}

/*
	交差エントロピー誤差関数の順伝播

	引数
		１．Softmax関数で求めたものを格納した配列の先頭アドレス
		２．one_hot表現の正解ラベルの配列の先頭アドレス
		３．この関数で求めた損失値を格納するための変数のアドレス
*/
void CrossEntropyError_forward(
	float rate[OUTPUT],
	float t[OUTPUT], 
	float* loss
) {
	float a = 0.0;
	for (int i = 0; i < OUTPUT; ++i) {
		a += log(rate[i] + 1.0e-10) * t[i];
	}
	*loss = a * (-1);
}
/*
	ソフトマックス関数と交差エントロピー誤差関数の偏微分関数

	引数
		１．損失値→出力レイヤ　誤差逆伝播値格納用配列の先頭アドレス
		２．Softmax_forwardで求めたものを格納した配列の先頭アドレス
		３．one_hot表現の正解ラベルの配列の先頭アドレス
*/
void SoftmaxWithLoss_backward(
	float drate[OUTPUT], 
	float y[OUTPUT], 
	float t[OUTPUT]
) {
	for (int i = 0; i < OUTPUT; ++i) {
		drate[i] = y[i] - t[i];
	}
}

/*
	計算に必要な配列（グローバル宣言）
*/
//	順伝播
float weight_1[HIDDEN][INPUT];
float weight_2[OUTPUT][HIDDEN];
float bias_1[HIDDEN];
float bias_2[OUTPUT];
//	逆伝播で勾配を求める
float dweight_1[HIDDEN][INPUT];
float dweight_2[OUTPUT][HIDDEN];
float dbias_1[HIDDEN];
float dbias_2[OUTPUT];

/*
	重みとバイアスの初期化

	引数
		１．中間層の活性化関数のアドレス
*/
void SimpleNeuralNetwork_init(void(*activation_func)(float*, float*) = ReLU_forward) {
#ifdef PREDICT
	printf("初期化していません！\n");
#else
	float a = 0.01;
	if (activation_func == Sigmoid_forward) {
		srand(time(NULL));
		for (int i = 0; i < HIDDEN; ++i) {
			bias_1[i] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) / (float)sqrt((double)INPUT);
			dbias_1[i] = 0;
			for (int j = 0; j < INPUT; ++j) {
				weight_1[i][j] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) / (float)sqrt((double)INPUT);
				dweight_1[i][j] = 0;
			}
		}
		for (int i = 0; i < OUTPUT; ++i) {
			bias_2[i] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) / (float)sqrt((double)HIDDEN);
			dbias_2[i] = 0;
			for (int j = 0; j < HIDDEN; ++j) {
				weight_2[i][j] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) / (float)sqrt((double)HIDDEN);
				dweight_2[i][j] = 0;
			}
		}
	}
	else if (activation_func == ReLU_forward) {
		srand(time(NULL));
		for (int i = 0; i < HIDDEN; ++i) {
			bias_1[i] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) / (float)sqrt((double)INPUT) * sqrt(2);
			dbias_1[i] = 0;
			for (int j = 0; j < INPUT; ++j) {
				weight_1[i][j] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) / (float)sqrt((double)INPUT) * sqrt(2);
				dweight_1[i][j] = 0;
			}
		}
		for (int i = 0; i < OUTPUT; ++i) {
			bias_2[i] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) / (float)sqrt((double)HIDDEN) * sqrt(2);
			dbias_2[i] = 0;
			for (int j = 0; j < HIDDEN; ++j) {
				weight_2[i][j] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) / (float)sqrt((double)HIDDEN * sqrt(2));
				dweight_2[i][j] = 0;
			}
		}
	}
	else
	{
		srand(time(NULL));
		for (int i = 0; i < HIDDEN; ++i) {
			bias_1[i] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) * a;
			dbias_1[i] = 0;
			for (int j = 0; j < INPUT; ++j) {
				weight_1[i][j] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) * a;
				dweight_1[i][j] = 0;
			}
		}
		for (int i = 0; i < OUTPUT; ++i) {
			bias_2[i] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) * a;
			dbias_2[i] = 0;
			for (int j = 0; j < HIDDEN; ++j) {
				weight_2[i][j] = (float)(1.0 - (double)((float)rand() / (RAND_MAX / 2))) * a;
				dweight_2[i][j] = 0;
			}
		}
	}
#endif // PREDICT

}
/*
	勾配を求める。順伝播の予測値を表示することもできる（#define PREDICT）

	引数
		１．教師画像データ（1次元配列）
		２．正解ラベル（1次元配列）
		３．損失関数の変数のアドレス
		４．中間層の活性化関数のアドレス
		５．出力層の活性化関数のアドレス
		６．損失関数のアドレス
		７．SoftmaxWithLoss勾配算出関数のアドレス
		８．中間層の活性化関数の勾配算出関数のアドレス
*/
void SimpleNeuralNetwork(
	float x[INPUT],
	float t[OUTPUT],
	float* loss,
	void(*fc_1)(float*, float*)					= ReLU_forward,
	void(*fc_2)(float[], float[])				= Softmax_forward,
	void(*lossFunc)(float[], float[], float*)	= CrossEntropyError_forward,
	void(*gc_2)(float[], float[], float[])		= SoftmaxWithLoss_backward,
	void(*gc_1)(float*, float*, float*)			= ReLU_backward
) {
	float tmp[HIDDEN];
	for (int i = 0; i < HIDDEN; ++i) {
		dbias_1[i] = 0;
		tmp[i] = 0;
		for (int j = 0; j < INPUT; ++j) {
			dbias_1[i] += x[j] * weight_1[i][j];
			dweight_1[i][j] = x[j];
		}
		dbias_1[i] += bias_1[i];
		fc_1(&dbias_1[i], &dbias_1[i]);
	}

	for (int i = 0; i < OUTPUT; ++i) {
		dbias_2[i] = 0;
		for (int j = 0; j < HIDDEN; ++j) {
			dbias_2[i] += dbias_1[j] * weight_2[i][j];
			dweight_2[i][j] = dbias_1[j];
		}
		dbias_2[i] += bias_2[i];
	}

	fc_2(dbias_2, dbias_2);

#ifdef PREDICT
	printf("\n\n----------予測値----------\n");
	for (int i = 0; i < OUTPUT; ++i) {
		printf("p[%d] = %f\n", i, dbias_2[i]);
	}
#endif // PREDICT

	lossFunc(dbias_2, t, loss);

	gc_2(dbias_2, dbias_2, t);

	for (int i = 0; i < OUTPUT; ++i) {
		for (int j = 0; j < HIDDEN; ++j) {
			tmp[j] += dbias_2[i] * weight_2[i][j];
			dweight_2[i][j] *= dbias_2[i];
		}
	}

	for (int i = 0; i < HIDDEN; ++i) {
		gc_1(&dbias_1[i], &dbias_1[i], &tmp[i]);
		tmp[i] = 0;
		for (int j = 0; j < INPUT; ++j) {
			dweight_1[i][j] *= dbias_1[i];
		}
	}
}

#ifdef MINI_BATCH_SIZE
/*
	ミニバッチ版の交差エントロピー誤差関数
*/
void MiniBatchCrossEntropyError_forward(
	float rate[MINI_BATCH_SIZE][OUTPUT],
	float t[MINI_BATCH_SIZE][OUTPUT],
	float* loss
) {
	float a = 0.0;
	for (int i = 0; i < MINI_BATCH_SIZE; ++i) {
		for (int j = 0; j < OUTPUT; ++j) {
			a += log(rate[i][j] + 1.0e-10) * t[i][j];
		}
	}
	*loss = a * (-1) / MINI_BATCH_SIZE;
}
/*
	ミニバッチ版のSimpleNeuralNetwork
*/
void MiniBatchSimpleNeuralNetwork(
	float x[MINI_BATCH_SIZE][INPUT],
	float t[MINI_BATCH_SIZE][OUTPUT],
	float* loss,
	void(*fc_1)(float*, float*)																		= ReLU_forward,
	void(*fc_2)(float[OUTPUT], float[OUTPUT])														= Softmax_forward,
	void(*miniBatchLossF)(float[MINI_BATCH_SIZE][OUTPUT], float[MINI_BATCH_SIZE][OUTPUT], float*)	= MiniBatchCrossEntropyError_forward,
	void(*gc_2)(float[OUTPUT], float[OUTPUT], float[OUTPUT])										= SoftmaxWithLoss_backward,
	void(*gc_1)(float*, float*, float*)																= ReLU_backward
) {
	static float xDw1[MINI_BATCH_SIZE][HIDDEN];
	static float yDw2[MINI_BATCH_SIZE][OUTPUT];
	static float tmp[MINI_BATCH_SIZE][HIDDEN];

	for (int n = 0; n < MINI_BATCH_SIZE; ++n) {
		for (int i = 0; i < HIDDEN; ++i) {
			xDw1[n][i] = 0;
			tmp[n][i] = 0;
			dbias_1[i] = 0;
			for (int j = 0; j < INPUT; ++j) {
				dweight_1[i][j] = 0;
				xDw1[n][i] += x[n][j] * weight_1[i][j];
			}
			xDw1[n][i] += bias_1[i];
			fc_1(&xDw1[n][i], &xDw1[n][i]);
		}

		for (int i = 0; i < OUTPUT; ++i) {
			dbias_2[i] = 0;
			yDw2[n][i] = 0;
			for (int j = 0; j < HIDDEN; ++j) {
				dweight_2[i][j] = 0;
				yDw2[n][i] += xDw1[n][j] * weight_2[i][j];
			}
			yDw2[n][i] += bias_2[i];
		}

		fc_2(yDw2[n], yDw2[n]);

#ifdef PREDICT
		printf("\n\n----------[予測値] [正解ラベル]----------\n");
		for (int i = 0; i < OUTPUT; ++i) {
			printf("p[%d] = %f t[%d] = %d\n", i, yDw2[n][i], i, t[n][i]);
		}
#endif // PREDICT
	}

	miniBatchLossF(yDw2, t, loss);

	for (int n = 0; n < MINI_BATCH_SIZE; ++n) {
		gc_2(yDw2[n], yDw2[n], t[n]);

		for (int i = 0; i < OUTPUT; ++i) {
			dbias_2[i] += yDw2[n][i];
			for (int j = 0; j < HIDDEN; ++j) {
				dweight_2[i][j] += yDw2[n][i] * xDw1[n][j];
				tmp[n][j] += yDw2[n][i] * weight_2[i][j];
			}
		}

		for (int i = 0; i < HIDDEN; ++i) {
			gc_1(&tmp[n][i], &xDw1[n][i], &tmp[n][i]);
			dbias_1[i] += tmp[n][i];
			for (int j = 0; j < INPUT; ++j) {
				dweight_1[i][j] += tmp[n][i] * x[n][j];
			}
		}
	}
}

float ganma = 1;
float beta_ = 0;
float dganma= 0;
float dbeta_= 0;
/*
	Batch Normalization付きのミニバッチ版のSimpleNeuralNetwork
	うまくできていないです。
	まだ完成していません。
*/
void MiniBatchSimpleNeuralNetwork_BatchNorm(
	float x[MINI_BATCH_SIZE][INPUT],
	float t[MINI_BATCH_SIZE][OUTPUT],
	float* loss,
	void(*fc_1)(float*, float*) = ReLU_forward,
	void(*fc_2)(float[OUTPUT], float[OUTPUT]) = Softmax_forward,
	void(*miniBatchLossF)(float[MINI_BATCH_SIZE][OUTPUT], float[MINI_BATCH_SIZE][OUTPUT], float*) = MiniBatchCrossEntropyError_forward,
	void(*gc_2)(float[OUTPUT], float[OUTPUT], float[OUTPUT]) = SoftmaxWithLoss_backward,
	void(*gc_1)(float*, float*, float*) = ReLU_backward
) {
	static float xDw1[MINI_BATCH_SIZE][HIDDEN];
	static float yDw2[MINI_BATCH_SIZE][OUTPUT];
	static float tmp[MINI_BATCH_SIZE][HIDDEN];

	float bn_mean[HIDDEN];
	float bn_variance[HIDDEN];
	static float bn_norm[MINI_BATCH_SIZE][HIDDEN];
	static float y__1[MINI_BATCH_SIZE][HIDDEN];

	for (int n = 0; n < MINI_BATCH_SIZE; ++n) {
		for (int i = 0; i < HIDDEN; ++i) {
			xDw1[n][i] = 0;
			tmp[n][i] = 0;
			dbias_1[i] = 0;
			if (n == 0){ bn_mean[i] = 0; }
			
			for (int j = 0; j < INPUT; ++j) {
				xDw1[n][i] += x[n][j] * weight_1[i][j];
			}
			xDw1[n][i] += bias_1[i];

//	-----	mini-batch mean (sum)	-----
			bn_mean[i] += xDw1[n][i];
		}
	}
	
//	-----	mini-batch mean (div)	-----
	for (int i = 0; i < HIDDEN; ++i) {
		bn_mean[i] /= MINI_BATCH_SIZE;
	}

//	-----	mini-batch variance(sum)-----
	for (int n = 0; n < MINI_BATCH_SIZE; ++n) {
		for (int i = 0; i < HIDDEN; ++i) {
			bn_variance[i] = pow(xDw1[n][i] - bn_mean[i], 2);
		}
	}

//	-----	mini-batch variance(div)-----
	for (int i = 0; i < HIDDEN; ++i) {
		bn_variance[i] /= MINI_BATCH_SIZE;
	}

//	-----	mini-batch normalize	-----
	for (int n = 0; n < MINI_BATCH_SIZE; ++n) {
		for (int i = 0; i < HIDDEN; ++i) {
			bn_norm[n][i] = (xDw1[n][i] - bn_mean[i]);
			bn_norm[n][i] /= sqrt(bn_variance[i] + 1.0e-10);
		}
	}

	for (int n = 0; n < MINI_BATCH_SIZE; ++n) {
		for (int i = 0; i < HIDDEN; ++i) {
			//	-----	scale and shift		-----
			y__1[n][i] = ganma * bn_norm[n][i] + beta_;
		
			//	活性化関数１
			fc_1(&y__1[n][i], &y__1[n][i]);
		}

		for (int i = 0; i < OUTPUT; ++i) {
			dbias_2[i] = 0;
			yDw2[n][i] = 0;
			for (int j = 0; j < HIDDEN; ++j) {
				dweight_2[n][j] = 0;
				yDw2[n][i] += y__1[n][j] * weight_2[i][j];
			}
			yDw2[n][i] += bias_2[i];
		}
		//	活性化関数２
		fc_2(yDw2[n], yDw2[n]);

#ifdef PREDICT
		printf("\n\n----------[予測値] [正解ラベル]----------\n");
		for (int i = 0; i < OUTPUT; ++i) {
			printf("p[%d] = %f t[%d] = %d\n", i, yDw2[n][i], i, t[n][i]);
		}
#endif // PREDICT
	}

	miniBatchLossF(yDw2, t, loss);

	for (int n = 0; n < MINI_BATCH_SIZE; ++n) {
		gc_2(yDw2[n], yDw2[n], t[n]);

		for (int i = 0; i < OUTPUT; ++i) {
			dbias_2[i] += yDw2[n][i];
			for (int j = 0; j < HIDDEN; ++j) {
				dweight_2[i][j] += yDw2[n][i] * xDw1[n][j];
				tmp[n][j] += yDw2[n][i] * weight_2[i][j];
			}
		}

		for (int i = 0; i < HIDDEN; ++i) {
			gc_1(&tmp[n][i], &y__1[n][i], &tmp[n][i]);

			dbeta_ += tmp[n][i];
			dganma += bn_norm[n][i] * tmp[n][i];
			bn_norm[n][i] = ganma * tmp[n][i];
			bn_norm[n][i] /= (bn_variance[i] + 1.0e-10);
		}
	}
}

#endif // MINI_BATCH_SIZE

/*
	最適化アルゴリズムSGD

	引数
		１．学習率
*/
void SGD(float lr = 0.01) {
	for (int i = 0; i < HIDDEN; ++i) {
		bias_1[i] -= lr * dbias_1[i];
		for (int j = 0; j < INPUT; ++j) {
			weight_1[i][j] -= lr * dweight_1[i][j];
		}
	}

	for (int i = 0; i < OUTPUT; ++i) {
		bias_2[i] -= lr * dbias_2[i];
		for (int j = 0; j < HIDDEN; ++j) {
			weight_2[i][j] -= lr * dweight_2[i][j];
		}
	}
}
/*
	最適化アルゴリズムMomentum

	引数
		１．学習率
		２．運動係数
*/
void Momentum(float lr = 0.01, float momentum = 0.9) {
	static int flag = 0;
	static float v_weight_1[HIDDEN][INPUT];
	static float v_weight_2[OUTPUT][HIDDEN];
	static float v_bias_1[HIDDEN];
	static float v_bias_2[OUTPUT];

	if (flag == 0) {
		flag = 1;
		for (int i = 0; i < HIDDEN; ++i) {
			v_bias_1[i] = 0;
			for (int j = 0; j < INPUT; ++j) {
				v_weight_1[i][j] = 0;
			}
		}
		for (int i = 0; i < OUTPUT; ++i) {
			v_bias_2[i] = 0;
			for (int j = 0; j < HIDDEN; ++j) {
				v_weight_2[i][j] = 0;
			}
		}
	}

	for (int i = 0; i < HIDDEN; ++i) {
		v_bias_1[i] *= momentum;
		v_bias_1[i] -= lr * dbias_1[i];
		bias_1[i] += v_bias_1[i];
		for (int j = 0; j < INPUT; ++j) {
			v_weight_1[i][j] *= momentum;
			v_weight_1[i][j] -= lr * dweight_1[i][j];
			weight_1[i][j] += v_weight_1[i][j];
		}
	}

	for (int i = 0; i < OUTPUT; ++i) {
		v_bias_2[i] *= momentum;
		v_bias_2[i] -= lr * dbias_2[i];
		bias_2[i] += v_bias_2[i];
		for (int j = 0; j < HIDDEN; ++j) {
			v_weight_2[i][j] *= momentum;
			v_weight_2[i][j] -= lr * dweight_2[i][j];
			weight_2[i][j] += v_weight_2[i][j];
		}
	}
}
/*
	最適化アルゴリズムAdaGrad

	引数
		１．学習率
*/
void AdaGrad(float lr = 0.01) {
	static int flag = 0;
	static float h_weight_1[HIDDEN][INPUT];
	static float h_weight_2[OUTPUT][HIDDEN];
	static float h_bias_1[HIDDEN];
	static float h_bias_2[OUTPUT];

	if (flag == 0) {
		flag = 1;
		for (int i = 0; i < HIDDEN; ++i) {
			h_bias_1[i] = 0;
			for (int j = 0; j < INPUT; ++j) {
				h_weight_1[i][j] = 0;
			}
		}
		for (int i = 0; i < OUTPUT; ++i) {
			h_bias_2[i] = 0;
			for (int j = 0; j < HIDDEN; ++j) {
				h_weight_2[i][j] = 0;
			}
		}
	}


	for (int i = 0; i < HIDDEN; ++i) {
		h_bias_1[i] += dbias_1[i] * dbias_1[i];
		bias_1[i] -= lr / sqrt(h_bias_1[i] + 1.0e-10) * dbias_1[i];
		for (int j = 0; j < INPUT; ++j) {
			h_weight_1[i][j] += dweight_1[i][j] * dweight_1[i][j];
			weight_1[i][j] -= lr / sqrt(h_weight_1[i][j] + 1.0e-10) * dweight_1[i][j];
		}
	}

	for (int i = 0; i < OUTPUT; ++i) {
		h_bias_2[i] += dbias_2[i] * dbias_2[i];
		bias_2[i] -= lr / sqrt(h_bias_2[i] + 1.0e-10) * dbias_2[i];
		for (int j = 0; j < HIDDEN; ++j) {
			h_weight_2[i][j] += dweight_2[i][j] * dweight_2[i][j];
			weight_2[i][j] -= lr / sqrt(h_weight_2[i][j] + 1.0e-10) * dweight_2[i][j];
		}
	}
}
/*
	最適化アルゴリズムRMSProp

	引数
		１．学習率
		２．過去の勾配の忘れていく度合いの係数
*/
void RMSProp(float lr = 0.01, float decay_rate = 0.99) {
	static int flag = 0;
	static float h_weight_1[HIDDEN][INPUT];
	static float h_weight_2[OUTPUT][HIDDEN];
	static float h_bias_1[HIDDEN];
	static float h_bias_2[OUTPUT];
	float a;

	if (flag == 0) {
		flag = 1;
		for (int i = 0; i < HIDDEN; ++i) {
			h_bias_1[i] = 0;
			for (int j = 0; j < INPUT; ++j) {
				h_weight_1[i][j] = 0;
			}
		}
		for (int i = 0; i < OUTPUT; ++i) {
			h_bias_2[i] = 0;
			for (int j = 0; j < HIDDEN; ++j) {
				h_weight_2[i][j] = 0;
			}
		}
	}


	for (int i = 0; i < HIDDEN; ++i) {
		h_bias_1[i] *= decay_rate;
		h_bias_1[i] += (1 - decay_rate) * dbias_1[i] * dbias_1[i];
		a = sqrt(h_bias_1[i] + 1.0e-10);
		bias_1[i] -= lr * dbias_1[i] / a;
		for (int j = 0; j < INPUT; ++j) {
			h_weight_1[i][j] *= decay_rate;
			h_weight_1[i][j] += (1 - decay_rate) * dweight_1[i][j] * dweight_1[i][j];
			a = sqrt(h_weight_1[i][j] + 1.0e-10);
			weight_1[i][j] -= lr * dweight_1[i][j] / a;
		}
	}

	for (int i = 0; i < OUTPUT; ++i) {
		h_bias_2[i] *= decay_rate;
		h_bias_2[i] += (1 - decay_rate) * dbias_2[i] * dbias_2[i];
		a = sqrt(h_bias_2[i] + 1.0e-10);
		bias_2[i] -= lr * dbias_2[i] / a;

		for (int j = 0; j < HIDDEN; ++j) {
			h_weight_2[i][j] *= decay_rate;
			h_weight_2[i][j] += (1 - decay_rate) * dweight_2[i][j] * dweight_2[i][j];
			a = sqrt(h_weight_2[i][j] + 1.0e-10);
			weight_2[i][j] -= lr * dweight_2[i][j] / a;
		}
	}
}
/*
	最適化アルゴリズムAdam

	引数
		１．学習率
		２．減衰率１
		３．減衰率２
*/
void Adam(float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999) {
	static float iter = 0;
	static float m_weight_1[HIDDEN][INPUT];
	static float m_weight_2[OUTPUT][HIDDEN];
	static float v_weight_1[HIDDEN][INPUT];
	static float v_weight_2[OUTPUT][HIDDEN];
	static float m_bias_1[HIDDEN];
	static float m_bias_2[OUTPUT];
	static float v_bias_1[HIDDEN];
	static float v_bias_2[OUTPUT];
	
	float a = 0;

	if (iter < 0.5) {
		for (int i = 0; i < HIDDEN; ++i) {
			m_bias_1[i] = 0;
			v_bias_1[i] = 0;
			for (int j = 0; j < INPUT; ++j) {
				m_weight_1[i][j] = 0;
				v_weight_1[i][j] = 0;
			}
		}
		for (int i = 0; i < OUTPUT; ++i) {
			m_bias_2[i] = 0;
			v_bias_2[i] = 0;
			for (int j = 0; j < HIDDEN; ++j) {
				m_weight_2[i][j] = 0;
				v_weight_2[i][j] = 0;
			}
		}
	}

	iter += 1.0;
	float lr_t = lr;
	lr_t *= sqrt(1.0 - pow(beta2, iter));
	lr_t /= (1.0 - pow(beta1, iter));

	for (int i = 0; i < HIDDEN; ++i) {
		m_bias_1[i] += (1 - beta1) * (dbias_1[i] - m_bias_1[i]);
		v_bias_1[i] += (1 - beta2) * (dbias_1[i] * dbias_1[i] - v_bias_1[i]);
		a = sqrt(v_bias_1[i] + 1.0e-10);
		bias_1[i] -= lr_t * m_bias_1[i] / a;
		for (int j = 0; j < INPUT; ++j) {
			m_weight_1[i][j] += (1 - beta1) * (dweight_1[i][j] - m_weight_1[i][j]);
			v_weight_1[i][j] += (1 - beta2) * (dweight_1[i][j] * dweight_1[i][j] - v_weight_1[i][j]);
			a = sqrt(v_weight_1[i][j] + 1.0e-10);
			weight_1[i][j] -= lr_t * m_weight_1[i][j] / a;
		}
	}

	for (int i = 0; i < OUTPUT; ++i) {
		m_bias_2[i] += (1 - beta1) * (dbias_2[i] - m_bias_2[i]);
		v_bias_2[i] += (1 - beta2) * (dbias_2[i] * dbias_2[i] - v_bias_2[i]);
		a = sqrt(v_bias_2[i] + 1.0e-10);
		bias_2[i] -= lr_t * m_bias_2[i] / a;
		for (int j = 0; j < HIDDEN; ++j) {
			m_weight_2[i][j] += (1 - beta1) * (dweight_2[i][j] - m_weight_2[i][j]);
			v_weight_2[i][j] += (1 - beta2) * (dweight_2[i][j] * dweight_2[i][j] - v_weight_2[i][j]);
			a = sqrt(v_weight_2[i][j] + 1.0e-10);
			weight_2[i][j] -= lr_t * m_weight_2[i][j] / a;
		}
	}
}
/*
	重みとバイアスを保存する関数

	引数
		１．ファイル名
*/
void save(const char* fileName = "test.model") {
	FILE* fp;
	fopen_s(&fp, fileName, "wb");
	if (fp == NULL) {
		printf("ファイルが開けませんでした。\n");
		exit(1);
	}

	fwrite((char*)bias_1, sizeof(bias_1[0]), HIDDEN, fp);
	for (int i = 0; i < HIDDEN; ++i) {
		fwrite((char*)weight_1[i], sizeof(weight_1[i][0]), INPUT, fp);
	}
	fwrite((char*)bias_2, sizeof(bias_2[0]), OUTPUT, fp);
	for (int i = 0; i < OUTPUT; ++i) {
		fwrite((char*)weight_2[i], sizeof(weight_2[i][0]), OUTPUT, fp);
	}
	fclose(fp);
	printf("パラメータを保存しました。\n");
}
/*
	保存した重みとバイアスをロードする関数

	引数
		１．ファイル名
*/
void load(const char* fileName = "test.model") {
	FILE* fp;
	fopen_s(&fp, fileName, "rb");
	if (fp == NULL) {
		printf("ファイルが開けませんでした。\n");
		exit(1);
	}

	fread((char*)bias_1, sizeof(bias_1[0]), HIDDEN, fp);
	for (int i = 0; i < HIDDEN; ++i) {
		fread((char*)weight_1[i], sizeof(weight_1[i][0]), INPUT, fp);
	}
	fread((char*)bias_2, sizeof(bias_2[0]), OUTPUT, fp);
	for (int i = 0; i < OUTPUT; ++i) {
		fread((char*)weight_2[i], sizeof(weight_2[i][0]), OUTPUT, fp);
	}
	fclose(fp);
	printf("モデルパラメータをロードしました。\n");
}

#endif // !_FastSimpleNeuralNetwork_H_
