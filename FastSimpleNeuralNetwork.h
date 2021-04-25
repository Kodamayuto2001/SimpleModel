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
	�������x����one_hot�\���ɂ���

	����
		�P�D�������x���i�����j�̃A�h���X
		�Q�Done_hot�ϊ���̊i�[�p�z��̐擪�A�h���X
		
 */
void Flatten(int* label, float one_hot[]) {
	for (int i = 0; i < OUTPUT; ++i) {
		one_hot[i] = 0;
	}
	one_hot[(*label)] = 1;
}

/*
	�V�O���C�h�֐��̏��`�d

	����
		�P�D�������֐���ʂ��O�̐���̘a�̕ϐ��̃A�h���X
		�Q�D��������i�[�p�ϐ��̃A�h���X
*/
void Sigmoid_forward(float* a, float* y) {
	*y = 1.0 / (1.0 + exp(-(*a)));
}
/*
	�V�O���C�h�֐��̋t�`�d

	����
		�P�D�O�m�[�h�ɋt�`�d�l���i�[���邽�߂̕ϐ��̃A�h���X
		�Q�D���`�d��������������̒l�i�[�p�ϐ��̃A�h���X
		�R�D�o�͑w������t�`�d���Ă����l�i�[�p�ϐ��̃A�h���X
*/
void Sigmoid_backward(float* da, float* y, float* dout) {
	*da = (*dout) * (1.0 - (*y)) * (*y);
}
/*
	ReLU�֐��̏��`�d

	����
		�P�D�������֐���ʂ��O�̐���̘a�̕ϐ��̃A�h���X
		�Q�D��������i�[�p�ϐ��̃A�h���X
*/
void ReLU_forward(float* a, float* y) {
	if ((*a) > 0.0) { (*y) = (*a); }
	else { (*y) = (float)0; }
}

/*
	ReLU�֐��̋t�`�d

	����
		�P�D�O�m�[�h�ɋt�`�d�l���i�[���邽�߂̕ϐ��̃A�h���X
		�Q�D���`�d���X�C�b�`�p�ϐ��i���`����������������ϐ��j�̃A�h���X
		�R�D�o�͑�����t�`�d���Ă����l�i�[�p�ϐ��̃A�h���X
*/
void ReLU_backward(float* da, float* y, float* dout) {
	if ((*y) > 0.0) { (*da) = (*dout); }
	else { (*da) = 0.0; }
}
/*
	Softmax�֐��̏��`�d

	����
		�P�D�֐��ɒʂ��O�̏o�͑w�̃T�C�Y�̔z��̐擪�A�h���X
		�Q�D�֐��ŋ��܂����m�����i�[���邽�߂̔z��̐擪�A�h���X
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
	�����G���g���s�[�덷�֐��̏��`�d

	����
		�P�DSoftmax�֐��ŋ��߂����̂��i�[�����z��̐擪�A�h���X
		�Q�Done_hot�\���̐������x���̔z��̐擪�A�h���X
		�R�D���̊֐��ŋ��߂������l���i�[���邽�߂̕ϐ��̃A�h���X
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
	�\�t�g�}�b�N�X�֐��ƌ����G���g���s�[�덷�֐��̕Δ����֐�

	����
		�P�D�����l���o�̓��C���@�덷�t�`�d�l�i�[�p�z��̐擪�A�h���X
		�Q�DSoftmax_forward�ŋ��߂����̂��i�[�����z��̐擪�A�h���X
		�R�Done_hot�\���̐������x���̔z��̐擪�A�h���X
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
	�v�Z�ɕK�v�Ȕz��i�O���[�o���錾�j
*/
//	���`�d
float weight_1[HIDDEN][INPUT];
float weight_2[OUTPUT][HIDDEN];
float bias_1[HIDDEN];
float bias_2[OUTPUT];
//	�t�`�d�Ō��z�����߂�
float dweight_1[HIDDEN][INPUT];
float dweight_2[OUTPUT][HIDDEN];
float dbias_1[HIDDEN];
float dbias_2[OUTPUT];

/*
	�d�݂ƃo�C�A�X�̏�����

	����
		�P�D���ԑw�̊������֐��̃A�h���X
*/
void SimpleNeuralNetwork_init(void(*activation_func)(float*, float*) = ReLU_forward) {
#ifdef PREDICT
	printf("���������Ă��܂���I\n");
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
	���z�����߂�B���`�d�̗\���l��\�����邱�Ƃ��ł���i#define PREDICT�j

	����
		�P�D���t�摜�f�[�^�i1�����z��j
		�Q�D�������x���i1�����z��j
		�R�D�����֐��̕ϐ��̃A�h���X
		�S�D���ԑw�̊������֐��̃A�h���X
		�T�D�o�͑w�̊������֐��̃A�h���X
		�U�D�����֐��̃A�h���X
		�V�DSoftmaxWithLoss���z�Z�o�֐��̃A�h���X
		�W�D���ԑw�̊������֐��̌��z�Z�o�֐��̃A�h���X
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
	printf("\n\n----------�\���l----------\n");
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
	�~�j�o�b�`�ł̌����G���g���s�[�덷�֐�
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
	�~�j�o�b�`�ł�SimpleNeuralNetwork
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
		printf("\n\n----------[�\���l] [�������x��]----------\n");
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
	Batch Normalization�t���̃~�j�o�b�`�ł�SimpleNeuralNetwork
	���܂��ł��Ă��Ȃ��ł��B
	�܂��������Ă��܂���B
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
		
			//	�������֐��P
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
		//	�������֐��Q
		fc_2(yDw2[n], yDw2[n]);

#ifdef PREDICT
		printf("\n\n----------[�\���l] [�������x��]----------\n");
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
	�œK���A���S���Y��SGD

	����
		�P�D�w�K��
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
	�œK���A���S���Y��Momentum

	����
		�P�D�w�K��
		�Q�D�^���W��
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
	�œK���A���S���Y��AdaGrad

	����
		�P�D�w�K��
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
	�œK���A���S���Y��RMSProp

	����
		�P�D�w�K��
		�Q�D�ߋ��̌��z�̖Y��Ă����x�����̌W��
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
	�œK���A���S���Y��Adam

	����
		�P�D�w�K��
		�Q�D�������P
		�R�D�������Q
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
	�d�݂ƃo�C�A�X��ۑ�����֐�

	����
		�P�D�t�@�C����
*/
void save(const char* fileName = "test.model") {
	FILE* fp;
	fopen_s(&fp, fileName, "wb");
	if (fp == NULL) {
		printf("�t�@�C�����J���܂���ł����B\n");
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
	printf("�p�����[�^��ۑ����܂����B\n");
}
/*
	�ۑ������d�݂ƃo�C�A�X�����[�h����֐�

	����
		�P�D�t�@�C����
*/
void load(const char* fileName = "test.model") {
	FILE* fp;
	fopen_s(&fp, fileName, "rb");
	if (fp == NULL) {
		printf("�t�@�C�����J���܂���ł����B\n");
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
	printf("���f���p�����[�^�����[�h���܂����B\n");
}

#endif // !_FastSimpleNeuralNetwork_H_
