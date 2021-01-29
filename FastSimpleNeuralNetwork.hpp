#ifndef _FastSimpleNeuralNetwork_HPP_
#define _FastSimpleNeuralNetwork_HPP_
#include<iostream>
#include<fstream>
#include<float.h>
#include<string>
#include<random>
#include<cmath>
using namespace std;
constexpr double Delta = 1.0e-100;

/****************************************************************************************
	FastSigmoid

	forward�֐�	
		#	double* x		���`�����̓|�C���^
		#	double* y		���`���o�̓|�C���^

	backward�֐�
		#	double* dout	�t�`�����̓|�C���^
		#	double* dx		�t�`���o�̓|�C���^

****************************************************************************************/
class FastSigmoid {
public:
	void forward(double* x, double* y) {
		*y = 1 / (1.0 + exp(-(*x)));
		FastSigmoid::y = y;
	}
	void backward(double* dout, double* dx) {
		*dx = (*dout) * (1.0 - (*y)) * (*y);
	}
private:
	double* y;
};

/****************************************************************************************
	FastReLU

	forward�֐�
		#	double* _x		���`�����̓|�C���^
		#	double* y		���`���o�̓|�C���^

	backward�֐�
		#	double* dout	�t�`�����̓|�C���^
		#	double* dx		�t�`���o�̓|�C���^

****************************************************************************************/
class FastReLU {
public:
	void forward(double* _x, double* y) {
		FastReLU::x = _x;
		if (*x > 0.0) { *y = *x; }
		else { *y = 0.0; }
	}
	void backward(double* dout, double* dx) {
		if (*x > 0.0) { *dx = *dout; }
		else { *dx = 0.0; }
	}
private:
	double* x;
};

/****************************************************************************************
	FastSoftmaxWithLoss

	__init__�֐�
		#	int* _size		�j���[�����l�b�g���[�N�ŏI�o�͑w��

	SoftmaxForward�֐�
		#	double* x		�\�t�g�}�b�N�X�֐����`������
		#	double* y		�\�t�g�}�b�N�X�֐����`���o��

	CrossEntropyErrorForward�֐�
		#	double* _t		�����G���g���s�[�덷�֐����`�����́i�������x���j
		#	double*	loss	�����G���g���s�[�덷�֐����`���o�́i�����l�j

	backward�֐�
		#	double* dx		�\�t�g�}�b�N�X�֐��ƌ����G���g���s�[�덷�֐��̋t�`���o��
****************************************************************************************/
class FastSoftmaxWithLoss {
public:
	void __init__(int* _size) {
		size = _size;
		tmp = new double[(*size)];
		a = b = 0.0;
		i = 0;
	}
	void SoftmaxForward(double* x, double* y) {
		a = x[0];
		for (i = 0; i < (int)(*size); ++i) {
			if (a < x[i]) {
				a = x[i];
			}
		}
		b = 0.0;
		for (i = 0; i < (int)(*size); ++i) {
			tmp[i] = exp(x[i] - a);
			b += tmp[i];
		}
		a = 1 / b;
		for (i = 0; i < (int)(*size); ++i) {
			y[i] = tmp[i] * a;
		}
		FastSoftmaxWithLoss::y = y;
	}

	void CrossEntropyErrorForward(double* _t, double* loss) {
		t = _t;
		_t = nullptr;
		b = 0.0;
		for (i = 0; i < (int)(*size); ++i) {
			a = log(y[i]);
			b += a * t[i];
		}
		*loss = b * (-1);
	}

	void backward(double* dx) {
		for (i = 0; i < (int)(*size); ++i) {
			dx[i] = y[i] - t[i];
		}
	}

	void del() {
		delete[] tmp;
	}
private:
	int* size;
	double* tmp;
	double* t;
	double a, b;
	int i;
	double* y;
};

/****************************************************************************************
	FastModel
	�l�H�m�\�̃��f���B�v�Z�𑬂����邽�߃|�C���^�Ȃǂ��g���B
	#	ActFunc			�������֐��i���ԑw�j
	#	SoftmaxWithLoss	�o�͂̊֐��i�������֐��{�����֐��j

	FastModel�R���X�g���N�^
		���Z�ɕK�v�ȕϐ��𓮓I�Ƀ����������蓖�āA����������B
		#	int&& inputSize		���f���̓��͑w�i���[�u�Z�}���e�B�N�X�j
		#	int&& hiddenSize	���f���̒��ԑw�i���[�u�Z�}���e�B�N�X�j
		#	int&& outputSize	���f���̏o�͑w�i���[�u�Z�}���e�B�N�X�j

	del�֐�
		���Z���I�������Ƃ��ɌĂяo���B���I�Ƀ��������蓖�Ă��ϐ����������

	predict�֐�
		���f�������`�������Ƃ��̗\���l���v�Z����B�\���l�́Anode[1][n]�Ɋi�[�����(n�̓��f���̍ŏI�w��)
		#	double* x	���t�f�[�^

	forward�֐�
		���f�������`�����A�����l���v�Z����B�����l�́Aloss�Ɋi�[�����B
		#	double* x	���t�f�[�^
		#	double* t	�������x��

	backward�֐�
		�p�����[�^���X�V���A�w�K����B

	save�֐�
		���f���̃p�����[�^��ۑ�����B
		#	const char* fileName	�ۑ�����t�@�C�����i�g���q�͓K���j

	load�֐�
		���f���̃p�����[�^��ǂݍ��ށB
		#	const char* fileName	�ǂݍ��ރ��f���̃t�@�C����
****************************************************************************************/
template<class ActFunc, class SoftmaxWithLoss> class FastModel {
public:
	double** weight[2];
	double* bias[2];
	double* node[2];
	double** dweight[2];
	double* dbias[2];
	double loss;
	int size[3];

	FastModel(int&& inputSize, int&& hiddenSize, int&& outputSize) {
		size[0] = inputSize;
		size[1] = hiddenSize;
		size[2] = outputSize;
		a = 0.0;

		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<double> dist(-1, 1);

		for (i = 0; i < 2; ++i) {
			weight[i] = new double* [size[i + 1]];
			bias[i] = new double[size[i + 1]];
			node[i] = new double[size[i + 1]];
			dweight[i] = new double* [size[i + 1]];
			dbias[i] = new double[size[i + 1]];
			for (j = 0; j < size[i + 1]; ++j) {
				weight[i][j] = new double[size[i]];
				dweight[i][j] = new double[size[i]];
				bias[i][j] = 0.0;
				node[i][j] = 0.0;
				dbias[i][j] = 0.0;
				for (int k = 0; k < size[i]; ++k) {
					weight[i][j][k] = dist(gen);
					dweight[i][j][k] = 0.0;
				}
			}
		}
		actf = new ActFunc[size[1]];
		swl.__init__(&size[2]);
	}

	void del() {
		for (i = 0; i < 2; ++i) {
			for (j = 0; j < size[i + 1]; ++j) {
				delete[] weight[i][j];
				delete[] dweight[i][j];
			}
			delete[] weight[i];
			delete[] bias[i];
			delete[] node[i];
			delete[] dweight[i];
			delete[] dbias[i];
		}
		delete[] actf;
		swl.del();
		cout << "����ɉ�����܂����iFastModel�j" << endl;
	}

	void predict(double* x) {
		_fc1(x);
		_fc2();
		swl.SoftmaxForward(node[1], node[1]);
	}

	void forward(double* x, double* t) {
		predict(x);
		swl.CrossEntropyErrorForward(t, &loss);
	}

	void backward() {
		swl.backward(dbias[1]);
		_dfc2(dbias[1]);
		_dfc1();
	}

	void save(const char* fileName = "model.kodamayuto") {
		ofstream ofs;
		ofs.open(fileName, ios::out | ios::binary | ios::trunc);
		if (!ofs) {
			cout << "�t�@�C�����J���܂���ł����B" << endl;
		}
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < size[i + 1]; ++j) {
				ofs.write((char*)&bias[i][j], sizeof(double));
				for (int k = 0; k < size[i]; ++k) {
					ofs.write((char*)&weight[i][j][k], sizeof(double));
				}
			}
		}
	}

	void load(const char* fileName = "model.kodamayuto") {
		ifstream ifs(fileName, ios::in | ios::binary);
		if (!ifs) {
			cout << "�t�@�C�����J���܂���ł����B" << endl;
		}
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < size[i + 1]; ++j) {
				ifs.read((char*)&bias[i][j], sizeof(double));
				for (int k = 0; k < size[i]; ++k) {
					ifs.read((char*)&weight[i][j][k], sizeof(double));
				}
			}
		}
	}

private:
	ActFunc* actf;
	SoftmaxWithLoss swl;
	int i, j;
	double a;

	void _fc1(double* x) {
		for (i = 0; i < size[1]; ++i) {
			a = 0.0;
			for (j = 0; j < size[0]; ++j) {
				a += x[j] * weight[0][i][j];
				dweight[0][i][j] = x[j];
			}
			a += bias[0][i];
			actf[i].forward(&a, &node[0][i]);
		}
	}
	void _fc2(void) {
		double b;
		for (i = 0; i < size[2]; ++i) {
			b = 0.0;
			for (j = 0; j < size[1]; ++j) {
				b += node[0][j] * weight[1][i][j];
				node[0][j] = 0.0;
			}
			node[1][i] += b + bias[1][i];
		}
	}

	void _dfc2(double* dout) {
		for (i = 0; i < size[2]; ++i) {
			for (j = 0; j < size[1]; ++j) {
				dweight[1][i][j] = (*dout) * node[0][j];
				node[0][j] += (*dout) * weight[1][i][j];
			}
			node[1][i] = 0.0;
		}
	}

	void _dfc1(void) {
		for (i = 0; i < size[1]; ++i) {
			actf[i].backward(&node[0][i], &dbias[0][i]);
			node[0][i] = 0.0;
			for (j = 0; j < size[0]; ++j) {
				dweight[0][i][j] *= dbias[0][i];
			}
		}
		a = 0.0;
	}
};

/****************************************************************************************
	FastSGD
	�m���I���z�~���@
	#	Net					���f���̃^�C�v��ݒ�

	FastSGD�R���X�g���N�^
		#	double lr		�w�K����ݒ肷��

	step�֐�
		���z���X�V����
		#	Net* model		���f���̃C���X�^���X�̃|�C���^
****************************************************************************************/
template<class Net> class FastSGD {
public:
	FastSGD(double _lr = 0.01) {
		lr = _lr;
	}

	void step(Net* model) {
		for (i = 0; i < 2; ++i) {
			for (j = 0; j < model->size[i + 1]; ++j) {
				model->bias[i][j] -= lr * model->dbias[i][j];
				for (k = 0; k < model->size[i]; ++k) {
					model->weight[i][j][k] -= lr * model->dweight[i][j][k];
				}
			}
		}
	}
private:
	double lr;
	int i, j, k;
};

/****************************************************************************************
	FastMomentum
	�œK���A���S���Y��Momentum
	#	Net					���f���̃^�C�v��ݒ�

	FastMomentum�R���X�g���N�^
		#	double lr		�w�K��
		#	momentum		�^���ʌW��

	step�֐�
		���z���X�V����
		#	Net* model		���f���̃C���X�^���X�̃|�C���^

****************************************************************************************/
template <class Net> class FastMomentum {
public:
	FastMomentum(double _lr = 0.01, double _momentum = 0.9) {
		lr = _lr;
		momentum = _momentum;
		i = j = k = isSecond = 0;
	}
	void step(Net* model) {
		if (isSecond == 0) {
			isSecond = 1;
			size[0] = model->size[0];
			size[1] = model->size[1];
			size[2] = model->size[2];
			for (i = 0; i < 2; ++i) {
				vW[i] = new double* [size[i + 1]];
				vb[i] = new double[size[i + 1]];
				for (j = 0; j < model->size[i + 1]; ++j) {
					vW[i][j] = new double[size[i]];
					vb[i][j] = 0.0;
					for (k = 0; k < model->size[i]; ++k) {
						vW[i][j][k] = 0.0;
					}
				}
			}
		}
		for (i = 0; i < 2; ++i) {
			for (j = 0; j < size[i + 1]; ++j) {
				vb[i][j] = momentum * vb[i][j] - lr * model->dbias[i][j];
				model->bias[i][j] += vb[i][j];
				for (k = 0; k < size[i]; ++k) {
					vW[i][j][k] = momentum * vW[i][j][k] - lr * model->dweight[i][j][k];
					model->weight[i][j][k] += vW[i][j][k];
				}
			}
		}
	}
	~FastMomentum() {
		for (i = 0; i < 2; ++i) {
			for (j = 0; j < size[i + 1]; ++j) {
				delete[] vW[i][j];
			}
			delete[] vW[i];
			delete[] vb[i];
		}
		cout << "����ɉ�����܂����iFastMomentum�j" << endl;
	}
private:
	double lr;
	double momentum;
	double** vW[2];
	double* vb[2];
	int i, j, k;
	int isSecond;
	int size[3];
};

/****************************************************************************************
	FastAdaGrad
	�œK���A���S���Y��AdaGrad
	#	Net					���f���̃^�C�v��ݒ�

	FastAdaGrad�R���X�g���N�^
		#	double lr		�w�K��
	
	step�֐�
		���z���X�V����
		#	Net* model		���f���̃C���X�^���X�̃|�C���^

****************************************************************************************/
template<class Net> class FastAdaGrad {
public:
	FastAdaGrad(double _lr = 0.01) {
		lr = _lr;
		i = j = k = isSecond = 0;
	}

	void step(Net* model) {
		if (isSecond == 0) {
			isSecond = 1;
			size[0] = model->size[0];
			size[1] = model->size[1];
			size[2] = model->size[2];

			for (i = 0; i < 2; ++i) {
				hW[i] = new double* [size[i + 1]];
				hb[i] = new double[size[i + 1]];
				for (j = 0; j < size[i + 1]; ++j) {
					hW[i][j] = new double[size[i]];
					hb[i][j] = 0.0;
					for (k = 0; k < size[i]; ++k) {
						hW[i][j][k] = 0.0;
					}
				}
			}
		}
		for (i = 0; i < 2; ++i) {
			for (j = 0; j < size[i + 1]; ++j) {
				hb[i][j] += model->dbias[i][j] * model->dbias[i][j];
				model->bias[i][j] -= lr / (sqrt(hb[i][j]) + Delta) * model->dbias[i][j];
				for (k = 0; k < size[i]; ++k) {
					hW[i][j][k] += model->dweight[i][j][k] * model->dweight[i][j][k];
					model->weight[i][j][k] -= lr / (sqrt(hW[i][j][k]) + Delta) * model->dweight[i][j][k];
				}
			}
		}
	}
	~FastAdaGrad() {
		for (i = 0; i < 2; ++i) {
			for (j = 0; j < size[i + 1]; ++j) {
				delete[] hW[i][j];
			}
			delete[] hW[i];
			delete[] hb[i];
		}
		cout << "����ɉ�����܂����iFastAdaGrad�j" << endl;
	}
private:
	double lr;
	double** hW[2];
	double* hb[2];
	int i, j, k;
	int isSecond;
	int size[3];
};

/****************************************************************************************
	FastAdam
	�œK���A���S���Y��Adam
	��悷��v���O����������̂ŁA�I�[�o�t���[����\��������B
	#	Net					���f���̃^�C�v��ݒ�

	FastAdam�R���X�g���N�^
		#	double _lr		�w�K��
		#	double _beta1	�n�C�p�[�p�����[�^�P
		#	double _beta2	�n�C�p�[�p�����[�^�Q

	step�֐�
		���z���X�V����
		#	Net* model		���f���̃C���X�^���X�̃|�C���^

****************************************************************************************/
template<class Net> class FastAdam {
public:
	FastAdam(double _lr = 0.001, double _beta1 = 0.9, double _beta2 = 0.999) {
		lr = _lr;
		beta1 = _beta1;
		beta2 = _beta2;
		isSecond = 0;
		lr_t = 0.0;
		i = j = k = 0;
	}
	void step(Net* model) {
		if (isSecond == 0) {
			isSecond = 1;
			size[0] = model->size[0];
			size[1] = model->size[1];
			size[2] = model->size[2];
			for (i = 0; i < 2; ++i) {
				mW[i] = new double* [model->size[i + 1]];
				vW[i] = new double* [model->size[i + 1]];
				mb[i] = new double[model->size[i + 1]];
				vb[i] = new double[model->size[i + 1]];
				for (j = 0; j < model->size[i + 1]; ++j) {
					mW[i][j] = new double[model->size[i]];
					vW[i][j] = new double[model->size[i]];
					mb[i][j] = 0.0;
					vb[i][j] = 0.0;
					for (k = 0; k < model->size[i]; ++k) {
						mW[i][j][k] = 0.0;
						vW[i][j][k] = 0.0;
					}
				}
			}
		}
		iter += 1.0;
		lr_t = lr * sqrt(1.0 - pow(beta2, iter)) / (1.0 - pow(beta1, iter));
		for (i = 0; i < 2; ++i) {
			for (j = 0; j < model->size[i + 1]; ++j) {
				mb[i][j] += (1 - beta1) * (model->dbias[i][j] - mb[i][j]);
				vb[i][j] += (1 - beta2) * ((model->dbias[i][j]) * (model->dbias[i][j]) - vb[i][j]);
				model->bias[i][j] -= lr_t * mb[i][j] / (sqrt(vb[i][j]) + Delta);
				for (k = 0; k < model->size[i]; ++k) {
					mW[i][j][k] += (1 - beta1) * (model->dweight[i][j][k] - mW[i][j][k]);
					vW[i][j][k] += (1 - beta2) * ((model->dweight[i][j][k]) * (model->dweight[i][j][k]) - vW[i][j][k]);
					model->weight[i][j][k] -= lr_t * mW[i][j][k] / (sqrt(vW[i][j][k]) + Delta);
				}
			}
		}
	}
	~FastAdam() {
		for (i = 0; i < 2; ++i) {
			for (j = 0; j < size[i + 1]; ++j) {
				delete[] mW[i][j];
				delete[] vW[i][j];
			}
			delete[] mW[i];
			delete[] vW[i];
			delete[] mb[i];
			delete[] vb[i];
		}
		cout << "����ɉ�����܂����iFastAdam�j" << endl;
	}
private:
	int i, j, k;
	double lr;
	double beta1;
	double beta2;
	int isSecond;
	double** mW[2];
	double* mb[2];
	double** vW[2];
	double* vb[2];
	double lr_t;
	double iter = 0.0;
	int size[3];
};

#endif // ! _FastSimpleNeuralNetwork_HPP_