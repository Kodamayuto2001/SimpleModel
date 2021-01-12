#include "SimpleNeuralNetwork.hpp"

void train() {
	// ���f���̃C���X�^���X���i���͑w�A���ԑw�A�o�͑w�j
	SimpleNet model(2, 3, 2);

	// �œK���A���S���Y���̃C���X�^���X���i�w�K���j
	SGD optimizer(0.01);

	double x[2] = { 2.35,4.97 };	// ���t�f�[�^
	double t[2] = { 1,0 };			// �������x��

	// �w�K��
	int epoch = 1000;

	double loss;
	for (int e = 0; e < epoch; e++) {
		// ���`��
		loss = model.Loss(x, t);

		// �t�`��
		model.backward();

		// �X�V
		optimizer.step(model);

		// �����l
		printf("loss = %lf\n", loss);
	}

	// �w�K�������f���̃p�����[�^��ۑ�
	model.save("model.txt");

	// �������̊J��
	model.del();
}

void test() {
	// ���f���̃C���X�^���X��
	SimpleNet model(2, 3, 2);

	// �ۑ��������f�������[�h
	model.load("model.txt");

	// ���؃f�[�^
	double x[2] = { 2.5,5.0 };

	// ���`��
	double* y = model.predict(x);
	
	// �\���l
	printf("y[0] = %lf ��\n",y[0]*100);
	
	// �������̊J��
	model.del();
}

int main(void) {
	train();
	test();
	return 0;
}