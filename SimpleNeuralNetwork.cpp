#include "SimpleNeuralNetwork.hpp"

void train() {
	SimpleNet<ReLU, Softmax, CrossEntropyError> model(2, 3, 2);
	SGD<SimpleNet<ReLU, Softmax, CrossEntropyError>> optimizer(0.01);
	
	double x[2] = { 2.35,4.97 };	// ���t�f�[�^
	double t[2] = { 1,0 };			// �������x��

	// �w�K��
	int epoch = 1000;

	double loss;
	for (int e = 0; e < epoch; e++) {
		// ���`�d
		loss = model.forward(x, t);

		// �t�`�d
		model.backward(1.0);

		// �X�V
		optimizer.step(model);

		// �����l
		printf("loss = %lf\n", loss);
	}

	// �w�K�������f���̃p�����[�^��ۑ�
	model.save("model.txt");

	// �������̉��
	model.del();
}

void test() {
	SimpleNet<ReLU, Softmax, CrossEntropyError> ai(2, 3, 2);

	// �ۑ��������f�������[�h
	ai.load("model.txt");

	// ���؃f�[�^
	double x[2] = { 2.5,5.0 };

	// ���`��
	double* y = ai.predict(x);

	// �\���l
	printf("y[0] = %lf ��\n", y[0] * 100);

	// �������̊J��
	ai.del();
}

int main() {
	train();
	test();
	return 0;
}