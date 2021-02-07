#include "dataset.hpp"
#define DATAMAX		1000
#define CHANNEL		1
#define IMG_HEIGHT	160
#define IMG_WIDTH	160
#define INPUT_SIZE  CHANNEL*IMG_HEIGHT*IMG_WIDTH
#define HIDDEN_SIZE 320
#define OUTPUT_SIZE 29
#include "FastSimpleNeuralNetwork.h"
#include "dataloader.hpp"

double x[DATAMAX][INPUT_SIZE];

enum {
	ando,
	enomaru,
	hamada,
	higashi,
	kataoka,
	kawano,
	kodama,
	masuda,
	matsuzaki,
	matui,
	miyatake,
	mizuki,
	nagao,
	okamura,
	ooshima,
	ryuuga,
	shinohara,
	soushi,
	suetomo,
	takemoto,
	tamejima,
	teppei,
	toriyabe,
	tsuchiyama,
	uemura,
	wada,
	watanabe,
	yamaji,
	yamashita,
	Denjyo2_MAX
};


int main(void) {
	//	�w�K�p�f�[�^�Z�b�g���ۑ�����Ă���path���w��
	char* path[Denjyo2_MAX] = {
		"29classes_dataset-main/train_data29/ando/",
		"29classes_dataset-main/train_data29/enomaru/",
		"29classes_dataset-main/train_data29/hamada/",
		"29classes_dataset-main/train_data29/higashi/",
		"29classes_dataset-main/train_data29/kataoka/",
		"29classes_dataset-main/train_data29/kawano/",
		"29classes_dataset-main/train_data29/kodama/",
		"29classes_dataset-main/train_data29/masuda/",
		"29classes_dataset-main/train_data29/matsuzaki/",
		"29classes_dataset-main/train_data29/matui/",
		"29classes_dataset-main/train_data29/miyatake/",
		"29classes_dataset-main/train_data29/mizuki/",
		"29classes_dataset-main/train_data29/nagao/",
		"29classes_dataset-main/train_data29/okamura/",
		"29classes_dataset-main/train_data29/ooshima/",
		"29classes_dataset-main/train_data29/ryuuga/",
		"29classes_dataset-main/train_data29/shinohara/",
		"29classes_dataset-main/train_data29/soushi/",
		"29classes_dataset-main/train_data29/suetomo/",
		"29classes_dataset-main/train_data29/takemoto/",
		"29classes_dataset-main/train_data29/tamejima/",
		"29classes_dataset-main/train_data29/teppei/",
		"29classes_dataset-main/train_data29/toriyabe/",
		"29classes_dataset-main/train_data29/tsuchiyama/",
		"29classes_dataset-main/train_data29/uemura/",
		"29classes_dataset-main/train_data29/wada/",
		"29classes_dataset-main/train_data29/watanabe/",
		"29classes_dataset-main/train_data29/yamaji/",
		"29classes_dataset-main/train_data29/yamashita/",
	};

	//	�����l�i�[�p�ϐ�
	double loss;
	//	���x���i�[�p�ϐ�
	double t[OUTPUT_SIZE];
	//	���f���̏�����
	SimpleNeuralNetwork_init();

	for (int i = 0; i < Denjyo2_MAX; ++i) {
		//	���t�f�[�^
		dataloader(path[i], x);
		//	�������x��
		Flatten(i, t);
		for (int j = 0; j < DATAMAX; ++j) {
			//	���`�d�t�`�d
			SimpleNeuralNetwork(
				Sigmoid_forward,
				Softmax_forward,
				CrossEntropyError_forward,
				SoftmaxWithLoss_backward,
				Sigmoid_backward,
				x[j],
				t,
				&loss
			);
			//	�œK��
			Adam();
			//	�\��
			printf("\r[%2d] [%3d] [%10f]", i, j, loss);
		}
		printf("\n");
	}

	//	���f���̕ۑ�
	save("test.model");
	
	return 0;
}