#include "SimpleNeuralNetwork.hpp"

void test() {
	ReLU relu;
	cout << relu.forward(-2.0) << endl;
	cout << relu.backward(1.0) << endl;
}

void test1() {
	SimpleNet<ReLU,Softmax,CrossEntropyError> model(2, 3, 2);

	model.del();
}

int main() {
	test1();
	return 0;
}
