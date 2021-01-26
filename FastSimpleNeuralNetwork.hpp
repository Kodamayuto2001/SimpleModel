#ifndef _FastSimpleNeuralNetwork_HPP_
#define _FastSimpleNeuralNetwork_HPP_


template<class ActFunc,class SoftmaxWithLoss>
class FastModel {
public:
    /*
	 *	FastMul‚Í•s•K—v
	 *	Forward
	 *	node		= x * weight(bias)
	 *
	 *	Backward
	 *	node		= dout * weight(bias)
	 *	weight(bias)= dout * x
	 */
	class Forward {
	public:
		double** weight;
		double* bias;
		double* node;	//	ƒ“`”À‹t“`”ÀŒ“—p
	};
	class Backward {
		double** dweight;
		double* dbias;
	};
	FastModel(
		
	)
};

#endif // ! _FastSimpleNeuralNetwork_HPP_
