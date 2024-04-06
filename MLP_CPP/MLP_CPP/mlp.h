#pragma once
#include "matrix.h"

class MLP
{
	// weight1
	Matrix W1;
	// bias1
	Matrix b1;
	// weight2
	Matrix W2;
	// bias2
	Matrix b2;
	// learning rate
	double eta;
	// epochs
	int epochs;
	// activation
	std::string activation;

public:
	MLP(size_t in_channel, size_t hidden_channel, size_t out_channel,
		double lr, int epoch, std::string activation);

	void training();
	void eval();
	void test();
	std::vector<Matrix> forward(const Matrix& inputs);
	void backward(std::vector<Matrix> datas, const Matrix& X, const Matrix& y);

	Matrix softmax(const Matrix& matrix);
	Matrix cross_entropy_loss(const Matrix& y, const Matrix& yhat);
};