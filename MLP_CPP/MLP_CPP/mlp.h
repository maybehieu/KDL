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
	int batch_size;
	// activation
	std::string activation;

public:
	MLP(size_t in_channel, size_t hidden_channel, size_t out_channel,
		double lr, int epoch, int batch_size, std::string activation);

	void fit(const Matrix& X, const Matrix& y);
	Matrix predict(const Matrix& X);
	std::vector<Matrix> forward(const Matrix& inputs);
	void backward(const std::vector<Matrix>& datas, const Matrix& X, const Matrix& y);

	void eval(const Matrix& X, const Matrix& y);
	void test();

	void load_model(const std::string& path);
	void save_model(const std::string& path);

	Matrix softmax(const Matrix& matrix);
	Matrix cross_entropy_loss(const Matrix& y, const Matrix& yhat);
};