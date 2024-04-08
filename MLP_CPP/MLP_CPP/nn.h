#pragma once
#include "matrix.h"

struct net_result
{
	std::vector<Matrix> inputs;
	std::vector<Matrix> layers;
	std::vector<Matrix> activations;
	Matrix yhat;

	std::vector<Matrix> grad_activations;
	std::vector<Matrix> grad_weights;
	std::vector<Matrix> grad_biases;

	net_result(std::vector<Matrix>& inputs, std::vector<Matrix>& layers, std::vector<Matrix>& activs, Matrix& yHat)
	{
		this->inputs = std::move(inputs);
		this->layers = std::move(layers);
		this->activations = std::move(activs);
		this->yhat = std::move(yHat);
	}

	net_result(std::vector<Matrix>& g_a, std::vector<Matrix>& g_w, std::vector<Matrix>& g_b)
	{
		this->grad_activations = std::move(g_a);
		this->grad_weights = std::move(g_w);
		this->grad_biases = std::move(g_b);
	}
};

class Optimizer
{
	std::string name;
	float learning_rate;



	void optimizer_step();

	void sgd_step();
	void adam_step();

public:
	Optimizer(std::string name);

	Optimizer();
};

class NeuralNet
{
	// weights of layers
	std::vector<Matrix> weights;
	// biases of each weight
	std::vector<Matrix> biases;
	// learning rate
	double eta;
	// epochs
	int epochs;
	int batch_size;
	// activation
	std::string activation;
	// optimizer
	Optimizer optimizer;

public:
	// use default size 100 for every hidden layer
	NeuralNet(size_t in_channel, size_t out_channel, int num_hidden_layers, 
		double lr, int epoch, int batch_size, std::string activation, std::string optimizer);

	// vector contains size for input, each hidden layer, output / model topology
	NeuralNet(std::vector<int> detailed_layers, 
		double lr, int epoch, int batch_size, std::string activation, std::string optimizer);

	void fit(const Matrix& X, const Matrix& y);
	Matrix predict(const Matrix& X);
	net_result forward(const Matrix& inputs);
	net_result backward(const net_result& datas, const Matrix& X, const Matrix& y);

	void eval(const Matrix& X, const Matrix& y);
	void test();

	void load_model(const std::string& path);
	void save_model(const std::string& path);
};