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

struct parameters
{
	// weights of layers
	std::vector<Matrix> weights;
	// biases of each weight
	std::vector<Matrix> biases;

	parameters() = default;

	parameters(const std::vector<Matrix>& weights, const std::vector<Matrix>& biases)
	{
		this->weights = weights;
		this->biases = biases;
	}

	parameters(const parameters& other)
	{
		weights = other.weights;
		biases = other.biases;
	}
};

class Optimizer
{
	std::string name;
	double learning_rate;
	double batch_size = 50;

	// adam
	int cnt;
	std::vector<Matrix> v_w;	// first moment of gradient of weights
	std::vector<Matrix> s_w;	// second moment of gradient of weights
	std::vector<Matrix> v_b;	// first moment of gradient of biases
	std::vector<Matrix> s_b;	// second moment of gradient of biases
	double beta1 = .9;
	double beta2 = .999;
	double epsilon = 1e-8;


	void sgd_step(parameters& params, const net_result& grads);
	void adam_step(parameters& params, const net_result& grads);

public:
	Optimizer() = default;

	// sgd
	Optimizer(double lr, double batch_size);

	// adam
	Optimizer(const std::vector<Matrix>& weights, const std::vector<Matrix>& biases, 
		const double& lr, const double& batch_sz, const double& b1, const double& b2, const double& e );

	void step(parameters& params, const net_result& grads);
};

class NeuralNet
{
	//// weights of layers
	//std::vector<Matrix> weights;
	//// biases of each weight
	//std::vector<Matrix> biases;
	parameters parameters;
	// learning rate
	double eta;
	// epochs
	int epochs;
	int batch_size;
	// activation for FC
	std::string activation;
	// loss function
	std::string loss_fn;
	// optimizer
	Optimizer optimizer;

public:
	// use default size 100 for every hidden layer
	NeuralNet(size_t in_channel, size_t out_channel, int num_hidden_layers, 
		double lr, int epoch, int batch_size, std::string activation, std::string loss, std::string optimizer);

	// vector contains size for input, each hidden layer, output / model topology
	NeuralNet(std::vector<int> detailed_layers, 
		double lr, int epoch, int batch_size, std::string activation, std::string loss, std::string optimizer);

	void fit(const Matrix& X, const Matrix& y);
	Matrix predict(const Matrix& X);
	net_result forward(const Matrix& inputs, bool activated);
	net_result backward(const net_result& datas, const Matrix& X, const Matrix& y);

	Matrix layer_forward();
	Matrix layer_backward();

	Matrix activation_forward();
	Matrix activation_backward();


	void print_eval(const Matrix& X, const Matrix& y);
	double eval(const Matrix& X, const Matrix& y);
	void test();
	void simple_test();

	void load_model(const std::string& path);
	void save_model(const std::string& path);
};