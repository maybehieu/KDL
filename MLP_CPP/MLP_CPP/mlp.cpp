#include "mlp.h"

// activations
double relu(double x)
{
	if (x <= 0.0) return 0.0;
	return x;
}
double d_relu(double x)
{
	if (x <= 0) return 0.0;
	return 1.0;
}
double sigmoid(double x)
{
	return 1. / (1. + std::exp(-1 * x));
}
double d_sigmoid(double x)
{
	return x * (1 - x);
}


MLP::MLP(size_t in_channel, size_t hidden_channel, size_t out_channel, double lr, int epoch, std::string activation)
{
	W1 = Matrix(in_channel, hidden_channel);
	b1 = Matrix(hidden_channel, 1);
	W2 = Matrix(hidden_channel, out_channel);
	b2 = Matrix(out_channel, 1);

	// initialize weights with random
	W1.randomize(-1, 1);
	W2.randomize(-1, 1);

	eta = lr;
	epochs = epoch;
	this->activation = activation;
}

std::vector<Matrix> MLP::forward(const Matrix& inputs)
{
	Matrix mul = (Matrix::transpose(W1) * inputs);
	Matrix layer1 = mul + b1.broadcast(mul);
	Matrix activation1 = Matrix::apply_func(layer1, activation == "relu" ? relu : sigmoid);
	mul = (Matrix::transpose(W2) * activation1);
	Matrix layer2 = mul + b2.broadcast(mul);
	Matrix yhat = softmax(layer2);

	std::vector<Matrix> outputs = {yhat, layer2, activation1, layer1};
	return outputs;
}

void MLP::backward(std::vector<Matrix> datas, const Matrix& X, const Matrix& y)
{
	Matrix yhat = datas[0];
	Matrix l2 = datas[1];
	Matrix A1 = datas[2];
	Matrix l1 = datas[3];

	// back propagation
	Matrix E2 = (yhat - y) / X.get_width();
	Matrix d_weight2 = A1 * Matrix::transpose(E2);
	Matrix d_bias2 = E2.sum(1);
	Matrix E1 = W2 * E2;

	for (size_t i = 0; i < E1.get_height(); i++)
		for (size_t j = 0; j < E1.get_width(); j++)
			E1(i, j) = l1(i, j) > 0 ? E1(i, j) : 0;

	Matrix d_weight1 = X * Matrix::transpose(E1);
	Matrix d_bias1 = E1.sum(1);

	// update to weights
	W1 += -eta * d_weight1;
	b1 += -eta * d_bias1;
	W2 += -eta * d_weight2;
	b2 += -eta * d_bias2;
}

Matrix MLP::softmax(const Matrix& matrix)
{
	Matrix max = matrix.max();
	Matrix _V = Matrix::mismatch_dim_subtract(matrix, max);
	Matrix e_V = Matrix::apply_func(_V, [](double x) {return std::exp(x); });
	Matrix sum = e_V.sum(0);
	Matrix Z = Matrix::mismatch_dim_divide(e_V, sum);
	return Z;
}

Matrix MLP::cross_entropy_loss(const Matrix& y, const Matrix& yhat)
{
	Matrix log_m = Matrix::apply_func(yhat, [](double x) {return std::log(x); });
	Matrix mul = Matrix::element_wise_mul(y, log_m);
	Matrix sum = mul.sum(-1);
	return (0. - sum) / yhat.get_width();
}

void MLP::test()
{
	// X co 50 hang, 16 features
	Matrix X(50, 16);
	// y co 50 hang, 26 output
	Matrix y(50, 26);

	/*X.randomize(-1,1);
	y.randomize(1, 2);*/

	X.load_data_str(16, 50, R"(F:\Documents\Code\Letter-Recognition-Using-Multi-layer-Perceptron-master\xxbatch.txt)");
	y.load_data_str(26, 50, R"(F:\Documents\Code\Letter-Recognition-Using-Multi-layer-Perceptron-master\xybatch.txt)");
	W1.load_data_str(16, 100,R"(F:\Documents\Code\Letter-Recognition-Using-Multi-layer-Perceptron-master\xw1.txt)");
	W2.load_data_str(100, 26, R"(F:\Documents\Code\Letter-Recognition-Using-Multi-layer-Perceptron-master\xw2.txt)");

	// forward
	std::vector<Matrix> net_outs = forward(X);

	// cal loss
	double loss = cross_entropy_loss(y, net_outs[0]).m_data[0];

	// back prop
	backward(net_outs, X, y);
}
