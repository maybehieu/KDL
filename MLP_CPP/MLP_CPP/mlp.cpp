#include "mlp.h"
#include "utils.h"
#include <filesystem>

MLP::MLP(size_t in_channel, size_t hidden_channel, size_t out_channel, double lr, int epoch, int batch_size, std::string activation)
{
	W1 = Matrix(in_channel, hidden_channel);
	b1 = Matrix(hidden_channel, 1);
	W2 = Matrix(hidden_channel, out_channel);
	b2 = Matrix(out_channel, 1);

	// initialize weights with random
	W1.randomize(0, 1, .01);
	W2.randomize(0, 1, .01);

	eta = lr;
	epochs = epoch;
	this->batch_size = batch_size;
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

void MLP::backward(const std::vector<Matrix>& datas, const Matrix& X, const Matrix& y)
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

void MLP::fit(const Matrix& X_in, const Matrix& y_in)
{
	Matrix X(X_in);
	Matrix y(y_in);
	double N = X.get_width();
	std::vector<double> epoch_losses;

	std::cout << "Training started!\n";
	int64_t time = getTickcount();
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		int64_t e_time = getTickcount();
		std::vector<double> batch_losses;

		Matrix::shuffle(X, y);
		for (int q = 0; q < N; q += batch_size)
		{
			Matrix X_batch = X.extract(q, q + batch_size - 1, 0);
			Matrix y_batch = y.extract(q, q + batch_size - 1, 0);

			X_batch.transpose();
			y_batch.transpose();

			std::vector<Matrix> outs = forward(X_batch);

			double batch_loss = cross_entropy_loss(y_batch, outs[0]).m_data[0];
			batch_losses.push_back(batch_loss);

			backward(outs, X_batch, y_batch);
		}

		epoch_losses.push_back(std::accumulate(batch_losses.begin(), batch_losses.end(), .0) / batch_losses.size());
		if (epoch % 50 == 0)
		{
			double train_accuracy = eval(X, y);
			printf("Epoch %d/%d: training loss: %f, training accuracy: %f, time taken: %f s\n", epoch, epochs, epoch_losses.back(), train_accuracy, (getTickcount() - e_time) / 1000.0);
		}
	}
	printf("Training complete! Time taken: %f s\n", (getTickcount() - time) / 1000.0);
}

void MLP::print_eval(const Matrix& X_in, const Matrix& y_in)
{
	Matrix X(X_in);
	Matrix y(y_in);
	double N = X.get_width();

	Matrix y_pred = predict(X);
	y.transpose();
	y = y.argmax(1);
	
#ifdef _DEBUG
	if (y_pred.m_data.size() != y.m_data.size())
	{
		throw MatrixError("Prediction output doesn't match ground truth shape!");
	}
#endif

	// accuracy
	std::vector<float> acc;
	for (size_t i = 0; i < y_pred.m_data.size(); i++)
		acc.push_back(y_pred.m_data[i] == y.m_data[i] ? 1 : 0);
	printf("Model accuracy: %f\n", std::accumulate(acc.begin(), acc.end(), .0) / acc.size());
}

double MLP::eval(const Matrix& X_in, const Matrix& y_in)
{
	Matrix X(X_in);
	Matrix y(y_in);
	double N = X.get_width();

	Matrix y_pred = predict(X);
	y.transpose();
	y = y.argmax(1);

#ifdef _DEBUG
	if (y_pred.m_data.size() != y.m_data.size())
	{
		throw MatrixError("Prediction output doesn't match ground truth shape!");
	}
#endif

	// accuracy
	std::vector<float> acc;
	for (size_t i = 0; i < y_pred.m_data.size(); i++)
		acc.push_back(y_pred.m_data[i] == y.m_data[i] ? 1 : 0);
	return std::accumulate(acc.begin(), acc.end(), .0) / acc.size();
}

Matrix MLP::predict(const Matrix& X_in)
{
	Matrix X(X_in);
	double N = X.get_width();

	X.transpose();

	Matrix mul = (Matrix::transpose(W1) * X);
	Matrix layer1 = mul + b1.broadcast(mul);
	Matrix activation1 = Matrix::apply_func(layer1, activation == "relu" ? relu : sigmoid);
	mul = (Matrix::transpose(W2) * activation1);
	Matrix layer2 = mul + b2.broadcast(mul);

	return layer2.argmax(1);
}

void MLP::test()
{
	// X co 50 hang, 16 features
	Matrix X(50, 16);
	// y co 50 hang, 26 output
	Matrix y(50, 26);

	/*X.randomize(-1,1);
	y.randomize(1, 2);*/

	X.load_data_txt(16, 50, R"(../data/test/xxbatch.txt)");
	y.load_data_txt(26, 50, R"(../data/test/xybatch.txt)");
	W1.load_data_txt(16, 100,R"(../data/test/xw1.txt)");
	W2.load_data_txt(100, 26, R"(../data/test/xw2.txt)");

	// forward
	std::vector<Matrix> net_outs = forward(X);

	// cal loss
	double loss = cross_entropy_loss(y, net_outs[0]).m_data[0];

	// back prop
	backward(net_outs, X, y);
}

void MLP::load_model(const std::string& path)
{
	W1.load_data(path + "W1.bin");
	b1.load_data(path + "b1.bin");
	W2.load_data(path + "W2.bin");
	b2.load_data(path + "b2.bin");
}

void MLP::save_model(const std::string& path)
{
	if (!std::filesystem::is_directory(path))
	{
		std::filesystem::create_directories(path);
		std::cout << "Created weights folder!\n";
	}
	W1.save_data(path + "W1.bin");
	b1.save_data(path + "b1.bin");
	W2.save_data(path + "W2.bin");
	b2.save_data(path + "b2.bin");
}
