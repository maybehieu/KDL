#include "nn.h"

#include "utils.h"

NeuralNet::NeuralNet(size_t in_channel, size_t out_channel, int num_hidden_layers, double lr, int epoch, int batch_size, std::string activation, std::string optimizer)
{
	// create weight matrices
	weights.push_back(Matrix(100, in_channel));
	for (int i = 0; i < num_hidden_layers - 1; i++)
		weights.push_back(Matrix(100, 100));
	weights.push_back(Matrix(out_channel, in_channel));

	// create bias matrices
	for (int i = 0; i < num_hidden_layers; i++)
		biases.push_back(Matrix(100, 1));
	biases.push_back(Matrix(out_channel, 1));

	// initialize weights with random values
	std::for_each(weights.begin(), weights.end(), [](Matrix& m) {return m.randomize(-1, 1); });

	eta = lr;
	epochs = epoch;
	this->batch_size = batch_size;
	this->activation = activation;
	if (optimizer == "")
	{
		this->optimizer = Optimizer();
	}
	
}

NeuralNet::NeuralNet(std::vector<int> detailed_layers, double lr, int epoch, int batch_size, std::string activation, std::string optimizer)
{
	for (int i = 1; i < detailed_layers.size(); i++)
	{
		// create weight matrices
		weights.push_back(Matrix(detailed_layers[i], detailed_layers[i-1]));
		
		// create bias matrices
		biases.push_back(Matrix(detailed_layers[i], 1));
	}
	
	// initialize weights with random values
	std::for_each(weights.begin(), weights.end(), [](Matrix& m) {return m.randomize(-1, 1); });

	eta = lr;
	epochs = epoch;
	this->batch_size = batch_size;
	this->activation = activation;
	if (optimizer == "")
	{
		this->optimizer = Optimizer();
	}
}

net_result NeuralNet::forward(const Matrix& X)
{
	Matrix input(X);
	Matrix mul;
	Matrix layer;
	Matrix activ;
	Matrix yhat;

	std::vector<Matrix> inputs;
	std::vector<Matrix> layers;
	std::vector<Matrix> activations;

	/// feed forward process:
	/// 


	// forward through each 'hidden layer'
	for (int i = 0; i < weights.size() - 1; i++)
	{
		inputs.push_back(input);

		mul = weights[i] * input;
		layer = mul + biases[i].broadcast(mul);
		// next_input = last_activ;
		input = Matrix::apply_func(layer, activation == "relu" ? relu : sigmoid);

		//layers.push_back(layer);
		activations.push_back(input);
	}
	// forward for last/output layer
	mul = weights.back() * input;
	layer = mul + biases.back().broadcast(mul);
	yhat = softmax(layer);

	//inputs.push_back(input);
	//layers.push_back(layer);
	//activations.push_back(yhat);


	return net_result(inputs, layers, activations, yhat);
}

net_result NeuralNet::backward(const net_result& datas, const Matrix& X, const Matrix& y)
{
	Matrix yhat = datas.yhat;
	std::vector<Matrix> inputs = datas.inputs;
	std::vector<Matrix> layers = datas.layers;
	std::vector<Matrix> activs = datas.activations;

	std::vector<Matrix> d_activs;
	std::vector<Matrix> d_weights;
	std::vector<Matrix> d_biases;

	// error with normalization
	Matrix d_activ = (yhat - y) / y.get_width();

	// last layer of net
	Matrix weight = activs.back();
	Matrix activ;
	Matrix input;

	Matrix d_weight = d_activ * Matrix::transpose(weight);
	Matrix d_bias = d_activ.sum(1);
	Matrix prev_d_activ = Matrix::transpose(weights.back()) * d_activ;

	d_weights.push_back(d_weight);		// W(n)
	d_biases.push_back(d_bias);			// B(n)
	d_activs.push_back(prev_d_activ);	// A(n-1)

	// remaining layers in backward order
	for (int i = weights.size() - 2; i >= 0; --i)
	{
		// layer-Z, activation-A
		// note: non-derivative weight/layer and bias
		input = inputs[i];
		weight = weights[i];
		activ = activs[i];

		d_activ = d_activs.back();
		d_activ = d_activ.element_wise_mul(Matrix::apply_func(activ, activation == "relu" ? d_relu : d_sigmoid));

		// calculate derivative of weight
		d_weight = d_activ * Matrix::transpose(input);
		// calculate derivative of bias
		d_bias = d_activ.sum(1);
		// calculate derivative of activation
		prev_d_activ = Matrix::transpose(weight) * d_activ;

		d_weights.push_back(d_weight);
		d_biases.push_back(d_bias);
		d_activs.push_back(prev_d_activ);
	}

	return net_result(d_activs, d_weights, d_biases);
}

void NeuralNet::fit(const Matrix& X_in, const Matrix& y_in)
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

			net_result result = forward(X);

			double batch_loss = cross_entropy_loss(y_batch, result.yhat).m_data[0];
			batch_losses.push_back(batch_loss);

			result = backward(result, X, y);
		}

		epoch_losses.push_back(std::accumulate(batch_losses.begin(), batch_losses.end(), .0) / batch_losses.size());
		if (epoch % 50 == 0)
		{
			printf("Epoch %d/%d: training loss: %f, time taken: %f s\n", epoch, epochs, epoch_losses.back(), (getTickcount() - e_time) / 1000.0);
		}
	}
	printf("Training complete! Time taken: %f s\n", (getTickcount() - time) / 1000.0);
}

Matrix NeuralNet::predict(const Matrix& X)
{
	return Matrix();
}

void NeuralNet::eval(const Matrix& X, const Matrix& y)
{
	
}

void NeuralNet::test()
{
	// X co 50 hang, 16 features
	Matrix X(16, 50);
	// y co 50 hang, 26 output
	Matrix y(26, 50);

	/*X.randomize(-1,1);
	y.randomize(1, 2);*/

	X.load_data_txt(16, 50, R"(../data/test_nn/x.txt)");
	y.load_data_txt(26, 50, R"(../data/test_nn/y.txt)");
	for (int i = 0; i < weights.size(); i++)
	{
		std::cout << "loading weight: " << R"(../data/test_nn/W)" + std::to_string(i + 1) + ".txt" << "\n";
		weights[i].load_data_txt(weights[i].get_height(), weights[i].get_width(), R"(../data/test_nn/W)" + std::to_string(i+1) + ".txt");
	}

	net_result result = forward(X);



	result = backward(result, X, y);
	return;
}

void NeuralNet::load_model(const std::string& path)
{
	
}

void NeuralNet::save_model(const std::string& path)
{
	
}
