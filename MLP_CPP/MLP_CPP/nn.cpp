#include "nn.h"

#include "utils.h"

Optimizer::Optimizer(double lr)
{
	name = "sgd";
	learning_rate = lr;
}

Optimizer::Optimizer(const std::vector<Matrix>& weights, const std::vector<Matrix>& biases, const double& lr, const double& b1 = .9, const double& b2 = .999, const double& e = 1e-8)
{
	name = "adam";
	for (const Matrix& w : weights)
	{
		v_w.push_back(Matrix(w.get_height(), w.get_width()));
		s_w.push_back(Matrix(w.get_height(), w.get_width()));
	}

	for (const Matrix& b : biases)
	{
		v_b.push_back(Matrix(b.get_height(), b.get_width()));
		s_b.push_back(Matrix(b.get_height(), b.get_width()));
	}

	cnt = 0;
	learning_rate = lr;
	beta1 = b1;
	beta2 = b2;
	epsilon = e;
}

void Optimizer::sgd_step(parameters& params, const net_result& grads)
{
	parameters result(params);
	int N = params.weights.size();
	for (int i = 0; i < N; i++)
	{
		result.weights[i] = params.weights[i] - learning_rate * grads.grad_weights[N - 1 - i];
		result.biases[i] = params.biases[i] - learning_rate * grads.grad_biases[N - 1 - i];
	}
	params = result;
}

void Optimizer::adam_step(parameters& params, const net_result& grads)
{
	cnt++;	// adam counter
	std::vector<Matrix> v_w_corrected;
	std::vector<Matrix> s_w_corrected;
	std::vector<Matrix> v_b_corrected;
	std::vector<Matrix> s_b_corrected;

	parameters result(params);
	int N = params.weights.size();
	for (int i = 0; i < N; i++)
	{
		v_w[i] = beta1 * v_w[i] + (1 - beta1) * grads.grad_weights[N - 1 - i];
		v_b[i] = beta1 * v_b[i] + (1 - beta1) * grads.grad_biases[N - 1 - i];
		v_w_corrected.push_back(v_w[i] / (1 - std::pow(beta1, cnt)));
		v_b_corrected.push_back(v_b[i] / (1 - std::pow(beta1, cnt)));

		s_w[i] = beta2 * s_w[i] + (1 - beta2) * Matrix::power(grads.grad_weights[N - 1 - i], 2);
		s_b[i] = beta2 * s_b[i] + (1 - beta2) * Matrix::power(grads.grad_biases[N - 1 - i], 2);
		s_w_corrected.push_back(s_w[i] / (1 - std::pow(beta2, cnt)));
		s_b_corrected.push_back(s_b[i] / (1 - std::pow(beta2, cnt)));

		Matrix vw = v_w_corrected[i] / Matrix::sqrt(s_w_corrected[i] + epsilon);
		Matrix vb = v_b_corrected[i] / Matrix::sqrt(s_b_corrected[i] + epsilon);
		result.weights[i] = params.weights[i] - learning_rate * vw;
		result.biases[i] = params.biases[i] - learning_rate * vb;
	}
	params = result;
}

void Optimizer::step(parameters& params, const net_result& grads)
{
	if (name == "adam")
		adam_step(params, grads);
	if (name == "sgd")
		sgd_step(params, grads);
}

NeuralNet::NeuralNet(size_t in_channel, size_t out_channel, int num_hidden_layers, double lr, int epoch, int batch_size, std::string activation, std::string optimizer)
{
	// create weight matrices
	parameters.weights.push_back(Matrix(100, in_channel));
	for (int i = 0; i < num_hidden_layers - 1; i++)
		parameters.weights.push_back(Matrix(100, 100));
	parameters.weights.push_back(Matrix(out_channel, in_channel));

	// create bias matrices
	for (int i = 0; i < num_hidden_layers; i++)
		parameters.biases.push_back(Matrix(100, 1));
	parameters.biases.push_back(Matrix(out_channel, 1));

	// initialize weights with random values
	std::for_each(parameters.weights.begin(), parameters.weights.end(), [](Matrix& m) {return m.randomize(0, 1, .01); });

	eta = lr;
	epochs = epoch;
	this->batch_size = batch_size;
	this->activation = activation;

	this->optimizer = Optimizer(eta);
	if (optimizer == "adam")
	{
		this->optimizer = Optimizer(parameters.weights, parameters.biases, eta);
	}
}

NeuralNet::NeuralNet(std::vector<int> detailed_layers, double lr, int epoch, int batch_size, std::string activation, std::string optimizer)
{
	for (int i = 1; i < detailed_layers.size(); i++)
	{
		// create weight matrices
		parameters.weights.push_back(Matrix(detailed_layers[i], detailed_layers[i-1]));
		
		// create bias matrices
		parameters.biases.push_back(Matrix(detailed_layers[i], 1));
	}
	
	// initialize weights with random values
	std::for_each(parameters.weights.begin(), parameters.weights.end(), [](Matrix& m) {return m.randomize(-1, 1, .01); });

	eta = lr;
	epochs = epoch;
	this->batch_size = batch_size;
	this->activation = activation;
	this->optimizer = Optimizer(eta);
	if (optimizer == "adam")
	{
		this->optimizer = Optimizer(parameters.weights, parameters.biases, eta);
	}
}

net_result NeuralNet::forward(const Matrix& X, bool activate_last)
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
	for (int i = 0; i < parameters.weights.size() - 1; i++)
	{
		inputs.push_back(input);

		mul = parameters.weights[i] * input;
		layer = mul + parameters.biases[i].broadcast(mul);
		// next_input = last_activ;
		input = Matrix::apply_func(layer, activation == "relu" ? relu : sigmoid);

		//layers.push_back(layer);
		activations.push_back(input);
	}
	// forward for last/output layer
	mul = parameters.weights.back() * input;
	layer = mul + parameters.biases.back().broadcast(mul);
	if (activate_last)
		yhat = softmax(layer);
	else
		yhat = layer;

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
	Matrix prev_d_activ = Matrix::transpose(parameters.weights.back()) * d_activ;

	d_weights.push_back(d_weight);		// W(n)
	d_biases.push_back(d_bias);			// B(n)
	d_activs.push_back(prev_d_activ);	// A(n-1)

	// remaining layers in backward order
	for (int i = parameters.weights.size() - 2; i >= 0; --i)
	{
		// layer-Z, activation-A
		// note: non-derivative weight/layer and bias
		input = inputs[i];
		weight = parameters.weights[i];
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
	for (int epoch = 1; epoch <= epochs; epoch++)
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

			// forward
			net_result result = forward(X_batch, true);

			// calculate loss
			double batch_loss = cross_entropy_loss(y_batch, result.yhat).m_data[0];
			batch_losses.push_back(batch_loss);

			// calculate gradients
			result = backward(result, X_batch, y_batch);

			// update gradients to model parameters
			optimizer.step(parameters, result);
		}

		epoch_losses.push_back(std::accumulate(batch_losses.begin(), batch_losses.end(), .0) / batch_losses.size());
		if (epoch % 50 == 0)
		{
			double train_accuracy = eval(X, y);
			printf("Epoch %d/%d: training loss: %f, training accuracy: %f, time taken: %f s\n", epoch, epochs, epoch_losses.back(), train_accuracy, (getTickcount() - e_time) / 1000.0);

			// check loss changes for the past number of epochs

			int num_epoch_to_check = 50;
			if (std::adjacent_find(epoch_losses.end() - num_epoch_to_check, epoch_losses.end(), std::not_equal_to<>()) == epoch_losses.end())
			{
				printf("Epoch %d/%d: training loss hasn't change for %d epochs, stopping training early!\n", epoch, epochs, num_epoch_to_check);
				break;
			}
		}
	}
	printf("Training complete! Time taken: %f s\n", (getTickcount() - time) / 1000.0);
}

Matrix NeuralNet::predict(const Matrix& X_in)
{
	Matrix X(X_in);
	X.transpose();

	net_result result = forward(X, false);

	return result.yhat.argmax(1);
}

void NeuralNet::print_eval(const Matrix& X_in, const Matrix& y_in)
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

double NeuralNet::eval(const Matrix& X_in, const Matrix& y_in)
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

	for (int i = 0; i < parameters.weights.size(); i++)
		parameters.weights[i].load_data_txt(parameters.weights[i].get_height(), parameters.weights[i].get_width(), 
			R"(../data/test_nn/W)" + std::to_string(i+1) + ".txt");

	for (int i = 0; i < 10; i++)
	{
		net_result result = forward(X, true);

		result = backward(result, X, y);

		optimizer.step(parameters, result);
	}
	return;
}

void NeuralNet::load_model(const std::string& path)
{
	
}

void NeuralNet::save_model(const std::string& path)
{
	
}
