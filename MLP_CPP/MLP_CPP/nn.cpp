#include "nn.h"

#include "utils.h"

Optimizer::Optimizer(double lr, double batch_sz)
{
	name = "sgd";
	learning_rate = lr;
	batch_size = batch_sz;
}

Optimizer::Optimizer(const std::vector<Matrix>& weights, const std::vector<Matrix>& biases, const double& lr, const double& batch_sz, const double& b1 = .9, const double& b2 = .999, const double& e = 1e-8)
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
	batch_size = batch_sz;
	beta1 = b1;
	beta2 = b2;
	epsilon = e;
}

void Optimizer::sgd_step(parameters& params, const net_result& grads)
{
	parameters result(params);
	net_result g(grads);
	int N = params.weights.size();

	/*std::for_each(g.grad_weights.begin(), g.grad_weights.end(), [this](Matrix& m) {return m / batch_size; });
	std::for_each(g.grad_biases.begin(), g.grad_biases.end(), [this](Matrix& m) {return m / batch_size; });*/

	for (int i = N - 1; i >= 0; i--)
	{
		result.weights[i] = params.weights[i] - learning_rate * g.grad_weights[N - 1 - i];
		result.biases[i] = params.biases[i] - learning_rate * g.grad_biases[N - 1 - i];
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

NeuralNet::NeuralNet(size_t in_channel, size_t out_channel, int num_hidden_layers, double lr, int epoch, int batch_size, std::string activation, std::string loss, std::string optimizer, std::string name = "weight")
{
	// version 2
	// create weight matrices
	parameters.weights.push_back(Matrix(in_channel, 100));
	for (int i = 0; i < num_hidden_layers - 1; i++)
		parameters.weights.push_back(Matrix(100, 100));
	parameters.weights.push_back(Matrix(100, out_channel));

	// create bias matrices
	for (int i = 0; i < num_hidden_layers; i++)
		parameters.biases.push_back(Matrix(100, 1));
	parameters.biases.push_back(Matrix(out_channel, 1));

	// initialize weights with random values
	std::for_each(parameters.weights.begin(), parameters.weights.end(), [](Matrix& m) {return m.randomize(0, 1, .001); });

	eta = lr;
	epochs = epoch;
	this->batch_size = batch_size;
	this->activation = activation;
	this->loss_fn = loss;
	this->name = name;

	this->optimizer = Optimizer(eta, this->batch_size);
	if (optimizer == "adam")
	{
		this->optimizer = Optimizer(parameters.weights, parameters.biases, eta, this->batch_size);
	}
}

NeuralNet::NeuralNet(std::vector<int> detailed_layers, double lr, int epoch, int batch_size, std::string activation, std::string loss, std::string optimizer, std::string name = "weight")
{
	// version 2
	for (int i = 0; i < detailed_layers.size() - 1; i++)
	{
		// create weight matrices
		parameters.weights.push_back(Matrix(detailed_layers[i], detailed_layers[i + 1]));

		// create bias matrices
		parameters.biases.push_back(Matrix(detailed_layers[i + 1], 1));
	}
	
	// initialize weights with random values
	std::for_each(parameters.weights.begin(), parameters.weights.end(), [](Matrix& m) {return m.randomize(0, 1, 1.0 / std::sqrt(m.get_height() + m.get_width())); });

	eta = lr;
	epochs = epoch;
	this->batch_size = batch_size;
	this->activation = activation;
	this->optimizer = Optimizer(eta, this->batch_size);
	this->loss_fn = loss;
	this->name = name;

	if (optimizer == "adam")
	{
		this->optimizer = Optimizer(parameters.weights, parameters.biases, eta, this->batch_size);
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

	// version 2
	// forward through each 'hidden layer'
	for (int i = 0; i < parameters.weights.size() - 1; i++)
	{
		inputs.push_back(input);

		mul = Matrix::transpose(parameters.weights[i]) * input;
		layer = mul + parameters.biases[i].broadcast(mul);
		layers.push_back(layer);

		activ = Matrix::apply_func(layer, activation == "relu" ? relu : sigmoid);
		activations.push_back(activ);

		// next_input = last_activ;
		input = activ;
	}
	// forward for last/output layer
	mul = Matrix::transpose(parameters.weights.back()) * input;
	layer = mul + parameters.biases.back().broadcast(mul);
	if (activate_last)
		yhat = softmax(layer);
	else
		yhat = layer;

	inputs.push_back(input);
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
	std::vector<Matrix> d_inputs;

	// version 2
	Matrix input_error;
	Matrix output_error;
	Matrix d_weight;
	Matrix d_bias;

	// Softmax layer
	// Calculate gradient of loss function
	if (loss_fn == "cross_entropy")
	{
		output_error = grad_cross_entropy_loss(y, yhat);
	}
	else if (loss_fn == "mse")
	{
		output_error = grad_mse_loss(y, yhat);
	}
	
	// Calculate gradient of softmax layer with respect to the loss function
	output_error = grad_softmax(output_error, yhat);
	
	// FC layer
	input_error = parameters.weights.back() * output_error;
	d_weight = inputs.back() * Matrix::transpose(output_error);
	d_bias = output_error.sum(1);

	// Append calculated d_weight and d_bias for updating model parameters
	d_weights.push_back(d_weight);
	d_biases.push_back(d_bias);

	// Remove used input
	inputs.pop_back();
	
	// calculating for parameters at layer i with i = len(layers) - 2 -> 1
	for (int i = parameters.weights.size() - 2; i >= 0; i--)
	{
		// Activation layer
		output_error = input_error.element_wise_mul(Matrix::apply_func(layers.back(), activation == "relu" ? d_relu : d_sigmoid));

		// FC layer
		input_error = parameters.weights[i] * output_error;
		d_weight = inputs.back() * Matrix::transpose(output_error);
		d_bias = output_error.sum(1);

		// Remove used input and layer
		inputs.pop_back();
		layers.pop_back();

		// Append calculated d_weight and d_bias for updating model parameters
		d_weights.push_back(d_weight);
		d_biases.push_back(d_bias);
	}

	//// version 1
	//Matrix error;
	//// Calculate gradient of loss function
	//if (loss_fn == "cross_entropy")
	//{
	//	error = grad_cross_entropy_loss(y, yhat);
	//}
	//else if (loss_fn == "mse")
	//{
	//	error = grad_mse_loss(y, yhat);
	//}
	//Matrix d_weight;
	//Matrix d_bias;
	//for (int i = parameters.weights.size() - 1; i >= 0; i--)
	//{
	//	d_weight = inputs.back() * Matrix::transpose(error);
	//	d_bias = error.sum(1);
	//	// there's no error at the first input layer
	//	if (i != 0)
	//	{
	//		error = parameters.weights[i] * error;
	//		error.element_wise_mul(Matrix::apply_func(layers.back(), activation == "relu" ? d_relu : d_sigmoid));
	//		inputs.pop_back();
	//		layers.pop_back();
	//	}

	//	d_weights.push_back(d_weight);
	//	d_biases.push_back(d_bias);
	//}

	return net_result(d_activs, d_weights, d_biases);
}

void NeuralNet::fit(const Matrix& X_in, const Matrix& y_in)
{
	Matrix X(X_in);
	Matrix y(y_in);
	double N = X.get_height();
	std::vector<double> epoch_losses;
	char path[500];
	sprintf_s(path, "../weights/%s.bin", name.c_str());
	std::string modelPath = path;

	std::cout << "Training started!\n";
	int64_t time = getTickcount();
	int64_t e_time = getTickcount();

	for (int epoch = 1; epoch <= epochs; epoch++)
	{
		printf("\rEpoch %d/%d (%f s/it): ", epoch, epochs, (getTickcount() - e_time) / 1000.0 );
		e_time = getTickcount();
		std::vector<double> batch_losses;

		Matrix::shuffle(X, y);
		int real_batch_size;
		for (int q = 0; q < N; q += batch_size)
		{
			if (q + batch_size - 1 > N)
				real_batch_size = N - q;
			else real_batch_size = batch_size;
			Matrix X_batch = X.extract(q, q + real_batch_size - 1, 0);
			Matrix y_batch = y.extract(q, q + real_batch_size - 1, 0);

			X_batch.transpose();
			y_batch.transpose();

			// forward
			net_result result = forward(X_batch, true);

			// calculate loss
			if (loss_fn == "cross_entropy")
			{
				double batch_loss = cross_entropy_loss(y_batch, result.yhat).m_data[0];
				batch_losses.push_back(batch_loss);
			}
			else if (loss_fn == "mse")
			{
				double batch_loss = mse_loss(y_batch, result.yhat);
				batch_losses.push_back(batch_loss);
			}

			// calculate gradients
			result = backward(result, X_batch, y_batch);

			// update gradients to model parameters
			optimizer.step(parameters, result);
		}

		epoch_losses.push_back(std::accumulate(batch_losses.begin(), batch_losses.end(), .0) / batch_losses.size());
		if (epoch % 10 == 0)
		{
			double train_accuracy = eval(X, y);
			printf("\rEpoch %d/%d (%f s/it): training loss: %f, training accuracy: %f\n", epoch, epochs, (getTickcount() - e_time) / 1000.0, epoch_losses.back(), train_accuracy);

			//check loss changes for the past number of epochs
			int num_epoch_to_check = 50;
			if (epoch >= num_epoch_to_check)
			{
				if (std::adjacent_find(epoch_losses.end() - num_epoch_to_check, epoch_losses.end(), std::not_equal_to<>()) == epoch_losses.end())
				{
					printf("Epoch %d/%d: training loss hasn't change for %d epochs, stopping training early!\n", epoch, epochs, num_epoch_to_check);
					break;
				}
			}
		}

		// saving model
		if (*std::min_element(epoch_losses.begin(), epoch_losses.end()) - epoch_losses.back() < std::numeric_limits<double>::epsilon())
		{
			save_model(modelPath);
		}
	}
	std::cout << "Saving losses to file...\n";
	sprintf_s(path, "../losses/%s.loss", name.c_str());
	save_vector_to_file(epoch_losses, path);
	printf("Training complete! Total time taken: %f s\n", (getTickcount() - time) / 1000.0);
}

Matrix NeuralNet::predict(const Matrix& X_in)
{
	Matrix X(X_in);
	X.transpose();

	net_result result = forward(X, true);

	return result.yhat.argmax(1);
	return result.yhat;
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

	// other metrics
	double precision = get_precision(y_pred, y);
	double recall = get_recall(y_pred, y);
	double f1 = get_f1(y_pred, y);

	printf("Model precision: %f, recall: %f, F1-score: %f\n", precision, recall, f1);

	// save prediction for presentation/ plotting
	save_vector_to_file(y_pred.m_data, "../prediction_results/" + this->name + ".txt");
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

void NeuralNet::simple_test()
{
	// X co 50 hang, 16 features
	Matrix X(16, 50);
	// y co 50 hang, 26 output
	Matrix y(26, 50);

	Matrix x_test;
	Matrix y_test;

	/*X.randomize(-1,1);
	y.randomize(1, 2);*/

	X.load_data_txt(16000, 16, R"(../data/test_nn/x.txt)");
	y.load_data_txt(16000, 26, R"(../data/test_nn/y.txt)");

	x_test.load_data_txt(4000, 16, R"(../data/test_nn/x_test.txt)");
	y_test.load_data_txt(4000, 26, R"(../data/test_nn/y_test.txt)");

	for (int i = 0; i < parameters.weights.size(); i++)
		parameters.weights[i].load_data_txt(parameters.weights[i].get_height(), parameters.weights[i].get_width(),
			R"(../data/test_nn/W)" + std::to_string(i + 1) + ".txt");
	std::vector<double> losses;
	for (int i = 0; i < X.get_height(); i++)
	{
		Matrix _x = X.extract(i, i + batch_size - 1, 0);
		Matrix _y = y.extract(i, i + batch_size - 1, 0);
		_x.transpose();
		_y.transpose();

		// forward
		net_result out_result = forward(_x, true);

		// calculate loss
		double loss;
		if (loss_fn == "cross_entropy")
		{
			loss = cross_entropy_loss(_y, out_result.yhat).m_data[0];
		}
		else if (loss_fn == "mse")
		{
			loss = mse_loss(_y, out_result.yhat);
		}
		losses.push_back(loss);
		// calculate gradients
		net_result result = backward(out_result, _x, _y);

		// update gradients to model parameters
		optimizer.step(parameters, result);

		save_snapshot_to_file(parameters, out_result, R"(../data/compare_test/)");
	}
	std::cout << "loss: " << std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size() << "\n";
	double train_accuracy = eval(X, y);
	std::cout << train_accuracy << "\n";
	print_eval(x_test, y_test);
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

	X.transpose();
	y.transpose();

	for (int i = 0; i < parameters.weights.size(); i++)
		parameters.weights[i].load_data_txt(parameters.weights[i].get_height(), parameters.weights[i].get_width(), 
			R"(../data/test_nn/W)" + std::to_string(i+1) + ".txt");

	double N = X.get_width();
	std::vector<double> epoch_losses;
	int64_t time = getTickcount();
	for (int i = 0; i < 10000; i++)
	{
		printf("\rEpoch %d/%d: ", i, 10000);
		Matrix::shuffle(X, y);
		std::vector<double> batch_losses;
		int64_t e_time = getTickcount();
		for (int q = 0; q < N; q += batch_size)
		{
			printf(".");
			Matrix X_batch = X.extract(q, q + batch_size - 1, 0);
			Matrix y_batch = y.extract(q, q + batch_size - 1, 0);

			X_batch.transpose();
			y_batch.transpose();

			// forward
			net_result result = forward(X_batch, true);

			// calculate loss
			if (loss_fn == "cross_entropy")
			{
				double batch_loss = cross_entropy_loss(y_batch, result.yhat).m_data[0];
				batch_losses.push_back(batch_loss);
			}
			else if (loss_fn == "mse")
			{
				double batch_loss = mse_loss(y_batch, result.yhat);
				batch_losses.push_back(batch_loss);
			}

			// calculate gradients
			result = backward(result, X_batch, y_batch);

			// update gradients to model parameters
			optimizer.step(parameters, result);
		}

		epoch_losses.push_back(std::accumulate(batch_losses.begin(), batch_losses.end(), .0) / batch_losses.size());
		if (i % 1000 == 0)
		{
			double train_accuracy = eval(X, y);
			printf("Epoch %d/%d: training loss: %f, training accuracy: %f, time taken: %f s\n", i, epochs, epoch_losses.back(), train_accuracy, (getTickcount() - e_time) / 1000.0);
		}
	}
	return;
}

void NeuralNet::load_model(const std::string& path)
{
	load_model_from_file(parameters, path);
}

void NeuralNet::save_model(const std::string& path)
{
	save_model_to_file(parameters, path);
}
