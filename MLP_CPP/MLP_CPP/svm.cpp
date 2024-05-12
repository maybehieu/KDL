#include "svm.h"

#include "utils.h"

SoftMarginSVM::SoftMarginSVM(double constant = 1.0, double learning_rate = 0.001, int epochs = 1000, std::string name="svm")
{
	this->constant = constant;

	this->lr = learning_rate;
	this->epochs = epochs;

	this->name = name;
}

Matrix SoftMarginSVM::cost(const Matrix& in_margins)
{
	Matrix margins(in_margins);
	Matrix reg_term = .5 * Matrix::transpose(weight) * weight;
	Matrix hinge_loss = 1.0 - margins;
	std::for_each(hinge_loss.m_data.begin(), hinge_loss.m_data.end(), [](double& x) {x = std::max(x, 0.0); });
	return reg_term + constant * hinge_loss.sum(-1);
}

void SoftMarginSVM::fit(const Matrix& x, const Matrix& y)
{
	int num_features = x.get_width();
	int num_samples = x.get_height();

	// initialize weight and bias
	weight = Matrix(num_features, 1);
	bias = 0.0;

	//// load previous model
	//load_svm_from_file(weight, bias, "../svm_weights/mnist_01.bin");

	std::vector<double> losses;
	Matrix margins;
	Matrix loss;
	std::cout << "Training started!\n";
	int64_t time = getTickcount();
	int64_t e_time = getTickcount();
	char path[500];
	sprintf_s(path, "../svm_weights/%s.bin", name.c_str());
	std::string modelPath = path;

	for (int epoch = 1; epoch <= epochs; epoch++)
	{
		printf("\rEpoch %d/%d (%f s/it): ", epoch, epochs, (getTickcount() - e_time) / 1000.0);
		e_time = getTickcount();
		margins = Matrix::element_wise_mul(y, x * weight + bias);
		loss = cost(margins);
		losses.push_back(loss.m_data[0]);

		// perform gradient descent on misclassified point
		Matrix y_miss(std::vector<double>(), 1, 1);
		Matrix x_miss(std::vector<double>(), 1, num_features);
		for (int idx = 0; idx < num_samples; idx++)
		{
			if (margins.m_data[idx] >= 1.0) continue;
			x_miss.concat(x, idx, 0);
			y_miss.concat(y, idx, 0);
		}
		// cut down excess row during matrix initialization
		x_miss.change_height(-1);
		y_miss.change_height(-1);

		Matrix a = constant * (Matrix::transpose(y_miss) * x_miss);

		Matrix grad_weight = weight - Matrix::transpose(a);
		double grad_bias = -constant * y_miss.sum(-1).m_data[0];

		// update weight and bias
		weight -= lr * grad_weight;
		bias -= lr * grad_bias;

		if (epoch % 10 == 0)
		{
			double train_acc = eval(x, y);
			printf("\rEpoch %d/%d (%f s/it): training loss: %f, training accuracy: %f\n", epoch, epochs, (getTickcount() - e_time) / 1000.0, losses.back(), train_acc);
		}

		// save SVM state
		if (*std::min_element(losses.begin(), losses.end()) - losses.back() < std::numeric_limits<double>::epsilon())
		{
			save_model(modelPath);
		}
	}

	std::cout << "Saving losses to file...\n";
	sprintf_s(path, "../losses/%s.loss", name.c_str());
	save_vector_to_file(losses, path);
	printf("Training complete! Total time taken: %f s\n", (getTickcount() - time) / 1000.0);
}

Matrix SoftMarginSVM::predict(const Matrix& x)
{
	Matrix y = x * weight + bias;
	std::for_each(y.m_data.begin(), y.m_data.end(), [](double& x)
		{
			if (x < 0) x = -1;
			else if (x - 0.0 < std::numeric_limits<double>::epsilon()) x = 0;
			else x = 1;
		});
	return y;
}

double SoftMarginSVM::eval(const Matrix& X_in, const Matrix& y_in)
{
	Matrix X(X_in);
	Matrix y(y_in);
	double N = X.get_width();

	Matrix y_pred = predict(X);

#ifdef _DEBUG
	if (y_pred.m_data.size() != y.m_data.size())
	{
		throw MatrixError("Prediction output doesn't match ground truth shape!");
	}
#endif

	// accuracy
	std::vector<float> acc;
	for (size_t i = 0; i < y_pred.m_data.size(); i++)
		acc.push_back(y_pred.m_data[i] - y.m_data[i] < std::numeric_limits<double>::epsilon() ? 1 : 0);
	return std::accumulate(acc.begin(), acc.end(), .0) / acc.size();
}

void SoftMarginSVM::print_eval(const Matrix& X_in, const Matrix& y_in)
{
	Matrix X(X_in);
	Matrix y(y_in);
	double N = X.get_width();

	Matrix y_pred = predict(X);

#ifdef _DEBUG
	if (y_pred.m_data.size() != y.m_data.size())
	{
		throw MatrixError("Prediction output doesn't match ground truth shape!");
	}
#endif

	// accuracy
	std::vector<float> acc;
	for (size_t i = 0; i < y_pred.m_data.size(); i++)
		acc.push_back(y_pred.m_data[i] - y.m_data[i] < std::numeric_limits<double>::epsilon() ? 1 : 0);
	printf("SVM accuracy: %f\n", std::accumulate(acc.begin(), acc.end(), .0) / acc.size());

	// convert -1 to 0 for sync (metrics evaluation)
	std::for_each(y.m_data.begin(), y.m_data.end(), [](double& x){	if (x == -1) x = 0;	});
	std::for_each(y_pred.m_data.begin(), y_pred.m_data.end(), [](double& x){ if (x == -1) x = 0; });

	// other metrics
	double precision = get_precision(y_pred, y);
	double recall = get_recall(y_pred, y);
	double f1 = get_f1(y_pred, y);

	printf("SVM precision: %f, recall: %f, F1-score: %f\n", precision, recall, f1);
}

void SoftMarginSVM::load_model(const std::string& path)
{
	load_svm_from_file(weight, bias, path);
}

void SoftMarginSVM::save_model(const std::string& path)
{
	save_svm_to_file(weight, bias, path);
}
