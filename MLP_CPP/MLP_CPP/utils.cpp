#include "utils.h"

#include <filesystem>
#include <functional>

#define GET_NAME(variable) (#variable)

int64_t getTickcount()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
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

Matrix softmax(const Matrix& matrix)
{
	Matrix max = matrix.max();
	Matrix _V = Matrix::mismatch_dim_subtract(matrix, max);
	Matrix e_V = Matrix::apply_func(_V, [](double x) {return std::exp(x); });
	Matrix sum = e_V.sum(0);
	Matrix Z = Matrix::mismatch_dim_divide(e_V, sum);
	return Z;
}

Matrix grad_softmax(Matrix error, Matrix yhat)
{
	Matrix result(error);
	for (size_t i = 0; i < error.get_width(); ++i)
	{
		// extract yhat from batch
		Matrix y = yhat.extract(i, 1);
		// create diagonal matrix
		int N = std::max(y.get_height(), y.get_width());
		Matrix diag(N,N);
		int index = 0;
		// error here
		for (int i = 0; i < y.m_data.size(); i++)
		{
			diag.m_data[i + index] = y.m_data[i];
			index += N;
		}
		Matrix b = y * Matrix::transpose(y);
		Matrix jacobian = diag - b;
		Matrix r = jacobian * error.extract(i, 1);
		for (size_t j = 0; j < r.get_height(); j++)
			result(j, i) = r(j, 0);
	}
	return result;
}

Matrix cross_entropy_loss(const Matrix& y, const Matrix& yhat)
{
	Matrix log_m = Matrix::apply_func(yhat, [](double x) {return std::log(x); });
	Matrix mul = Matrix::element_wise_mul(y, log_m);
	Matrix sum = mul.sum(-1);
	return (0. - sum) / yhat.get_width();
}

Matrix grad_cross_entropy_loss(const Matrix& y, const Matrix& yhat)
{
	return yhat - y / yhat.get_width();
}

double mse_loss(const Matrix& y, const Matrix& yhat)
{
	Matrix loss = Matrix::power(y - yhat, 2);
	return std::accumulate(loss.m_data.begin(), loss.m_data.end(), 0.0) / yhat.m_data.size();
}

Matrix grad_mse_loss(const Matrix& y, const Matrix& yhat)
{
	Matrix result = yhat - y;
	result = 2.0 * result;
	return result / result.get_height();
}

void save_model_to_file(parameters& params, const std::string& filepath)
{
	if (!std::filesystem::is_directory(std::filesystem::path(filepath).parent_path()))
	{
		std::filesystem::create_directories(std::filesystem::path(filepath).parent_path());
	}
	std::ofstream outFile;
	outFile.open(filepath, std::ios::binary);
	for (int i = 0; i < params.weights.size(); i++)
		params.weights[i].save_data(outFile);
	for (int i = 0; i < params.biases.size(); i++)
		params.biases[i].save_data(outFile);
}

void load_model_from_file(parameters& params, const std::string& filepath)
{
	std::ifstream inFile;
	inFile.open(filepath, std::ios::in | std::ios::binary);
	for (int i = 0; i < params.weights.size(); i++)
		params.weights[i].load_data(inFile);
	for (int i = 0; i < params.biases.size(); i++)
		params.biases[i].load_data(inFile);
}

void save_svm_to_file(Matrix weight, double bias, const std::string& filepath)
{
	if (!std::filesystem::is_directory(std::filesystem::path(filepath).parent_path()))
	{
		std::filesystem::create_directories(std::filesystem::path(filepath).parent_path());
	}
	std::ofstream outFile;
	outFile.open(filepath, std::ios::binary);
	size_t rows = weight.get_height();
	size_t cols = weight.get_width();
	outFile.write((char*)(&rows), sizeof(rows));
	outFile.write((char*)(&cols), sizeof(cols));
	outFile.write((char*) &weight.m_data[0], sizeof(double) * weight.get_height() * weight.get_width());

	outFile.write((char*)(&bias), sizeof(bias));
}

void load_svm_from_file(Matrix& weight, double& bias, const std::string& filepath)
{
	std::ifstream inFile;
	inFile.open(filepath, std::ios::in | std::ios::binary);
	size_t rows, columns;
	inFile.read((char*)&rows, sizeof(rows));
	inFile.read((char*)&columns, sizeof(columns));
	weight = Matrix(rows, columns);
	double* m = new double[rows * columns];
	inFile.read((char*)m, sizeof(double) * rows * columns);
	std::vector<double> mat(m, m + rows * columns);
	weight.m_data = std::move(mat);
	delete[] m;
}

void save_snapshot_to_file(const parameters& params, const net_result& net_output, const std::string& directory)
{
	std::ofstream outFile;
	for (int i = 0; i < params.weights.size(); i++)
	{
		std::string filename = "W" + std::to_string(i + 1) + ".txt";
		outFile.open(directory + filename, std::ios::binary);
		outFile.write((char*)&params.weights[i].m_data[0], sizeof(double) * params.weights[i].get_width() * params.weights[i].get_height());
	}
	for (int i = 0; i < params.biases.size(); i++)
	{
		std::string filename = "B" + std::to_string(i + 1) + ".txt";
		outFile.open(directory + filename, std::ios::binary);
		outFile.write((char*)&params.biases[i].m_data[0], sizeof(double) * params.biases[i].get_width() * params.biases[i].get_height());
	}
	std::string filename = "net_output(yhat).txt";
	outFile.open(directory + filename, std::ios::binary);
	outFile.write((char*)&net_output.yhat.m_data[0], sizeof(double) * net_output.yhat.get_width() * net_output.yhat.get_height());
}