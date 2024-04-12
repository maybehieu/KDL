#include "utils.h"
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
	for (size_t i = 0; i < error.get_height(); ++i)
	{
		// create diagonal matrix
		int N = std::max(yhat.get_height(), yhat.get_width());
		Matrix diag(N,N);
		int index = 0;
		// error here
		for (int i = 0; i < yhat.m_data.size(); i++)
		{
			diag.m_data[i + index] = yhat.m_data[i];
			index += N;
		}
		Matrix b = yhat * Matrix::transpose(yhat);
		Matrix jacobian = diag - b;
		Matrix r = jacobian * error.extract(i, 1);
		std::copy_n(r.m_data.begin(), r.m_data.size(), result.m_data.begin() + i * result.get_height());
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
	return std::accumulate(loss.m_data.begin(), loss.m_data.end(), 0.0) / loss.m_data.size();
}

Matrix grad_mse_loss(const Matrix& y, const Matrix& yhat)
{
	Matrix result = yhat - y;
	result = 2.0 * result;
	return result / yhat.m_data.size();
}

void save_parameters_to_file(const parameters& params, const std::string& directory)
{
	
}
