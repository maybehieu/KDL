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

Matrix mse_loss(const Matrix& y, const Matrix& yhat)
{
	
}

Matrix grad_mse_loss(const Matrix& y, const Matrix& yhat)
{
	
}

void save_parameters_to_file(const parameters& params, const std::string& directory)
{
	
}
