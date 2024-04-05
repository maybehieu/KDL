#include "matrix.h"

Matrix::Matrix(size_t rows, size_t cols) : m_cols(cols), m_rows(rows), m_data({})
{
	m_data.resize(cols * rows, double());
	m_shape = { rows, cols };
}

Matrix::Matrix(std::vector<double>& data, size_t rows, size_t cols) : m_cols(cols), m_rows(rows), m_data(data)
{
	
}


Matrix::Matrix() : m_cols(0), m_rows(0), m_data({})
{
	m_shape = { m_rows, m_cols };
}

Matrix::Matrix(const Matrix& matrix) : m_rows(matrix.m_rows), m_cols(matrix.m_cols), m_data(matrix.m_data)
{

}

Matrix&::Matrix::operator =(const Matrix& matrix)
{
	m_rows = matrix.m_rows;
	m_cols = matrix.m_cols;
	m_shape = matrix.m_shape;
	m_numel = matrix.m_numel;
	m_data = matrix.m_data;
	return *this;
}

Matrix& ::Matrix::operator =(Matrix&& matrix)
{
	m_rows = matrix.m_rows;
	m_cols = matrix.m_cols;
	m_shape = matrix.m_shape;
	m_numel = matrix.m_numel;
	m_data = std::move(matrix.m_data);
	return *this;
}

double& Matrix::operator ()(size_t row, size_t col)
{
#ifdef _DEBUG
	if ((row * m_cols + col) >= m_rows * m_cols)
		throw MatrixError("Index out of range!");
#endif
	return m_data[row * m_cols + col];
}

const double& Matrix::operator ()(size_t row, size_t col) const
{
#ifdef _DEBUG
	if ((row * m_cols + col) >= m_rows * m_cols)
		throw MatrixError("Index out of range!");
#endif
	return m_data[row * m_cols + col];
}

double& Matrix::operator [](const std::pair<size_t, size_t>& index)
{
#ifdef _DEBUG
	if ((index.first * m_cols + index.second) >= m_rows * m_cols)
		throw MatrixError("Index out of range!");
#endif
	return m_data[index.first * m_cols + index.second];
}

const double& Matrix::operator [](const std::pair<size_t, size_t>& index) const
{
#ifdef _DEBUG
	if ((index.first * m_cols + index.second) >= m_rows * m_cols)
		throw MatrixError("Index out of range!");
#endif
	return m_data[index.first * m_cols + index.second];
}

Matrix& Matrix::operator +=(const Matrix& other) {
#ifdef _DEBUG
	if (!check_dims(other))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	for (size_t i = 0; i < m_rows; i++)
		for (size_t j = 0; j < m_cols; j++)
				m_data[i * m_cols + j] += other(i, j);
	return *this;
}

Matrix& Matrix::operator +=(double scalar) {
	for (size_t i = 0; i < m_rows; i++)
		for (size_t j = 0; j < m_cols; j++)
			m_data[i * m_cols + j] += scalar;
	return *this;
}

Matrix& Matrix::operator -=(const Matrix& other) {
#ifdef _DEBUG
	if (!check_dims(other))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	for (size_t i = 0; i < m_rows; i++)
		for (size_t j = 0; j < m_cols; j++)
			m_data[i * m_cols + j] -= other(i, j);
	return *this;
}

Matrix  Matrix::operator -=(double scalar) {
	for (size_t i = 0; i < m_rows; i++)
		for (size_t j = 0; j < m_cols; j++)
			m_data[i * m_cols + j] -= scalar;
	return *this;
}

Matrix  Matrix::operator *=(double scalar) {
	for (size_t i = 0; i < m_rows; i++)
		for (size_t j = 0; j < m_cols; j++)
			(*this)(i, j) =  scalar;
	return *this;
}

Matrix& Matrix::operator *=(const Matrix& other) {
#ifdef _DEBUG
	if (m_cols != other.m_rows)
		throw MatrixError("Number of columns of the left matrix has to match number of rows of the right matrix!");
#endif // _DEBUG
	* this = *this * other;
	return *this;
}

Matrix  Matrix::operator /=(double scalar) {
#ifdef _DEBUG
	if (scalar == 0.0)
		throw MatrixError("Cannot divide by zero!");
#endif // _DEBUG
	for (size_t i = 0; i < m_rows; i++)
		for (size_t j = 0; j < m_cols; j++)
			(*this)(i, j) /= scalar;
	return *this;
}

std::ostream& operator <<(std::ostream& out, const Matrix& m) {
	for (unsigned int i = 0; i < m.m_rows; i++)
	{
		for (unsigned int j = 0; j < m.m_cols; j++)
		{
			out << m.m_data[i * m.m_cols + j] << " ";
		}
		out << "\n";
	}
	return out;
}

Matrix operator +(const Matrix& left, const Matrix& right) {
#ifdef _DEBUG
	if (!left.check_dims(right))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	Matrix result(left);
	for (size_t i = 0; i < left.m_rows; i++)
		for (size_t j = 0; j < left.m_cols; j++)
			result(i, j) = left(i, j) + right(i, j);
	return result;
}

Matrix operator *(const Matrix& matrix, double scalar) {
	Matrix result(matrix);
	for (size_t i = 0; i < matrix.m_rows; i++)
		for (size_t j = 0; j < matrix.m_cols; j++)
			result(i, j) = matrix(i, j) * scalar;
	return result;
}

Matrix operator *(double scalar, const Matrix& matrix) {
	Matrix result(matrix);
	for (size_t i = 0; i < matrix.m_rows; i++)
		for (size_t j = 0; j < matrix.m_cols; j++)
			result(i, j) = matrix(i, j) * scalar;
	return result;
}

Matrix operator -(const Matrix& left, const Matrix& right) {
#ifdef _DEBUG
	if (!left.check_dims(right))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	Matrix result(left);
	for (size_t i = 0; i < left.m_rows; i++)
		for (size_t j = 0; j < left.m_cols; j++)
			result(i, j) = left(i, j) - right(i, j);
	return result;
}

Matrix operator -(double scalar, const Matrix& matrix) {
	Matrix result(matrix);
	for (size_t i = 0; i < matrix.m_rows; i++)
		for (size_t j = 0; j < matrix.m_cols; j++)
			result(i, j) = scalar - matrix(i, j);
	return result;
}

Matrix operator *(const Matrix& left, const Matrix& right) {
#ifdef _DEBUG
	if (left.m_cols != right.m_rows)
		throw MatrixError("Number of columns of the left matrix has to match number of rows of the right matrix!");
#endif // _DEBUG

	Matrix result(left.m_rows, right.m_cols);
	for (unsigned int i = 0; i < left.m_rows; i++)
		for (unsigned int k = 0; k < left.m_cols; k++)
			for (unsigned int j = 0; j < right.m_cols; j++)
				result(i, j) += left(i, k) * right(k, j);
	return result;
}

Matrix operator /(const Matrix& matrix, double scalar) {
#ifdef _DEBUG
	if (scalar == 0.0)
		throw MatrixError("Cannot divide by zero!");
#endif // _DEBUG
	Matrix result(matrix);
	for (size_t i = 0; i < matrix.m_rows; i++)
		for (size_t j = 0; j < matrix.m_cols; j++)
			result(i, j) = matrix(i, j) / scalar;
	return result;
}

Matrix operator /(const Matrix& left, const Matrix& right) {
#ifdef _DEBUG
	if (!left.check_dims(right))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	Matrix result(left);
	for (size_t i = 0; i < left.m_rows; i++)
		for (size_t j = 0; j < left.m_cols; j++)
			result(i, j) = left(i, j) / right(i, j);
	return result;
}

bool Matrix::check_dims(const Matrix& other) const
{
	return m_rows == other.m_rows && m_cols == other.m_cols;
}

void Matrix::print()
{
	for (unsigned int i = 0; i < m_rows; i++)
	{
		for (unsigned int j = 0; j < m_cols; j++)
		{
			std::cout << m_data[i * m_cols + j] << " ";
		}
		std::cout << "\n";
	}
}
