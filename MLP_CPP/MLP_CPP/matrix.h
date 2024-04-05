#pragma once
#include <stdexcept>
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <random>
#include <numeric>
#include <functional>

struct MatrixError : std::runtime_error
{
	MatrixError(const char* error) : std::runtime_error(error) {};
};

class Matrix
{
	size_t m_cols;
	size_t m_rows;
public:
	std::vector<double> m_data;
	std::tuple<size_t, size_t> m_shape;
	int m_numel = m_rows * m_cols;

	Matrix(size_t rows, size_t cols);
	Matrix(std::vector<double>& data, size_t rows, size_t cols);
	Matrix();
	Matrix(const Matrix&);

	// utils
	void print_shape();
	void print();
	inline size_t get_width() const { return m_cols; }
	inline size_t get_height() const { return m_rows; }
	bool check_dims(const Matrix& other) const;
	Matrix clone()
	{
		std::vector<double> data = std::vector<double>(0);
		for (auto z : m_data) data.push_back(z);
		return Matrix(data, m_rows, m_cols);
	}

	// override operations
	Matrix& operator =(const Matrix& matrix);							// get/ copy matrix
	Matrix& operator =(Matrix&& matrix);								// move matrix

	// access elements
	double& operator ()(size_t row, size_t col);
	const double& operator ()(size_t row, size_t col) const;
	double& operator [](const std::pair<size_t, size_t>& index);
	const double& operator [](const std::pair<size_t, size_t>& index) const;

	// assignment
	Matrix& operator +=(const Matrix& other);
	Matrix& operator +=(double scalar);
	Matrix& operator -=(const Matrix& other);
	Matrix operator -=(double scalar);
	Matrix operator *=(double scalar);
	Matrix& operator *= (const Matrix& other);
	Matrix operator /= (double scalar);

	//
	friend std::ostream& operator << (std::ostream& out, const Matrix& m);

	friend Matrix operator +(const Matrix& left, const Matrix& right);
	friend Matrix operator -(const Matrix& left, const Matrix& right);
	friend Matrix operator -(double scalar, const Matrix& matrix);
	friend Matrix operator *(const Matrix& left, const Matrix& right);
	friend Matrix operator *(const Matrix& matrix, double scalar);
	friend Matrix operator *(double scalar, const Matrix& matrix);
	friend Matrix operator /(const Matrix& left, const Matrix& right);
	friend Matrix operator /(const Matrix& matrix, double scalar);
};
