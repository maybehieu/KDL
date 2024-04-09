#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
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
	Matrix(size_t rows, size_t cols, double initValue);
	Matrix(std::vector<double>& data, size_t rows, size_t cols);
	Matrix();
	Matrix(const Matrix&);
	Matrix(size_t elem_num);

	// utils
	void print();
	inline size_t get_width() const { return m_cols; }
	inline size_t get_height() const { return m_rows; }
	inline void print_shape() { std::cout << "{ " << m_rows << ", " << m_cols << " }\n"; }

	// data
	Matrix& load_data_txt(const size_t rows, const size_t cols, const std::string& s);
	Matrix& load_data_csv(const size_t rows, const size_t cols, const std::string& file, const bool isHeader = true);
	Matrix& load_data(const std::string& file);

	void save_data(const std::string& file);

	bool check_dims(const Matrix& other) const;
	Matrix clone()
	{
		std::vector<double> data = std::vector<double>(0);
		for (auto z : m_data) data.push_back(z);
		return Matrix(data, m_rows, m_cols);
	}
	void randomize(double min, double max, double scalar = 1.0);
	void soft_reset();
	Matrix& drop(const int& index, const int& type);
	Matrix extract(const int& index, const int& type);
	Matrix extract(const int& start_index, const int& end_index, const int& type);
	Matrix argmax(const int& type);

	Matrix& element_wise_mul(const Matrix& other);
	Matrix& transpose();
	Matrix& power(double x);
	Matrix& sqrt();
	Matrix& square();
	Matrix  max();
	Matrix  max() const;
	Matrix  sum(int axis);

	// different functions instead for 'broadcasting' (cast one matrix to others shape)
	Matrix broadcast(const Matrix& other);
	static Matrix mismatch_dim_subtract(const Matrix& left, const Matrix& right);
	static Matrix mismatch_dim_divide(const Matrix& left, const Matrix& right);
	static Matrix mismatch_dim_add(const Matrix& left, const Matrix& right);

	static Matrix element_wise_mul(const Matrix& left, const Matrix& right);
	static Matrix square(const Matrix& matrix);
	static Matrix transpose(const Matrix& matrix);
	static Matrix power(const Matrix& mat, const double& x);
	static Matrix sqrt(const Matrix& mat);

	static void shuffle(Matrix& X, Matrix& y);

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
	friend std::ostream& operator << (std::ostream& out, std::tuple<size_t, size_t> m);

	friend Matrix operator +(const Matrix& left, const Matrix& right);
	friend Matrix operator +(const Matrix& mat, double scalar);
	friend Matrix operator -(const Matrix& left, const Matrix& right);
	friend Matrix operator -(double scalar, const Matrix& matrix);
	friend Matrix operator *(const Matrix& left, const Matrix& right);
	friend Matrix operator *(const Matrix& matrix, double scalar);
	friend Matrix operator *(double scalar, const Matrix& matrix);
	friend Matrix operator /(const Matrix& left, const Matrix& right);
	friend Matrix operator /(const Matrix& matrix, double scalar);

	template<typename _Func> Matrix& apply_func(_Func&& func);
	template<typename _Func> static Matrix apply_func(const Matrix& matrix, _Func&& func);
};

template<typename _Func>
inline Matrix& Matrix::apply_func(_Func&& func)
{
	std::for_each(m_data.begin(), m_data.end(), [&func](double& x) { x = func(x); });
	return *this;
}

template<typename _Func>
inline Matrix Matrix::apply_func(const Matrix& matrix, _Func&& func)
{
	Matrix result = matrix;
	result.apply_func(func);
	return result;
}