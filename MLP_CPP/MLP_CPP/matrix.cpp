#include "matrix.h"

Matrix::Matrix(size_t rows, size_t cols) : m_cols(cols), m_rows(rows), m_data({})
{
	m_data.resize(cols * rows, double());
	m_shape = { rows, cols };
}

Matrix::Matrix(size_t rows, size_t cols, double initValue) : m_cols(cols), m_rows(rows)
{
	m_data.resize(cols * rows, initValue);
	m_shape = { rows, cols };
}

Matrix::Matrix(std::vector<double>& data, size_t rows, size_t cols) : m_cols(cols), m_rows(rows), m_data(data)
{
	m_shape = { rows, cols };
}


Matrix::Matrix() : m_cols(0), m_rows(0), m_data({})
{
	m_shape = { 0, 0 };
}

Matrix::Matrix(size_t elem_num)
{
	
}


Matrix::Matrix(const Matrix& matrix)/* : m_cols(matrix.m_cols), m_rows(matrix.m_rows), m_data(matrix.m_data)*/
{
	m_rows = matrix.m_rows;
	m_cols = matrix.m_cols;
	m_data = matrix.m_data;
	m_shape = { matrix.m_rows, matrix.m_cols };
}

Matrix&::Matrix::operator =(const Matrix& matrix)
{
	m_rows = matrix.m_rows;
	m_cols = matrix.m_cols;
	m_data = matrix.m_data;
	return *this;
}

Matrix&::Matrix::operator =(Matrix&& matrix)
{
	m_rows = matrix.m_rows;
	m_cols = matrix.m_cols;
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
			(*this)(i, j) *= scalar;
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
	out << "[";
	for (unsigned int i = 0; i < m.m_rows; i++)
	{
		out << "[ ";
		for (unsigned int j = 0; j < m.m_cols; j++)
		{
			out << m.m_data[i * m.m_cols + j] << " ";
		}
		out << "]\n";
	}
	out << "]\n";
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
	std::cout << "[";
	for (unsigned int i = 0; i < m_rows; i++)
	{
		std::cout << "[ ";
		for (unsigned int j = 0; j < m_cols; j++)
		{
			std::cout << m_data[i * m_cols + j] << " ";
		}
		std::cout << "]\n";
	}
	std::cout << "]\n";
}

Matrix Matrix::broadcast(const Matrix& input)
{
#ifdef _DEBUG
	if (m_rows != 1 && m_cols != 1)
		throw MatrixError("Broadcasting matrix with shape different from (1x, 1x)!");
#endif // _DEBUG
	// Check if the current matrix can be broadcasted to match the input matrix's shape
	if (m_rows == 1 && m_cols == input.m_cols) {
		// Broadcast rows
		Matrix result(input.m_rows, input.m_cols);
		for (size_t i = 0; i < input.m_rows; ++i) {
			for (size_t j = 0; j < input.m_cols; ++j) {
				result.m_data[i * input.m_cols + j] = m_data[j];
			}
		}
		return result;
	}
	if (m_cols == 1 && m_rows == input.m_rows) {
		// Broadcast columns
		Matrix result(input.m_rows, input.m_cols);
		for (size_t i = 0; i < input.m_rows; ++i) {
			for (size_t j = 0; j < input.m_cols; ++j) {
				result.m_data[i * input.m_cols + j] = m_data[i];
			}
		}
		return result;
	}
}

Matrix Matrix::mismatch_dim_subtract(const Matrix& left, const Matrix& right)
{
#ifdef _DEBUG
	if (left.m_rows < right.m_rows || left.m_cols < right.m_cols)
		throw MatrixError("This matrix dim is smaller than other!");
	if (left.m_cols != right.m_cols)
		throw MatrixError("Matrices cols number mismatch!");
#endif // _DEBUG
	Matrix result(left.m_rows, left.m_cols);
	for (size_t i = 0; i < left.m_rows; i++)
		for (size_t j = 0; j < left.m_cols; j++)
		{
			result(i, j) = left(i, j) - right(0, j);
		}
	return result;
}

Matrix Matrix::mismatch_dim_divide(const Matrix& left, const Matrix& right)
{
#ifdef _DEBUG
	if (left.m_rows < right.m_rows || left.m_cols < right.m_cols)
		throw MatrixError("This matrix dim is smaller than other!");
	if (left.m_cols != right.m_cols)
		throw MatrixError("Matrices cols number mismatch!");
#endif // _DEBUG
	Matrix result(left.m_rows, left.m_cols);
	for (size_t i = 0; i < left.m_rows; i++)
		for (size_t j = 0; j < left.m_cols; j++)
		{
			result(i, j) = left(i, j) / right(0, j);
		}
	return result;
}

Matrix Matrix::mismatch_dim_add(const Matrix& left, const Matrix& right)
{
#ifdef _DEBUG
	if (left.m_rows < right.m_rows || left.m_cols < right.m_cols)
		throw MatrixError("This matrix dim is smaller than other!");
#endif // _DEBUG
	Matrix result(std::max(left.m_rows, right.m_rows), std::max(left.m_cols, right.m_cols));
	if (right.m_cols == 1)
	{
		for (size_t i = 0; i < left.m_rows; i++)
			for (size_t j = 0; j < left.m_cols; j++)
			{
				result(i, j) = left(i, j) + right(i, 0);
			}
	}
	if (right.m_rows == 1)
	{
		for (size_t i = 0; i < left.m_rows; i++)
			for (size_t j = 0; j < left.m_cols; j++)
			{
				result(i, j) = left(i, j) + right(0, j);
			}
	}
	return result;
}

Matrix& Matrix::load_data_txt(const size_t rows, const size_t cols, const std::string& filepath)
{
	m_rows = rows, m_cols = cols;
	std::vector<double> data = {};
	std::ifstream file(filepath);
	if (!file.is_open())
	{
		std::cout << "Failed to open file " + filepath << "\n";
		return *this;
	}
	std::string line;
	while (std::getline(file, line))
	{
		std::istringstream iss(line);
		std::string temp;
		char sep = ',';
		while (std::getline(iss, temp, sep))
		{
			try
			{
				data.push_back(std::stod(temp));
			}
			catch (...)
			{
				std::cout << "Failed to parse " << temp << "\n";
			}
		}
	}
	m_data = std::move(data);
#ifdef _DEBUG
	if (m_data.size() != m_rows * m_cols)
	{
		throw MatrixError("Load data from string failed! Mismatch size!");
	}
#endif
	return *this;
}

Matrix& Matrix::load_data_csv(const size_t rows, const size_t cols, const std::string& filepath, const bool header)
{
	m_rows = rows, m_cols = cols;
	std::vector<double> data = {};
	std::ifstream file(filepath);
	if (!file.is_open())
	{
		std::cout << "Failed to open file " + filepath << "\n";
		return *this;
	}
	std::string line;
	if (header)
	{
		std::getline(file, line);
	}
	while (std::getline(file, line))
	{
		std::istringstream iss(line);
		std::string temp;
		char sep = ',';
		while (std::getline(iss, temp, sep))
		{
			try
			{
				data.push_back(std::stod(temp));
			}
			catch (...)
			{
				std::cout << "Failed to parse " << temp << "\n";
			}
		}
	}
	m_data = std::move(data);
#ifdef _DEBUG
	if (m_data.size() != m_rows * m_cols)
	{
		throw MatrixError("Load data from string failed! Mismatch size!");
	}
#endif
	return *this;
}

Matrix& Matrix::load_data(const std::string& file)
{
	std::ifstream inFile;
	inFile.open(file, std::ios::in | std::ios::binary);

	unsigned int rows, columns;
	inFile.read((char*)&rows, sizeof(rows));
	inFile.read((char*)&columns, sizeof(columns));
	m_rows = rows, m_cols = columns;
	double* m = new double[rows * columns];
	inFile.read((char*)m, sizeof(double) * rows * columns);
	std::vector<double> mat(m, m + rows * columns);
	m_data = std::move(mat);
	delete[] m;

	return *this;
}

void Matrix::save_data(const std::string& file)
{
	std::ofstream outFile;
	outFile.open(file, std::ios::binary);

	outFile.write((char*)(&m_rows), sizeof(m_rows));
	outFile.write((char*)(&m_cols), sizeof(m_cols));
	outFile.write((char*)&m_data[0], sizeof(double) * m_rows * m_cols);
}

void Matrix::shuffle(Matrix& X, Matrix& y)
{
#ifdef _DEBUG
	if (X.m_rows != y.m_rows)
	{
		throw MatrixError("Matrices have different rows!");
	}
#endif
	// Create a permutation vector
	std::vector<size_t> perm(X.m_rows);
	for (size_t i = 0; i < X.m_rows; ++i) {
		perm[i] = i;
	}

	// Shuffle the permutation vector
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(perm.begin(), perm.end(), g);

	// Apply the permutation to both matrices
	std::vector<double> tempX(X.m_data.size());
	std::vector<double> tempY(y.m_data.size());
	for (size_t i = 0; i < X.m_rows; ++i) {
		std::copy(X.m_data.begin() + perm[i] * X.m_cols, X.m_data.begin() + perm[i] * X.m_cols + X.m_cols, tempX.begin() + i * X.m_cols);
		std::copy(y.m_data.begin() + perm[i] * y.m_cols, y.m_data.begin() + perm[i] * y.m_cols + y.m_cols, tempY.begin() + i * y.m_cols);
	}

	X.m_data.swap(tempX);
	y.m_data.swap(tempY);
}

Matrix& Matrix::drop(const int& index, const int& type)
{
	// type: 0-row, 1-col
#ifdef _DEBUG
	if (index >= m_rows && type == 0)
	{
		throw MatrixError("Index exceed number of rows!");
	}
	if (index >= m_cols && type == 1)
	{
		throw MatrixError("Index exceed number of columns!");
	}
#endif
	if (type == 0)
	{
		// Remove the row from the data vector
		m_data.erase(m_data.begin() + index * m_cols, m_data.begin() + (index + 1) * m_cols);
		m_rows--;
	}
	if (type == 1)
	{
		// Create a temporary vector to hold the new data without the dropped column
		std::vector<double> newData;
		newData.reserve(m_data.size() - m_rows); // Reserve space for the new data
		// Copy all elements except those in the dropped column
		for (size_t i = 0; i < m_data.size(); ++i) {
			if (i % m_cols != index) { // Skip elements in the dropped column
				newData.push_back(m_data[i]);
			}
		}
		// Replace the old data with the new data
		m_data = std::move(newData);
		--m_cols;
	}
	return *this;
}

Matrix Matrix::extract(const int& index, const int& type)
{
	// type: 0-row, 1-col
#ifdef _DEBUG
	if (index > m_rows && type == 0)
	{
		throw MatrixError("Index exceed number of rows!");
	}
	if (index > m_cols && type == 1)
	{
		throw MatrixError("Index exceed number of columns!");
	}
#endif
	if (type == 0)
	{
		Matrix result(1, m_cols);
		std::copy(m_data.begin() + index * m_cols, m_data.begin() + index * m_cols + m_cols, result.m_data.begin());
		return result;
	}
	if (type == 1)
	{
		Matrix result(m_rows, 1);
		for (size_t i = 0; i < m_rows; ++i) {
			result.m_data[i] = m_data[i * m_cols + index];
		}
		return result;
	}
}

Matrix Matrix::extract(const int& start, const int& end, const int& type)
{
	// type: 0-row, 1-col
#ifdef _DEBUG
	if ((end > m_rows || start > m_rows) && type == 0)
	{
		throw MatrixError("Index exceed number of rows!");
	}
	if ((end > m_cols || start > m_rows) && type == 1)
	{
		throw MatrixError("Index exceed number of columns!");
	}
#endif
	if (type == 0)
	{
		Matrix result(end - start + 1, m_cols);
		for (size_t i = start; i <= end; i++)
			std::copy(m_data.begin() + i * m_cols, m_data.begin() + i * m_cols + m_cols, result.m_data.begin() + (i - start) * m_cols);
		return result;
	}
	if (type == 1)
	{
		Matrix result(m_rows, end - start + 1);
		for (size_t i = 0; i < m_rows; i++) {
			for (size_t j = start; j <= end; j++) {
				result.m_data[i * result.m_cols + (j - start)] = m_data[i * m_cols + j];
			}
		}
		return result;
	}
}

Matrix Matrix::argmax(const int& type)
{
	if (type == 0) 
	{ // find max along rows
		Matrix result(m_rows, 1);
		for (size_t row = 0; row < m_rows; ++row) {
			double maxElement = m_data[row * m_cols];
			size_t maxIndex = 0;
			for (size_t col = 1; col < m_cols; ++col) {
				if (m_data[row * m_cols + col] > maxElement) {
					maxElement = m_data[row * m_cols + col];
					maxIndex = col;
				}
			}
			result.m_data[row] = static_cast<double>(maxIndex);
		}
		return result;
	}
	if (type == 1)
	{ // find max along columns
		Matrix result(1, m_cols);
		for (size_t col = 0; col < m_cols; ++col) {
			double maxElement = m_data[col];
			size_t maxIndex = 0;
			for (size_t row = 1; row < m_rows; ++row) {
				if (m_data[row * m_cols + col] > maxElement) {
					maxElement = m_data[row * m_cols + col];
					maxIndex = row;
				}
			}
			result.m_data[col] = static_cast<double>(maxIndex);
		}
		return result;
	}
}

void Matrix::randomize(double min, double max)
{
	std::random_device rand;
	std::mt19937 engine(rand());
	std::uniform_real_distribution<double> valueDistribution(min, max);
	for (size_t i = 0; i < m_rows * m_cols; i++)
		m_data[i] = valueDistribution(engine);
}

void Matrix::soft_reset()
{
	std::fill(m_data.begin(), m_data.end(), 0);
}

Matrix Matrix::max()
{
	Matrix result(1, m_cols);
	for (size_t i = 0; i < m_cols; i++)
	{
		result(0, i) = (*this)(0, i);
		for (size_t j = 1; j < m_rows; j++)
			result(0, i) = std::max(result(0, i), (*this)(j, i));
	}
	return result;
}

Matrix Matrix::max() const
{
	Matrix result(1, m_cols);
	for (size_t i = 0; i < m_cols; i++)
	{
		result(0, i) = (*this)(0, i);
		for (size_t j = 1; j < m_rows; j++)
			result(0, i) = std::max(result(0, i), (*this)(j, i));
	}
	return result;
}

Matrix Matrix::sum(int axis)
{
	if (axis == -1)
	{
		Matrix result(1, 1);
		for (size_t i = 0; i < m_rows; i++)
		{
			for (size_t j = 0; j < m_cols; j++)
				result(0, 0) += (*this)(i, j);
		}
		return result;
	}
	if (axis == 0)
	{
		Matrix result(1, m_cols);
		for (size_t i = 0; i < m_cols; i++)
		{
			for (size_t j = 0; j < m_rows; j++)
				result(0, i) += (*this)(j, i);
		}
		return result;
	}
	if (axis == 1)
	{
		Matrix result(m_rows, 1);
		for (size_t i = 0; i < m_rows; i++)
		{
			for (size_t j = 0; j < m_cols; j++)
				result(i, 0) += (*this)(i, j);
		}
		return result;
	}
}

Matrix& Matrix::element_wise_mul(const Matrix& other)
{
#ifdef _DEBUG
	if (!check_dims(other))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	for (size_t i = 0; i < other.m_rows; i++)
		for (size_t j = 0; j < other.m_cols; j++)
			(*this)(i, j) = (*this)(i, j) * other(i, j);
	return *this;
}

Matrix& Matrix::square()
{
	return this->element_wise_mul(*this);
}

Matrix& Matrix::transpose()
{
	if (m_cols != 1 && m_rows != 1)
	{
		std::vector<double> transposeData(m_cols * m_rows);
		for (size_t i = 0; i < m_rows; i++)
			for (size_t j = 0; j < m_cols; j++)
				transposeData[j * m_rows + i] = (*this)(i, j);
		m_data = std::move(transposeData);
	}
	std::swap(m_rows, m_cols);
	return *this;
}

Matrix Matrix::element_wise_mul(const Matrix& left, const Matrix& right)
{
#ifdef _DEBUG
	if (!left.check_dims(right))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	Matrix result(left);
	for (size_t i = 0; i < left.m_rows; i++)
		for (size_t j = 0; j < left.m_cols; j++)
			result(i, j) = left(i, j) * right(i, j);
	return result;
}

Matrix Matrix::square(const Matrix& matrix)
{
	return element_wise_mul(matrix, matrix);
}

Matrix Matrix::transpose(const Matrix& matrix)
{
	Matrix transposed(matrix.m_cols, matrix.m_rows);
	if (matrix.m_cols != 1 && matrix.m_rows != 1)
	{
		for (size_t i = 0; i < matrix.m_rows; i++)
			for (size_t j = 0; j < matrix.m_cols; j++)
				transposed(j, i) = matrix(i, j);
	}
	return transposed;
}
