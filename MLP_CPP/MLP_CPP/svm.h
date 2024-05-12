#pragma once
#include "matrix.h"

class SoftMarginSVM
{
	double constant;
	Matrix weight;
	double bias;

	double lr;
	int epochs;

	std::string name;
public:
	SoftMarginSVM(double constant, double learning_rate, int epochs, std::string name);

	Matrix cost(const Matrix& margins);
	void fit(const Matrix& x, const Matrix& y);
	Matrix predict(const Matrix& x);
	double eval(const Matrix& x, const Matrix& y);
	void print_eval(const Matrix& x, const Matrix& y);

	void load_model(const std::string& path);
	void save_model(const std::string& path);
};
