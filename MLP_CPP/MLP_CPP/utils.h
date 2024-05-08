#pragma once
#include <chrono>
#include "matrix.h"
#include "nn.h"

#define GET_NAME(variable)

int64_t getTickcount();

// activations
double relu(double x);
double d_relu(double x);
double sigmoid(double x);
double d_sigmoid(double x);
Matrix softmax(const Matrix& matrix);
Matrix grad_softmax(Matrix error, Matrix yhat);

// loss
Matrix cross_entropy_loss(const Matrix& y, const Matrix& yhat);
Matrix grad_cross_entropy_loss(const Matrix& y, const Matrix& yhat);
double mse_loss(const Matrix& y, const Matrix& yhat);
Matrix grad_mse_loss(const Matrix& y, const Matrix& yhat);

// debugging
void save_snapshot_to_file(const parameters& params, const net_result& net_output, const std::string& directory);

// model utils
void save_model_to_file(parameters& params, const std::string& filePath);
void load_model_from_file(parameters& params, const std::string& filepath);
void save_svm_to_file(Matrix weight, double bias, const std::string& filepath);
void load_svm_from_file(Matrix& weight, double& bias, const std::string& filepath);

// evaluation utils
void save_vector_to_file(const std::vector<double>& losses, const std::string& filePath);
double get_precision(const Matrix& ypred, const Matrix& y);
double get_recall(const Matrix& ypred, const Matrix& y);
double get_f1(const Matrix& ypred, const Matrix& y);
//std::unordered_map<int, double> get_precision(Matrix ypred, Matrix y);
//std::unordered_map<int, double> get_recall(Matrix ypred, Matrix y);
//std::unordered_map<int, double> get_f1(Matrix ypred, Matrix y);