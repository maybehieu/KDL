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