#pragma once
#include <chrono>
#include "matrix.h"

#define GET_NAME(variable)

int64_t getTickcount();

// activations
double relu(double x);
double d_relu(double x);
double sigmoid(double x);
double d_sigmoid(double x);
Matrix softmax(const Matrix& matrix);

// loss
Matrix cross_entropy_loss(const Matrix& y, const Matrix& yhat);