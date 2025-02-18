#pragma once
#include "neuralnetwork.hpp"

using namespace neuralnets;

namespace math{
    double relu(double x);
    double relu_derivative(double x);
    void softmax(LAYER* outputLayer);
    double softmax_derivative(double activation);
    double normal_distribution(double mean, double stddev);
    double leaky_relu(double x, double alpha = 0.01);
    double leaky_relu_derivative(double x, double alpha = 0.01);
}