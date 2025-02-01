#pragma once
#include "neuralnetwork.hpp"

using namespace neuralnets;

namespace math{
    double relu(double x);
    double relu_derivative(double x);
    void softmax(LAYER* outputLayer);
    void softmax_derivative(LAYER* outputLayer);
}