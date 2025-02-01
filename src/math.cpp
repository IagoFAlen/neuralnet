#include "math.hpp"

using namespace math;

// ReLU activation function
double math::relu(double x){
    return (x > 0.0) ? x : 0.0;
}

// Derivative of ReLU (used during backpropagation)
double math::relu_derivative(double x){
    return (x > 0.0) ? 1.0 : 0.0;
}

void softmax(LAYER* outputLayer){
    double sumExp = 0.0;

    for(NEURON* currentNeuron = outputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
        currentNeuron->activation = exp(currentNeuron->neuronValue);
        sumExp += currentNeuron->activation;
    }

    for(NEURON* currentNeuron = outputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
        currentNeuron->activation /= sumExp;
    }
}

void softmax_derivative(LAYER* outputLayer){
    for(NEURON* currentNeuron = outputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
        currentNeuron->deltaLoss = currentNeuron->activation - currentNeuron->target;
    }
}

