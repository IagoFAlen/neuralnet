#include "math.hpp"

using namespace math;

namespace math {
    // ReLU activation function
    double relu(double x){
        return (x > 0.0) ? x : 0.0;
    }

    // Derivative of ReLU (used during backpropagation)
    double relu_derivative(double x){
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

    double softmax_derivative(double activation){
        return activation * (1.0 - activation);
    }
}
