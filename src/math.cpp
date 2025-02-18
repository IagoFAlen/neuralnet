#include <stdlib.h>
#include <math.h>
#include <time.h>
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

    // Leaky ReLU activation function (alpha = 0.01 for small negative slope)
    double leaky_relu(double x, double alpha){
        return (x > 0) ? x : alpha * x;
    }

    // Derivative of Leaky ReLU
    double leaky_relu_derivative(double x, double alpha){
        return (x > 0) ? 1.0 : alpha;
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

    double normal_distribution(double mean, double stddev) {
        static int has_spare = 0;
        static double spare;

        if (has_spare) {
            has_spare = 0;
            return mean + stddev * spare;
        }

        has_spare = 1;
        static double u, v, s;
        do {
            u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
            v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);

        s = sqrt(-2.0 * log(s) / s);
        spare = v * s;
        return mean + stddev * u * s;
    }
}
