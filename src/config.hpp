#pragma once

#include <iostream>
#include <cstring>
#include "neuralnetwork.hpp"
#include "list.hpp"

using namespace std;
using namespace neuralnets;
using namespace ds_list;

namespace config {
    NEURAL_NETWORK* initialize(unsigned int id, LIST_INFO* numNeuronsPerLayerList, int argc, char *argv[], double learning_rate);
    void initialize_neurons(NEURAL_NETWORK* nn, LIST_INFO* inputList, LIST_INFO* targetList, string line);
    void save_loss_function(double lossFunction);
    void train(NEURAL_NETWORK* nn, string filePath);
}
