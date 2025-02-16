#pragma once

#include <iostream>
#include <cstring>
#include "neuralnetwork.hpp"
#include "list.hpp"
#include "file_list.hpp"

using namespace std;
using namespace neuralnets;
using namespace ds_list;
using namespace file_list;

namespace config {
    NEURAL_NETWORK* initialize(unsigned int id, LIST_INFO* numNeuronsPerLayerList, int argc, char *argv[], double learning_rate);
    void initialize_neurons(NEURAL_NETWORK* nn, LIST_INFO* inputList, LIST_INFO* targetList, string line);
    void save_loss_function(double lossFunction, char type);
    void train_with_epochs(NEURAL_NETWORK* nn, string filePath, int epochs);
    void train_with_epochs_randomly(NEURAL_NETWORK* nn, string filePath, int epochs);
    void classify(NEURAL_NETWORK* nn, string filePath);
    void remove_file();
}
