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
    NEURAL_NETWORK* initialize(unsigned int id, LIST_INFO* numNeuronsPerLayerList, int argc, char *argv[]);
    void initialize_neurons(NEURAL_NETWORK* nn, LIST_INFO* inputList, LIST_INFO* targetList, string line);
    void save_loss_function(double lossFunction, char type);
    void save_predictions(double targetIndex, double predictionIndex);
    void train_with_epochs(NEURAL_NETWORK* nn, string filePath, bool saving_mode);
    void train_with_epochs_randomly(NEURAL_NETWORK* nn, string filePath, bool saving_mode);
    void classify(NEURAL_NETWORK* nn, string filePath);
    void predict(NEURAL_NETWORK* nn, string filePath);
    void remove_file_train();
    void remove_file_predict();
    void save_neural_network(NEURAL_NETWORK* nn, const string& filePath);
    NEURAL_NETWORK* parse_neural_network(const string& filePath);
}
