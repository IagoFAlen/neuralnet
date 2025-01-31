#include <iostream>
#include <cstring>
#include "neuralnetwork.hpp"
#include "list.hpp"

using namespace std;
using namespace neuralnets;
using namespace ds_list;

namespace config {
    NEURAL_NETWORK* initialize(unsigned int id, LIST_INFO* numNeuronsPerLayerList, int argc, char *argv[], double learning_rate);
    void config::initialize_neurons(NEURAL_NETWORK* nn, LIST_INFO* inputList, LIST_INFO* targetList, string line);
    void config::train(NEURAL_NETWORK* nn, string filePath);
}
