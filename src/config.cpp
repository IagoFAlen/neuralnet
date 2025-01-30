#include "config.hpp"
#include "neuralnetwork.hpp"

using namespace config;
using namespace neuralnets;

NEURAL_NETWORK* initialize(ds_list::LIST_INFO* list, unsigned int id, double learning_rate){
    NEURAL_NETWORK* newNeuralNetwork = create_neural_network(id, list, learning_rate);

    for(LAYER* currentLayer = newNeuralNetwork->inputLayer; currentLayer->next != NULL; currentLayer = currentLayer->next){
        connect_layers(currentLayer, currentLayer->next);
    }

    return newNeuralNetwork;
}
