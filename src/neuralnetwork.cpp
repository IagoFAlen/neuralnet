#include <ctime>

#include "neuralnetwork.hpp"
#include "list.hpp"

using namespace std;
using namespace neuralnets;

void neuralnets::create_connection(unsigned int id, NEURON* backwardNeuron, double weight, NEURON* afterwardNeuron){
    CONNECTION* newConnection = new CONNECTION();
    
    newConnection->id = id;
    newConnection->backwardNeuron = backwardNeuron;
    newConnection->afterwardNeuron = afterwardNeuron;
    newConnection->weight = weight; // Use the passed weight
    newConnection->deltaWeight = 0.0;
    newConnection->next = NULL;

    // Case 1: First connection in the neuron
    if (backwardNeuron->connections == NULL) {
        backwardNeuron->connections = newConnection; // Update head
        backwardNeuron->lastConnection = newConnection; // Update tail
    }
    // Case 2: Append to existing connections
    else {
        backwardNeuron->lastConnection->next = newConnection; // Link last node to new node
        backwardNeuron->lastConnection = newConnection; // Update tail
    }

    // Add to incoming list of the destination neuron (afterwardNeuron)
    if (afterwardNeuron->previousConnections == NULL) {
        afterwardNeuron->previousConnections = newConnection;
        afterwardNeuron->lastPreviousConnection = newConnection;
    } else {
        afterwardNeuron->lastPreviousConnection->next = newConnection;
        afterwardNeuron->lastPreviousConnection = newConnection;
    }

}

// Create and initialize a neuron
NEURON* neuralnets::create_neuron(unsigned int id, double bias) {
    NEURON* newNeuron = new NEURON();
    newNeuron->id = id;
    newNeuron->activation = 0.0;
    newNeuron->bias = bias;
    newNeuron->deltaLoss = 0.0;
    newNeuron->connections = NULL;
    newNeuron->lastConnection = NULL;
    newNeuron->previousConnections = NULL;
    newNeuron->lastPreviousConnection = NULL;
    newNeuron->next = NULL;
    return newNeuron;
}

void neuralnets::add_neuron(LAYER* layer, unsigned int id, double bias) {
    if (!layer) return; // Check if layer exists

    NEURON* newNeuron = create_neuron(id, bias);

    if (layer->neurons == NULL) {
        // First neuron in the layer
        layer->neurons = newNeuron;
        layer->lastNeuron = newNeuron;
    } else {
        // Append to the end using lastNeuron
        layer->lastNeuron->next = newNeuron;
        layer->lastNeuron = newNeuron;
    }

    layer->numNeurons++; // Update neuron count
}

LAYER* neuralnets::create_layer(int num_neurons, unsigned int layer_id) {
    LAYER* new_layer = new LAYER();
    new_layer->id = layer_id;

    for (int i = 0; i < num_neurons; i++) {
        add_neuron(new_layer, i /*id*/, 0.0 /*bias*/);
    }

    return new_layer;
}

NEURAL_NETWORK* neuralnets::create_neural_network(unsigned int id, ds_list::LIST_INFO* layer_sizes_list, double learning_rate) {
    NEURAL_NETWORK* nn = new NEURAL_NETWORK();
    nn->id = id;
    nn->learningRate = learning_rate;
    nn->layersInfo = layer_sizes_list;

    if (nn->layersInfo->size == 0) return nn;

    LAYER* prev_layer = nullptr;

    // 1. Create layers using a for loop
    for (ds_list::BLOCK* current_block = nn->layersInfo->list; current_block != nullptr; current_block = current_block->next) {
        LAYER* layer = create_layer(current_block->value, current_block->index);

        // Link layers
        if (prev_layer != nullptr) {
            prev_layer->next = layer;
            layer->previous = prev_layer;
        }

        // Set input/output layer pointers
        if (current_block == nn->layersInfo->list) {
            nn->inputLayer = layer;
        }
        if (current_block->next == nullptr) {
            nn->outputLayer = layer;
        }

        prev_layer = layer;
    }

    // 2. Connect layers using another for loop
    for (LAYER* current_layer = nn->inputLayer; current_layer != nullptr && current_layer->next != nullptr; current_layer = current_layer->next) {
        connect_layers(current_layer, current_layer->next);
    }

    return nn;
}

void neuralnets::connect_layers(LAYER* current_Layer, LAYER* next_layer) {
    if (!current_Layer || !next_layer) return;

    // He initialization for ReLU: sqrt(6.0 / n_input_neurons)
    double he_scale = sqrt(6.0 / current_Layer->numNeurons);
    
    unsigned int conn_id = 0;

    for(NEURON* src_neuron = current_Layer->neurons; src_neuron != NULL; src_neuron = src_neuron->next){
        for(NEURON* dest_neuron = next_layer->neurons; dest_neuron != NULL; dest_neuron = dest_neuron->next){
            // Generate weight between -he_scale and he_scale
            double weight = ((double)rand() / RAND_MAX) * 2 * he_scale - he_scale;
            create_connection(conn_id++, src_neuron, weight, dest_neuron);
        }
    }
}