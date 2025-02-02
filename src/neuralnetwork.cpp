#include <iomanip>
#include <ctime>

#include "neuralnetwork.hpp"
#include "list.hpp"
#include "math.hpp"

using namespace std;
using namespace neuralnets;
using namespace math;

namespace neuralnets {
    void create_connection(unsigned int id, NEURON* backwardNeuron, double weight, NEURON* afterwardNeuron) {
        CONNECTION* newConnection = new CONNECTION();
        
        // Initialize connection properties
        newConnection->id = id;
        newConnection->backwardNeuron = backwardNeuron;
        newConnection->afterwardNeuron = afterwardNeuron;
        newConnection->weight = weight;
    
        // Add to backwardNeuron's outgoing connections
        if (backwardNeuron->connections == nullptr) {
            backwardNeuron->connections = newConnection;
            backwardNeuron->connections->lastConnection = newConnection;
        } else {
            CONNECTION* current = backwardNeuron->connections->lastConnection;
            current->next = newConnection;
            backwardNeuron->connections->lastConnection = newConnection;
        }

        afterwardNeuron->previousConnections = backwardNeuron->connections;
    }


    // Create and initialize a neuron
    NEURON* create_neuron(unsigned int id, double bias) {
        NEURON* newNeuron = new NEURON();
        newNeuron->id = id;
        newNeuron->activation = 0.0;
        newNeuron->bias = bias;
        newNeuron->deltaLoss = 0.0;
        newNeuron->connections = NULL;
        newNeuron->previousConnections = NULL;
        newNeuron->next = NULL;
        return newNeuron;
    }

    void add_neuron(LAYER* layer, unsigned int id, double bias) {
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

    LAYER* create_layer(int num_neurons, unsigned int layer_id) {
        LAYER* new_layer = new LAYER();
        new_layer->id = layer_id;

        for (int i = 0; i < num_neurons; i++) {
            add_neuron(new_layer, i /*id*/, 0.0 /*bias*/);
        }

        return new_layer;
    }

    NEURAL_NETWORK* create_neural_network(unsigned int id, ds_list::LIST_INFO* layer_sizes_list, double learning_rate) {
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

    void connect_layers(LAYER* currentLayer, LAYER* next_layer) {
        if (!currentLayer || !next_layer) return;

        double he_scale = sqrt(6.0 / currentLayer->numNeurons);

        for (NEURON* src = currentLayer->neurons; src != nullptr; src = src->next) {
            unsigned int conn_id = 0; // Reset ID for each source neuron
            for (NEURON* dest = next_layer->neurons; dest != nullptr; dest = dest->next) {
                double weight = ((double)rand() / RAND_MAX) * 2 * he_scale - he_scale;
                create_connection(conn_id++, src, weight, dest);
            }
        }
    }


    void feed_forward(NEURAL_NETWORK* nn){
        for(LAYER* currentLayer = nn->inputLayer->next; currentLayer != NULL; currentLayer = currentLayer->next){
            for(NEURON* currentNeuron = currentLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
                currentNeuron->neuronValue = 0;

                for(CONNECTION* currentConnection = currentNeuron->previousConnections; currentConnection != NULL; currentConnection = currentConnection->next){
                    currentNeuron->neuronValue += (currentConnection->backwardNeuron->activation * currentConnection->weight);
                }

                currentNeuron->neuronValue += currentNeuron->bias;

                // Apply ReLU to neuron
                if(currentLayer != nn->outputLayer)
                    currentNeuron->activation = math::relu(currentNeuron->neuronValue);
            }
        }

        math::softmax(nn->outputLayer);
    }

    void print_nn_io(NEURAL_NETWORK* nn){
        cout << "\033[1;36m----------------------------------------------------------------------------------------------------------------\033[0m" << endl;

        for (LAYER* currentLayer = nn->inputLayer; currentLayer != nullptr; currentLayer = currentLayer->next) {
            cout << "\033[1;34mLayer " << currentLayer->id << ":\033[0m" << endl;

            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next) {
                // Check if the neuron is from the output layer
                bool isOutputLayer = (currentLayer == nn->outputLayer);

                // Set precision for output
                cout << fixed << setprecision(5); // 5 digits after the decimal

                // If it's from the output layer, compare target and activation
                if (isOutputLayer && currentNeuron->activation == currentNeuron->target) {
                    // Change target color to green if it matches the activation
                    cout << "\t\033[1;32mNeuron " << currentNeuron->id << ":\033[0m " 
                        << currentNeuron->activation << " - \033[1;32mtarget:\033[0m " << currentNeuron->target
                        << " - \033[1;33mbias:\033[0m " << currentNeuron->bias << endl;
                } else {
                    // Default color for target when it's not equal to activation
                    cout << "\t\033[1;32mNeuron " << currentNeuron->id << ":\033[0m " 
                        << currentNeuron->activation << " - \033[1;31mtarget:\033[0m " << currentNeuron->target
                        << " - \033[1;33mbias:\033[0m " << currentNeuron->bias << endl;
                }

                // Print connections (no change needed here)
                for (CONNECTION* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next) {
                    cout << "\t\t\033[1;35mConnection " << currentConnection->id << ":\033[0m ("
                        << currentConnection->backwardNeuron->id << ") _ \033[1;36m" << currentConnection->weight << "\033[0m _ (" 
                        << currentConnection->afterwardNeuron->id << ")"
                        << " -> \033[1;31mWeight gradient:\033[0m " << currentConnection->deltaWeight << endl;
                }
            }
        }
        cout << defaultfloat; // Reset to default floating-point format after printing
    }
}
