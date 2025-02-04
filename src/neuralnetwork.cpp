#include <iomanip>
#include <ctime>

#include "neuralnetwork.hpp"
#include "list.hpp"
#include "math.hpp"

using namespace std;
using namespace neuralnets;
using namespace math;

namespace neuralnets {
    static int index = 0;
    void create_connection(unsigned int id, NEURON* backwardNeuron, double weight, NEURON* afterwardNeuron){
        CONNECTION* newConnection = new CONNECTION();

        newConnection->id = id;
        newConnection->backwardNeuron = backwardNeuron;
        newConnection->weight = weight;
        newConnection->afterwardNeuron = afterwardNeuron;

        if(!backwardNeuron->connections){
            backwardNeuron->connections = newConnection;
            backwardNeuron->connections->lastConnection = newConnection;
        }else{
            backwardNeuron->connections->lastConnection->next = newConnection;
            backwardNeuron->connections->lastConnection = newConnection;
        }

        // Link to afterward neuron's previous (incoming) connections
        if (!afterwardNeuron->previousConnections) {
            afterwardNeuron->previousConnections = newConnection;
            newConnection->lastConnectionAsPrevious = newConnection; // <-- KEY FIX
        } else {
            // Get the current tail of the incoming connections list
            afterwardNeuron->previousConnections->lastConnectionAsPrevious->nextAsPrevious = newConnection;
            afterwardNeuron->previousConnections->lastConnectionAsPrevious = newConnection; // Update tail
        }

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


    void feed_forward(NEURAL_NETWORK* nn) {
        for (LAYER* currentLayer = nn->inputLayer->next; currentLayer != NULL; currentLayer = currentLayer->next) {
            cout << "STARTING FROM LAYER " << currentLayer->id << endl;
            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next) {
                cout << "\tSTARTING FROM NEURON " << currentNeuron->id << endl;
                currentNeuron->neuronValue = 0.00;

                double test_sum = 0.0;
                if (currentNeuron->previousConnections != NULL) {
                    for (CONNECTION* currentConnection = currentNeuron->previousConnections; currentConnection != nullptr; currentConnection = currentConnection->nextAsPrevious){
                        cout << "\t\tSTARTING FROM CONNECTION " << currentConnection->id << endl;
                        double multiplication = (currentConnection->backwardNeuron->activation * currentConnection->weight);
                        test_sum += multiplication;
                        cout << "\t\t\tAdding to afterward neuron SUM(" << currentConnection->afterwardNeuron->id << ") - backNeuron(" << currentConnection->backwardNeuron->activation << "): "
                            << currentConnection->backwardNeuron->activation
                            << " x " << currentConnection->weight << " = " << multiplication << endl;
                        
                        cout << "\t\t\t\tThen the test_sum should be: " << test_sum << endl;
                    }
                }
                cout << endl;
                currentNeuron->neuronValue = test_sum;
                currentNeuron->neuronValue += currentNeuron->bias;
                currentNeuron->activation = math::relu(currentNeuron->neuronValue);

            }
            cout << endl << endl;
        }

        math::softmax(nn->outputLayer);
    }

}
