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
            newConnection->lastConnectionAsPrevious = newConnection;
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

    NEURAL_NETWORK* create_neural_network(unsigned int id, ds_list::LIST_INFO* layer_sizes_list, double learning_rate, double lambda, int epochs) {
        NEURAL_NETWORK* nn = new NEURAL_NETWORK();
        nn->id = id;
        nn->learningRate = learning_rate;
        nn->lambda = lambda;
        nn->epochs = epochs;
        nn->layersInfo = layer_sizes_list;

        if (nn->layersInfo->size == 0) return nn;

        LAYER* prev_layer = nullptr;

        for (ds_list::BLOCK* current_block = nn->layersInfo->list; current_block != nullptr; current_block = current_block->next) {
            LAYER* layer = create_layer(current_block->value, current_block->index);

            if (prev_layer != nullptr) {
                prev_layer->next = layer;
                layer->previous = prev_layer;
            }

            if (current_block == nn->layersInfo->list) {
                nn->inputLayer = layer;
            }
            if (current_block->next == nullptr) {
                nn->outputLayer = layer;
            }

            prev_layer = layer;
        }

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
                double weight = math::normal_distribution(0.0, sqrt(2.0 / currentLayer->numNeurons));
                create_connection(conn_id++, src, weight, dest);
            }
        }
    }


    void feed_forward(NEURAL_NETWORK* nn) {
        for (LAYER* currentLayer = nn->inputLayer->next; currentLayer != NULL; currentLayer = currentLayer->next) {
            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next) {
                currentNeuron->neuronValue = 0.00;

                double test_sum = 0.0;
                if (currentNeuron->previousConnections != NULL) {
                    for (CONNECTION* currentConnection = currentNeuron->previousConnections; currentConnection != nullptr; currentConnection = currentConnection->nextAsPrevious){
                        double multiplication = (currentConnection->backwardNeuron->activation * currentConnection->weight);
                        test_sum += multiplication;                        
                    }
                }
                
                currentNeuron->neuronValue = test_sum;
                currentNeuron->neuronValue += currentNeuron->bias;
                //currentNeuron->activation = math::relu(currentNeuron->neuronValue);
                currentNeuron->activation = math::leaky_relu(currentNeuron->neuronValue);
            }
        }

        math::softmax(nn->outputLayer);
    }

    void loss_function(NEURAL_NETWORK* nn) {
        double lossFunction = 0.00;
        for (NEURON* currentNeuron = nn->outputLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next) {
            lossFunction += currentNeuron->target * log(currentNeuron->activation + 1e-9); // Add 1e-9 to avoid log(0)
        }
        lossFunction = -lossFunction;

        double l2_penalty = 0.0;
        for (LAYER* currentLayer = nn->inputLayer; currentLayer != nullptr; currentLayer = currentLayer->next) {
            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next) {
                for (CONNECTION* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next) {
                    l2_penalty += currentConnection->weight * currentConnection->weight;
                }
            }
        }
        l2_penalty *= (nn->lambda / 2.0);

        // Total loss = cross-entropy loss + L2 regularization
        nn->lossFunction = lossFunction + l2_penalty;
    }

    void track_output_layer_errors(NEURAL_NETWORK* nn) {
        for (NEURON* currentNeuron = nn->outputLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next) {
            currentNeuron->deltaLoss = currentNeuron->activation - currentNeuron->target;
        }
    }

    void propagate_error(NEURAL_NETWORK* nn){
        for(LAYER* currentLayer = nn->outputLayer->previous; currentLayer != NULL; currentLayer = currentLayer->previous){
            for(NEURON* currentNeuron = currentLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
                double sum = 0.0;
                for(CONNECTION* currentConnection = currentNeuron->connections; currentConnection != NULL; currentConnection = currentConnection->next){
                    sum += currentConnection->weight * currentConnection->afterwardNeuron->deltaLoss;
                }
                //currentNeuron->deltaLoss = sum * math::relu_derivative(currentNeuron->activation);
                currentNeuron->deltaLoss = sum * math::leaky_relu_derivative(currentNeuron->activation);
            }
        }
    }

    double clip_gradient(double gradient, double min_value, double max_value) {
        if (gradient < min_value) {
            return min_value;
        } else if (gradient > max_value) {
            return max_value;
        } else {
            return gradient;
        }
    }

    void update_weights_and_biases(NEURAL_NETWORK* nn) {
        for (LAYER* currentLayer = nn->inputLayer; currentLayer != nullptr; currentLayer = currentLayer->next) {
            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next) {
                for (CONNECTION* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next) {
                    double gradient = currentConnection->afterwardNeuron->deltaLoss * currentNeuron->activation;

                    gradient += nn->lambda * currentConnection->weight;

                    double clipped_gradient = clip_gradient(gradient, -1.0, 1.0);

                    currentConnection->weight -= nn->learningRate * clipped_gradient;
                }

                double bias_gradient = currentNeuron->deltaLoss;

                double clipped_bias_gradient = clip_gradient(bias_gradient, -1.0, 1.0);

                currentNeuron->bias -= nn->learningRate * clipped_bias_gradient;
            }
        }
    }

    void backpropagation(NEURAL_NETWORK* nn){
        loss_function(nn);
        track_output_layer_errors(nn);
        propagate_error(nn);
        update_weights_and_biases(nn);
    }

}
