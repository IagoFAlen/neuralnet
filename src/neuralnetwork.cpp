#include <iomanip>
#include <ctime>

#include "neuralnetwork.hpp"
#include "list.hpp"
#include "math.hpp"
#include "config.hpp"
#include "utils.hpp"

using namespace std;
using namespace neuralnets;
using namespace math;
using namespace config;
using namespace utils;

namespace neuralnets {
    static int index = 0;
    void create_connection(NEURON* backwardNeuron, double weight, NEURON* afterwardNeuron){
        CONNECTION* newConnection = new CONNECTION();

        newConnection->id = afterwardNeuron->id;
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

    CONNECTION* find_connection(unsigned int connectionId, NEURON* backwardNeuron, NEURON* afterwardNeuron){
        CONNECTION* tmpConnection = NULL;

        for(CONNECTION* currentConnection = backwardNeuron->connections; currentConnection != NULL; currentConnection = currentConnection->next){
            if(currentConnection->id == connectionId && currentConnection->backwardNeuron->id == backwardNeuron->id && currentConnection->afterwardNeuron->id == afterwardNeuron->id){
                return currentConnection;
            }
        }

        utils::handle_error("Connection not found", 1);
        return tmpConnection;
    }

    CONNECTION* find_previous_connection(unsigned int connectionId, NEURON* backwardNeuron, NEURON* afterwardNeuron){
        CONNECTION* tmpConnection = NULL;

        for(CONNECTION* currentConnection = afterwardNeuron->connections; currentConnection != NULL; currentConnection = currentConnection->nextAsPrevious){
            if(currentConnection->id == connectionId && currentConnection->backwardNeuron->id == backwardNeuron->id && currentConnection->afterwardNeuron->id == afterwardNeuron->id){
                return currentConnection;
            }
        }

        utils::handle_error("Connection not found", 1);
        return tmpConnection;
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

        utils::handle_success("Neuron added");
    }

    NEURON* find_neuron(LAYER* layer, unsigned int neuronId){
        NEURON* tmpNeuron = NULL;

        for(NEURON* currentNeuron = layer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
            if(currentNeuron->id == neuronId)
                return currentNeuron;
        }

        utils::handle_warning("Could not take the neuron, returning NULL value");
        return tmpNeuron;
    }

    LAYER* create_layer(int num_neurons, unsigned int layer_id) {
        LAYER* new_layer = new LAYER();
        new_layer->id = layer_id;

        for (int i = 0; i < num_neurons; i++) {
            add_neuron(new_layer, i /*id*/, 0.0 /*bias*/);
        }

        utils::handle_success("Layer created.");
        return new_layer;
    }

    NEURAL_NETWORK* create_neural_network_base(unsigned int id, ds_list::LIST_INFO* layer_sizes_list, double learning_rate, double lambda, int epochs, bool render){
        NEURAL_NETWORK* nn = new NEURAL_NETWORK();
        nn->id = id;
        nn->learningRate = learning_rate;
        nn->lambda = lambda;
        nn->epochs = epochs;
        nn->layersInfo = layer_sizes_list;
        nn->render = render;

        if (nn->layersInfo->size == 0) return nn;

        LAYER* prev_layer = NULL;

        for (ds_list::BLOCK* current_block = nn->layersInfo->list; current_block != NULL; current_block = current_block->next) {
            LAYER* layer = create_layer(current_block->value, current_block->index);

            if (prev_layer != NULL) {
                prev_layer->next = layer;
                layer->previous = prev_layer;
            }

            if (current_block == nn->layersInfo->list) {
                nn->inputLayer = layer;
            }
            if (current_block->next == NULL) {
                nn->outputLayer = layer;
            }

            prev_layer = layer;

        }

        return nn;

        
    }

    void connect_layers(LAYER* currentLayer, LAYER* next_layer) {
        if (!currentLayer || !next_layer) return;

        double xavierScale = sqrt(2.0 / (currentLayer->numNeurons + next_layer->numNeurons));

        for (NEURON* src = currentLayer->neurons; src != NULL; src = src->next) {
            for (NEURON* dest = next_layer->neurons; dest != NULL; dest = dest->next) {
                double weight = math::normal_distribution(0.0, xavierScale);
                create_connection(src, weight, dest);
            }
        }
    }

    NEURAL_NETWORK* create_neural_network(unsigned int id, ds_list::LIST_INFO* layer_sizes_list, double learning_rate, double lambda, int epochs, bool render) {
        NEURAL_NETWORK* nn = create_neural_network_base(id, layer_sizes_list, learning_rate, lambda, epochs, render);
        
        for (LAYER* current_layer = nn->inputLayer; current_layer != NULL && current_layer->next != NULL; current_layer = current_layer->next) {
            connect_layers(current_layer, current_layer->next);
        }

        return nn;
    }


    NEURAL_NETWORK* load_neural_network(unsigned int id, ds_list::LIST_INFO* layer_sizes_list, double learning_rate, double lambda, int epochs, bool render) {
        NEURAL_NETWORK* nn = create_neural_network_base(id, layer_sizes_list, learning_rate, lambda, epochs, render);
        
        return nn;
    }

    void feed_forward(NEURAL_NETWORK* nn) {
        for (LAYER* currentLayer = nn->inputLayer->next; currentLayer != NULL; currentLayer = currentLayer->next) {
            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next) {
                currentNeuron->neuronValue = 0.00;

                double test_sum = 0.0;
                if (currentNeuron->previousConnections != NULL) {
                    for (CONNECTION* currentConnection = currentNeuron->previousConnections; currentConnection != NULL; currentConnection = currentConnection->nextAsPrevious){
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
        for (NEURON* currentNeuron = nn->outputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next) {
            lossFunction += currentNeuron->target * log(currentNeuron->activation + 1e-9); // Add 1e-9 to avoid log(0)
        }
        lossFunction = -lossFunction;

        double l2_penalty = 0.0;
        for (LAYER* currentLayer = nn->inputLayer; currentLayer != NULL; currentLayer = currentLayer->next) {
            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next) {
                for (CONNECTION* currentConnection = currentNeuron->connections; currentConnection != NULL; currentConnection = currentConnection->next) {
                    l2_penalty += currentConnection->weight * currentConnection->weight;
                }
            }
        }
        l2_penalty *= (nn->lambda / 2.0);

        // Total loss = cross-entropy loss + L2 regularization
        nn->lossFunction = lossFunction + l2_penalty;
    }

    void track_output_layer_errors(NEURAL_NETWORK* nn) {
        for (NEURON* currentNeuron = nn->outputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next) {
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
        for (LAYER* currentLayer = nn->inputLayer; currentLayer != NULL; currentLayer = currentLayer->next) {
            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next) {
                for (CONNECTION* currentConnection = currentNeuron->connections; currentConnection != NULL; currentConnection = currentConnection->next) {
                    double gradient = currentConnection->afterwardNeuron->deltaLoss * currentNeuron->activation;

                    gradient += nn->lambda * currentConnection->weight;

                    double clipped_gradient = clip_gradient(gradient, -10.0, 10.0);

                    currentConnection->weight -= nn->learningRate * clipped_gradient;
                }

                double bias_gradient = currentNeuron->deltaLoss;

                double clipped_bias_gradient = clip_gradient(bias_gradient, -5.0, 5.0);

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

    void predict(NEURAL_NETWORK* nn){
        double prediction = nn->outputLayer->neurons->activation;
        unsigned int predictionIndex = nn->outputLayer->neurons->id;

        double targetIndex = nn->outputLayer->neurons->id;

        for(NEURON* currentNeuron = nn->outputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
            if(currentNeuron->target == 1)
                targetIndex = currentNeuron->id;

            if(prediction < currentNeuron->activation){
                prediction = currentNeuron->activation;
                predictionIndex = currentNeuron->id;
            }
        }

        config::save_predictions(targetIndex, predictionIndex);
    }
}
