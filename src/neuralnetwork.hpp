#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "list.hpp"

namespace neuralnets {

    // STRUCTS DECLARATIONS
    struct Connection;
    struct Neuron;
    struct Layer;
    struct NeuralNetwork;

    // Connection between two neurons
    typedef struct Connection {
        unsigned int id;
        Neuron* backwardNeuron;                 // Pointer to the source neuron
        Neuron* afterwardNeuron;                // Pointer to the destination neuron
        double weight;                          // Connection weight
        Connection* next;                       // Next connection in the list
        Connection* lastConnection;             // Pointer to the lastConnection from neuron
        Connection* nextAsPrevious;             // Pointer to the connections from the afterward Neuron
        Connection* lastConnectionAsPrevious;   // Pointer to the last connection from the afterward Neuron

        Connection(){
            id = 0;
            backwardNeuron = NULL;
            afterwardNeuron = NULL;
            weight = 0.0;
            next = NULL;
            lastConnection = NULL;
            nextAsPrevious = NULL;
            lastConnection = NULL;
            lastConnectionAsPrevious = NULL;
        }

    } CONNECTION;

    // Neuron in a layer
    typedef struct Neuron {
        unsigned int id;
        double neuronValue;                 // Neuron's output (before activation)
        double activation;                  // Neuron's output (after activation)
        double bias;                        // Bias term
        double deltaLoss;                   // Error gradient (delta)
        double target;                      // Target Value
        Connection* connections;            // Outgoing connections (to next layer)
        Connection* previousConnections;    // Connections linked to the previous layer
        Neuron* next;                       // Next neuron in the layer

        Neuron(){
            id = 0;
            neuronValue = 0;
            activation = 0;
            bias = 0;
            deltaLoss = 0;
            target = -1.0;
            connections = NULL;
            previousConnections = NULL;
            next = NULL;
        }

    } NEURON;

    // Layer in the network
    typedef struct Layer {
        unsigned int id;
        int numNeurons;                 // Number of neurons in this layer
        NEURON* neurons;                // Head of the neuron list
        NEURON* lastNeuron;             // Tail of the neuron list (NEW)
        Layer* next;                    // Next layer in the network
        Layer* previous;                // Previous layer in the network

        Layer(){
            id = 0;
            numNeurons = 0;
            neurons = NULL;
            next = NULL;
            previous = NULL;
        }

    } LAYER;

    // Neural network
    typedef struct NeuralNetwork {
        unsigned int id;
        LAYER* inputLayer;              // Pointer to the input layer
        LAYER* outputLayer;             // Pointer to the output layer
        double learningRate;            // Learning rate for gradient descent
        double lossFunction;            // Cross Entropy Loss Function
        ds_list::LIST_INFO* layersInfo; // Contains the number of layers and number of neurons on each layer

        NeuralNetwork(){
            id = 0;
            inputLayer = NULL;
            outputLayer = NULL;
            learningRate = 0;
            lossFunction = 0;
        }
        
    } NEURAL_NETWORK;

    void create_connection(unsigned int id, NEURON* backwardNeuron, double weight, NEURON* afterwardNeuron); 
    NEURON* create_neuron(unsigned int id = 0, double bias = 0.0);
    void add_neuron(LAYER* layer, unsigned int id, double bias);
    LAYER* create_layer(int num_neurons, unsigned int layer_id);
    void connect_layers(LAYER* prev_layer, LAYER* next_layer);
    NEURAL_NETWORK* create_neural_network(unsigned int id, ds_list::LIST_INFO* layer_sizes_list, double learning_rate);
    void feed_forward(NEURAL_NETWORK* nn);
    void loss_function(NEURAL_NETWORK* nn);
    void track_output_layer_errors(NEURAL_NETWORK* nn);
    void propagate_error(NEURAL_NETWORK* nn);
    void update_weights_and_biases(NEURAL_NETWORK* nn);
    void backpropagation(NEURAL_NETWORK* nn);

}

#endif // NEURALNETWORK_H