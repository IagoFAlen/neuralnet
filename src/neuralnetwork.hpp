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
        Neuron* backwardNeuron;  // Pointer to the source neuron
        Neuron* afterwardNeuron; // Pointer to the destination neuron
        double weight;                  // Connection weight
        double deltaWeight;             // Weight update during backpropagation
        Connection* next;        // Next connection in the list

        Connection(){
            id = 0;
            backwardNeuron = NULL;
            afterwardNeuron = NULL;
            weight = 0.0;
            deltaWeight = 0.0;
            next = NULL;
        }

    } CONNECTION;

    // Neuron in a layer
    typedef struct Neuron {
        unsigned int id;
        double activation;              // Neuron's output (after activation)
        double bias;                    // Bias term
        double deltaLoss;               // Error gradient (delta)
        Connection* connections;          // Outgoing connections (to next layer)
        Connection* lastConnection;       // Tail of outgoing connections
        Connection* previousConnections;  // Incoming connections (from previous layer)
        Connection* lastPreviousConnection; // Tail of incoming connections
        Neuron* next;                     // Next neuron in the layer

        Neuron(){
            id = 0;
            activation = 0;
            bias = 0;
            deltaLoss = 0;
            connections = NULL;
            lastConnection = NULL;
            previousConnections = NULL;
            lastPreviousConnection = NULL;
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
        ds_list::LIST_INFO* layersInfo; // Contains the number of layers and number of neurons on each layer

        NeuralNetwork(){
            id = 0;
            inputLayer = NULL;
            outputLayer = NULL;
            learningRate = 0;
        }
        
    } NEURAL_NETWORK;

    void create_connection(unsigned int id, NEURON* backwardNeuron, double weight, NEURON* afterwardNeuron); 
    NEURON* create_neuron(unsigned int id = 0, double bias = 0.0);
    void add_neuron(LAYER* layer, unsigned int id, double bias);
    LAYER* create_layer(int num_neurons, unsigned int layer_id);
    void connect_layers(LAYER* prev_layer, LAYER* next_layer);
    NEURAL_NETWORK* create_neural_network(unsigned int id, ds_list::LIST_INFO* layer_sizes_list, double learning_rate);
} // namespace neuralnets

#endif // NEURALNETWORK_H