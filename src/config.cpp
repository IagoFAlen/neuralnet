#include <fstream>
#include <cstring>
#include <sstream>

#include "config.hpp"
#include "neuralnetwork.hpp"
#include "utils.hpp"

using namespace std;
using namespace config;
using namespace neuralnets;
using namespace utils;

namespace config {
    NEURAL_NETWORK* initialize(unsigned int id, LIST_INFO* numNeuronsPerLayerList, int argc, char *argv[], double learning_rate){
        int num_layers = 0;

        // Taking the parameters and setting it into layer list informations
        for(int i = 1; i < argc; i++) {
            if(strcmp(argv[i], "-l") == 0) {
                num_layers = atoi(argv[i+1]);
                i++;
            } else if(strcmp(argv[i], "-n") == 0) {
                for(int j = 0; j < num_layers; j++) {
                    ds_list::push(numNeuronsPerLayerList, atoi(argv[i+j+1]));
                }
                i += num_layers;
            }
        }

        NEURAL_NETWORK* newNeuralNetwork = create_neural_network(id, numNeuronsPerLayerList, learning_rate);

        return newNeuralNetwork;
    }

    void initialize_neurons(NEURAL_NETWORK* nn, LIST_INFO* inputList, LIST_INFO* targetList, string line){
        string tempStringTrain = "";
        double trainData;

        stringstream inputString(line);
        for(int i = 0; i < (nn->layersInfo->list->value + nn->layersInfo->lastBlock->value); i++){
            getline(inputString, tempStringTrain, ',');
            trainData = atof(tempStringTrain.c_str());
            if(i < nn->layersInfo->list->value)
                push(inputList, trainData);
            else{
                push(targetList, trainData);
            }

            string tempStringTrain = "";
        }
        
        int inputIndex = 0;

        for(NEURON* currentNeuron = nn->inputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
            double normalizedInput = (get_value_by_index(inputList->list, inputIndex) / 100.00);
            currentNeuron->neuronValue = normalizedInput;
            currentNeuron->activation = currentNeuron->neuronValue;
            inputIndex++;
        }

        int targetIndex = 0;
        for(NEURON* currentNeuron = nn->outputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
            currentNeuron->target = get_value_by_index(targetList->list, targetIndex);
            targetIndex++;
        }

        //utils::print_nn_io(nn);

        empty(inputList);
        empty(targetList);

    }

    void train(NEURAL_NETWORK* nn, string filePath){
        string line = "";
        LIST_INFO* inputList = new LIST_INFO();
        LIST_INFO* targetList = new LIST_INFO();

        ifstream trainFile;
        trainFile.open(filePath);

        if (!trainFile.is_open()) {
            cout << "Error opening file: " << filePath << endl;
            return;
        }

        while(getline(trainFile, line)){
            cout << "Read line: " << line << endl;
            initialize_neurons(nn, inputList, targetList, line);
            neuralnets::feed_forward(nn);
            utils::print_nn_io(nn);
            //utils::separator();
            //utils::print_nn_io_previous(nn);
            neuralnets::backpropagation(nn);
            //utils::print_nn_io(nn);
            cout << "Loss Function: " << nn->lossFunction << endl;
        }
    }
}