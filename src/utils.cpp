#include "utils.hpp"
#include "neuralnetwork.hpp"

using namespace utils;
using namespace neuralnets;
using namespace std;

namespace utils{
    void separator(){
        cout << "\033[1;36m----------------------------------------------------------------------------------------------------------------\033[0m" << endl;
    }

    void print_nn_io(NEURAL_NETWORK* nn){
        separator();

        for (LAYER* currentLayer = nn->inputLayer; currentLayer != nullptr; currentLayer = currentLayer->next) {
            cout << "\033[1;34mLayer " << currentLayer->id << ":\033[0m" << endl;
            
            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next) {
                // Check if the neuron is from the output layer
                bool isOutputLayer = (currentLayer == nn->outputLayer);

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

                // Print connections
                for (CONNECTION* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next) {
                    cout << "\t\t\033[1;35mConnection " << currentConnection->id << ":\033[0m ("
                        << currentConnection->backwardNeuron->id << ") _ \033[1;36m" << currentConnection->weight << "\033[0m _ (" 
                        << currentConnection->afterwardNeuron->id << ")"
                        << " -> \033[1;31mWeight gradient:\033[0m " << currentConnection->deltaWeight << endl;
                }
            }
        }
    }

    void print_nn_io_previous(NEURAL_NETWORK* nn){
        separator();

        for (LAYER* currentLayer = nn->inputLayer; currentLayer!= nullptr; currentLayer = currentLayer->next) {
            cout << "\033[1;34mLayer " << currentLayer->id << ":\033[0m" << endl;

            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron!= nullptr; currentNeuron = currentNeuron->next) {
                bool isOutputLayer = (currentLayer == nn->outputLayer);

                if (isOutputLayer && currentNeuron->activation == currentNeuron->target) {
                    cout << "\t\033[1;32mNeuron " << currentNeuron->id << ":\033[0m "
                        << currentNeuron->activation << " - \033[1;32mtarget:\033[0m " << currentNeuron->target
                        << " - \033[1;33mbias:\033[0m " << currentNeuron->bias << endl;
                } else {
                    cout << "\t\033[1;32mNeuron " << currentNeuron->id << ":\033[0m "
                        << currentNeuron->activation << " - \033[1;31mtarget:\033[0m " << currentNeuron->target
                        << " - \033[1;33mbias:\033[0m " << currentNeuron->bias << endl;
                }

                // Check if there are previous connections before iterating
                if (currentNeuron->previousConnections!= nullptr) { // The crucial check
                    for (CONNECTION* currentConnection = currentNeuron->previousConnections; currentConnection!= nullptr; currentConnection = currentConnection->nextAsPrevious) {
                        cout << "\t\t\033[1;35mConnection " << currentConnection->id << ":\033[0m ("
                            << currentConnection->backwardNeuron->id << ") _ \033[1;36m" << currentConnection->weight << "\033[0m _ ("
                            << currentConnection->afterwardNeuron->id << ")"
                            << " -> \033[1;31mWeight gradient:\033[0m " << currentConnection->deltaWeight << endl;
                    }
                }
            }
        }
    }
}