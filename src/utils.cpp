#include <iomanip>
#include "utils.hpp"
#include "neuralnetwork.hpp"

using namespace utils;
using namespace neuralnets;
using namespace std;

/* COLORS */
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

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
                        << currentConnection->afterwardNeuron->id << ")" << endl;
                }
            }
        }
    }

    void print_nn_io_previous(NEURAL_NETWORK* nn){
        separator();

        //cout << fixed << setprecision(8);

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
                            << currentConnection->afterwardNeuron->id << ")" << endl;
                    }
                }
            }
        }
    }

    void print_train(int epoch, int epochs){
        if(epoch == epochs){
            cout << ANSI_COLOR_CYAN << "\rEpoch " << setw(4) << setfill('0') << ANSI_COLOR_BLUE << (epoch + 1)
                << ANSI_COLOR_RESET << "/" << ANSI_COLOR_BLUE << epochs << ANSI_COLOR_CYAN << " completed. [" << ANSI_COLOR_GREEN << setw(3) 
                << setfill(' ') << (int)((epoch + 1) * 100.0 / epochs) << "%" 
                << ANSI_COLOR_CYAN << "]" << ANSI_COLOR_RESET << flush;
        }else {
            cout << ANSI_COLOR_CYAN << "\rEpoch " << setw(4) << setfill('0') << ANSI_COLOR_RED << (epoch + 1)
                << ANSI_COLOR_RESET << "/" << ANSI_COLOR_BLUE << epochs << ANSI_COLOR_CYAN << " completed. [" << ANSI_COLOR_GREEN << setw(3) 
                << setfill(' ') << (int)((epoch + 1) * 100.0 / epochs) << "%" 
                << ANSI_COLOR_CYAN << "]" << ANSI_COLOR_RESET << flush;

        }

    }
    
}