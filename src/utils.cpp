#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "utils.hpp"
#include "neuralnetwork.hpp"

using namespace utils;
using namespace neuralnets;
using namespace std;

/* Standard Colors */
#define ANSI_COLOR_BLACK   "\x1b[30m"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_WHITE   "\x1b[37m"
#define ANSI_COLOR_TEAL    "\x1b[38;5;37m"   // Teal
#define ANSI_COLOR_LIME    "\x1b[38;5;10m"   // Lime Green

/* Bright Colors */
#define ANSI_COLOR_BRIGHT_BLACK   "\x1b[90m"
#define ANSI_COLOR_BRIGHT_RED     "\x1b[91m"
#define ANSI_COLOR_BRIGHT_GREEN   "\x1b[92m"
#define ANSI_COLOR_BRIGHT_YELLOW  "\x1b[93m"
#define ANSI_COLOR_BRIGHT_BLUE    "\x1b[94m"
#define ANSI_COLOR_BRIGHT_MAGENTA "\x1b[95m"
#define ANSI_COLOR_BRIGHT_CYAN    "\x1b[96m"
#define ANSI_COLOR_BRIGHT_WHITE   "\x1b[97m"


/* Bold Text */
#define ANSI_BOLD         "\x1b[1m"
#define ANSI_BOLD_RESET   "\x1b[22m"

/* Reset */
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
        if (epoch + 1 == epochs){
            cout << ANSI_COLOR_BRIGHT_CYAN << "\rEpoch " 
                << ANSI_COLOR_BRIGHT_WHITE << setw(4) << setfill('0') 
                << ANSI_BOLD << ANSI_COLOR_BRIGHT_GREEN << (epoch + 1) 
                << ANSI_BOLD << ANSI_COLOR_RESET << "/" 
                << ANSI_BOLD << ANSI_COLOR_BRIGHT_GREEN << epochs 
                << ANSI_BOLD << ANSI_COLOR_BRIGHT_CYAN << " completed. [" 
                << ANSI_COLOR_BRIGHT_GREEN << ANSI_BOLD << setw(3) << setfill(' ') 
                << (int)((epoch + 1) * 100.0 / epochs) << "%" 
                << ANSI_BOLD_RESET << ANSI_COLOR_BRIGHT_CYAN << "]" << ANSI_COLOR_RESET << flush;

            cout << endl;
        }else{
            cout << ANSI_COLOR_BRIGHT_CYAN << "\rEpoch " 
                << ANSI_COLOR_BRIGHT_WHITE << setw(4) << setfill('0') 
                << ANSI_COLOR_TEAL << (epoch + 1) 
                << ANSI_COLOR_RESET << "/" 
                << ANSI_BOLD << ANSI_COLOR_BRIGHT_BLUE << epochs 
                << ANSI_BOLD_RESET << ANSI_COLOR_BRIGHT_CYAN << " completed. [" 
                << ANSI_COLOR_TEAL << ANSI_BOLD << setw(3) << setfill(' ')
                << (int)((epoch + 1) * 100.0 / epochs) << "%" 
                << ANSI_BOLD_RESET << ANSI_COLOR_BRIGHT_CYAN << "]" << ANSI_COLOR_RESET << flush;
        }
    }

    void handle_error(const string& message, int exitCode){
        cerr << ANSI_BOLD << ANSI_COLOR_BRIGHT_RED << "ERROR: " << message << ANSI_BOLD_RESET << ANSI_COLOR_RESET << endl;
        exit(exitCode);
    }


    void handle_warning(const string& message){
        cout << ANSI_BOLD << ANSI_COLOR_BRIGHT_YELLOW << "WARNING: " << message << ANSI_BOLD_RESET << ANSI_COLOR_RESET << endl;
    }

    void handle_success(const string& message){
        cout << ANSI_BOLD << ANSI_COLOR_BRIGHT_GREEN << "SUCCESS: " << message << ANSI_BOLD_RESET << ANSI_COLOR_RESET << endl;
    }

    string tab_format(int sizeFormat){
        string tabs;
        
        tabs = string(sizeFormat, '\t');

        return tabs;
    }

    void clear_console(){
        cout << "\033[2J\033[1;1H";
    }
}

