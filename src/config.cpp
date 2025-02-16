#include <fstream>
#include <cstring>
#include <sstream>
#include <filesystem>
#include <cstring>

#include "config.hpp"
#include "neuralnetwork.hpp"
#include "utils.hpp"

using namespace std;
using namespace config;
using namespace neuralnets;
using namespace utils;

namespace fs = filesystem;

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

    void remove_file(){
        string current_directory = fs::current_path().string();

        fs::path log_path;
        
        
        fs::path train_log_path = fs::path(current_directory) / "visualize" / "loss_function.log";
        fs::path classify_log_path = fs::path(current_directory) / "visualize" / "classify.log";
        

        if(fs::exists(train_log_path))
            fs::remove(train_log_path);
        

        if(fs::exists(classify_log_path))
            fs::remove(classify_log_path);    
    }

    void save_loss_function(double lossFunction, char type) {
        string current_directory = fs::current_path().string();

        fs::path log_path;
        
        if (type == 'T') {
            log_path = fs::path(current_directory) / "visualize" / "loss_function.log";
        } else {
            log_path = fs::path(current_directory) / "visualize" / "classify.log";
        }

        ofstream file(log_path, ios::app);
        
        if (!file.is_open()) {
            perror("Error opening log file");
            return;
        }

        file << lossFunction << "\n";

        file.close();
    }

    void train_with_epochs_randomly(NEURAL_NETWORK* nn, string filePath, int epochs, bool saving_mode){
        FILE_LIST_INFO* lines = new FILE_LIST_INFO();

        ifstream trainFile(filePath);

        if(!saving_mode)
            remove_file();

        if (!trainFile.is_open()) {
            cout << "Error opening file: " << filePath << endl;
            return;
        }

        string line = "";

        while (getline(trainFile, line)) {
            file_list::push(lines, line);
            line = "";
        }

        trainFile.close();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            int lineIndex = rand() % lines->size;

            LIST_INFO* inputList = new LIST_INFO();
            LIST_INFO* targetList = new LIST_INFO();

            string currentLine = file_list::get_line_by_index(lines->file_list, lineIndex);
            initialize_neurons(nn, inputList, targetList, currentLine);
            neuralnets::feed_forward(nn);
            neuralnets::backpropagation(nn);
            save_loss_function(nn->lossFunction, 'T');
            //cout << currentLine << endl;
            print_train(epoch, epochs);
        }
    }

    void train_with_epochs(NEURAL_NETWORK* nn, string filePath, int epochs, bool saving_mode){
        FILE_LIST_INFO* lines = new FILE_LIST_INFO();

        ifstream trainFile(filePath);

        if(!saving_mode)
            remove_file();

        if (!trainFile.is_open()) {
            cout << "Error opening file: " << filePath << endl;
            return;
        }

        string line = "";

        while (getline(trainFile, line)) {
            file_list::push(lines, line);
            line = "";
        }

        trainFile.close();

        int index = 0;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            LIST_INFO* inputList = new LIST_INFO();
            LIST_INFO* targetList = new LIST_INFO();
            for(int i = 0; i < lines->size; i++){
                string currentLine = file_list::get_line_by_index(lines->file_list, i);
                initialize_neurons(nn, inputList, targetList, currentLine);
                neuralnets::feed_forward(nn);
                neuralnets::backpropagation(nn);
                save_loss_function(nn->lossFunction, 'T');
                //cout << currentLine << endl;
                print_train(index, epochs * lines->size);
                index++;
            }
        }
    }

    void classify(NEURAL_NETWORK* nn, string filePath){
        FILE_LIST_INFO* lines = new FILE_LIST_INFO();

        ifstream classifyFile(filePath);
        
        if (!classifyFile.is_open()) {
            cout << "Error opening file: " << filePath << endl;
            return;
        }

        string line = "";

        while (getline(classifyFile, line)) {
            file_list::push(lines, line);
            line = "";
        }

        classifyFile.close();

        LIST_INFO* inputList = new LIST_INFO();
        LIST_INFO* targetList = new LIST_INFO();

        for(int i = 0; i < lines->size; i++){
            string currentLine = file_list::get_line_by_index(lines->file_list, i);
            initialize_neurons(nn, inputList, targetList, currentLine);
            neuralnets::feed_forward(nn);
            neuralnets::loss_function(nn);
            save_loss_function(nn->lossFunction, 'C');
            //cout << currentLine << endl;
        }
    }
    
}