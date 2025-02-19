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

#define ANSI_COLOR_BRIGHT_YELLOW  "\x1b[93m"
#define ANSI_COLOR_BRIGHT_RED     "\x1b[91m"

namespace config {
    NEURAL_NETWORK* initialize(unsigned int id, LIST_INFO* numNeuronsPerLayerList, int argc, char *argv[]) {
        int num_layers = 0;
        bool found_l = false, found_n = false, found_lr = false, found_lambda = false, found_epochs = false, found_batches = false;

        double learning_rate = 0.001;   // Default learning rate
        double lambda = 0.01;           // Default lambda
        int epochs = 1000;              // Default epochs
        int batches = 32;               // Default batches size

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-l") == 0) {
                num_layers = atoi(argv[i + 1]);
                i++;
                found_l = true;
            } else if (strcmp(argv[i], "-n") == 0) {
                if (num_layers == 0) {
                    std::cerr << "Error: -n must be preceded by -l" << std::endl;
                    exit(1);
                }
                for (int j = 0; j < num_layers; j++) {
                    if (i + j + 1 >= argc || !isdigit(*argv[i+j+1])) {
                        std::cerr << "Error: Invalid number of neurons or missing value after -n" << std::endl;
                        exit(1);
                    }
                    ds_list::push(numNeuronsPerLayerList, atoi(argv[i + j + 1]));
                }
                i += num_layers;
                found_n = true;
            } else if (strcmp(argv[i], "--lr") == 0) {
                learning_rate = atof(argv[i + 1]);
                i++;
                found_lr = true;
            } else if (strcmp(argv[i], "--lambda") == 0) {
                lambda = atof(argv[i + 1]);
                i++;
                found_lambda = true;
            } else if (strcmp(argv[i], "--epochs") == 0) {
                epochs = atoi(argv[i + 1]);
                i++;
                found_epochs = true;
            } else if (strcmp(argv[i], "--batches") == 0) {
                batches = atoi(argv[i + 1]);
                i++;
                found_batches = true;
            }
        }

        if (!found_l || !found_n) {
            cerr << ANSI_COLOR_BRIGHT_RED << "Error: -l and -n are required arguments." << endl;
            exit(1);
        }
        if (!found_lr) {
            cerr << ANSI_COLOR_BRIGHT_YELLOW << "Warning: --lr not specified, using default value: " << learning_rate << endl;
        }
        if (!found_lambda) {
            cerr << ANSI_COLOR_BRIGHT_YELLOW << "Warning: --lambda not specified, using default value: " << lambda << endl;
        }
        if (!found_epochs) {
            cerr << ANSI_COLOR_BRIGHT_YELLOW << "Warning: --epochs not specified, using default value: " << epochs << endl;
        }
        if (!found_batches) {
            cerr << ANSI_COLOR_BRIGHT_YELLOW << "Warning: --batches not specified, using default value: " << batches << endl;
        }

        NEURAL_NETWORK* newNeuralNetwork = create_neural_network(id, numNeuronsPerLayerList, learning_rate, lambda, epochs, batches);
        return newNeuralNetwork;
    }

    void compute_mean_std(const string& filePath, double& mean, double& stddev){
        ifstream trainFile(filePath);
        if (!trainFile.is_open()) {
            cerr << "Error opening file: " << filePath << endl;
            exit(1);
        }

        string line;
        double sum = 0.0, sum_sq = 0.0;
        int count = 0;

        while (getline(trainFile, line)) {
            stringstream inputString(line);
            string value;
            while (getline(inputString, value, ',')) {
                double trainData = atof(value.c_str());
                sum += trainData;
                sum_sq += trainData * trainData;
                count++;
            }
        }

        trainFile.close();

        if (count > 0) {
            mean = sum / count;
            stddev = sqrt((sum_sq / count) - (mean * mean));
            if (stddev == 0.0) stddev = 1.0; // Avoid division by zero
        } else {
            cerr << "Error: No data found in training file" << endl;
            exit(1);
        }

        cout << "Computed Mean: " << mean << ", Standard Deviation: " << stddev << endl;
    }

    void initialize_neurons(NEURAL_NETWORK* nn, LIST_INFO* inputList, LIST_INFO* targetList, string line, double mean, double stddev) {
        string tempStringTrain = "";
        double trainData;

        stringstream inputString(line);
        for (int i = 0; i < (nn->layersInfo->list->value + nn->layersInfo->lastBlock->value); i++) {
            getline(inputString, tempStringTrain, ',');
            trainData = atof(tempStringTrain.c_str());

            if (i < nn->layersInfo->list->value) {
                // Normalize using precomputed mean and stddev
                double normalizedValue = (trainData - mean) / stddev;
                push(inputList, normalizedValue);
            } else {
                push(targetList, trainData); // Target values remain unchanged
            }
        }

        int inputIndex = 0;
        for (NEURON* currentNeuron = nn->inputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next) {
            currentNeuron->neuronValue = get_value_by_index(inputList->list, inputIndex);
            currentNeuron->activation = currentNeuron->neuronValue;
            inputIndex++;
        }

        int targetIndex = 0;
        for (NEURON* currentNeuron = nn->outputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next) {
            currentNeuron->target = get_value_by_index(targetList->list, targetIndex);
            targetIndex++;
        }

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

    /*
    void train_with_epochs_randomly(NEURAL_NETWORK* nn, string filePath, int epochs, bool saving_mode, double lambda){
        FILE_LIST_INFO* lines = new FILE_LIST_INFO();

        ifstream trainFile(filePath);
        if (!trainFile.is_open()) {
            cout << "Error opening file: " << filePath << endl;
            return;
        }

        if (!saving_mode)
            remove_file();

        string line;
        while (getline(trainFile, line)) {
            file_list::push(lines, line);
        }
        trainFile.close();

        // Compute Mean and Std once for normalization
        double mean = 0.0, stddev = 1.0;
        compute_mean_std(filePath, mean, stddev);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            int lineIndex = rand() % lines->size;

            LIST_INFO* inputList = new LIST_INFO();
            LIST_INFO* targetList = new LIST_INFO();

            string currentLine = file_list::get_line_by_index(lines->file_list, lineIndex);
            initialize_neurons(nn, inputList, targetList, currentLine, mean, stddev);
            neuralnets::feed_forward(nn);
            neuralnets::backpropagation(nn);
            save_loss_function(nn->lossFunction, 'T');
            print_train(epoch, epochs);
        }
    }
    */

    void train_with_epochs(NEURAL_NETWORK* nn, string filePath, bool saving_mode) {
        FILE_LIST_INFO* lines = new FILE_LIST_INFO();

        ifstream trainFile(filePath);
        if (!trainFile.is_open()) {
            cerr << "Error opening file: " << filePath << endl;
            return;
        }

        if (!saving_mode)
            remove_file();

        string line;
        while (getline(trainFile, line)) {
            file_list::push(lines, line);
        }
        trainFile.close();

        // Compute Mean and Std over entire dataset
        double mean = 0.0, stddev = 1.0;
        compute_mean_std(filePath, mean, stddev);

        int num_batches = (lines->size + nn->batchesSize - 1) / nn->batchesSize;

        for (int epoch = 0; epoch < nn->epochs; ++epoch) {
            for (int batch = 0; batch < num_batches; ++batch) {
                LIST_INFO* inputList = new LIST_INFO();
                LIST_INFO* targetList = new LIST_INFO();

                int start_idx = batch * nn->batchesSize;
                int end_idx = min(start_idx + nn->batchesSize, lines->size);

                for (int i = start_idx; i < end_idx; ++i) {
                    string currentLine = file_list::get_line_by_index(lines->file_list, i);
                    initialize_neurons(nn, inputList, targetList, currentLine, mean, stddev);
                    neuralnets::feed_forward(nn);
                    neuralnets::backpropagation(nn); // Accumulate gradients
                    save_loss_function(nn->lossFunction, 'T');
                }

                // Update weights and biases using the accumulated gradients
                neuralnets::update_weights_and_biases(nn);

                // Save loss function for the batch
                print_train(epoch * num_batches + batch, nn->epochs * num_batches);
            }
        }
    }

    void classify(NEURAL_NETWORK* nn, string filePath, double mean, double stddev) {
        FILE_LIST_INFO* lines = new FILE_LIST_INFO();

        ifstream classifyFile(filePath);
        
        if (!classifyFile.is_open()) {
            cout << "Error opening file: " << filePath << endl;
            return;
        }

        string line = "";
        while (getline(classifyFile, line)) {
            file_list::push(lines, line);
        }
        classifyFile.close();

        LIST_INFO* inputList = new LIST_INFO();
        LIST_INFO* targetList = new LIST_INFO();

        for (int i = 0; i < lines->size; i++) {
            string currentLine = file_list::get_line_by_index(lines->file_list, i);
            initialize_neurons(nn, inputList, targetList, currentLine, mean, stddev);
            neuralnets::feed_forward(nn);
            neuralnets::loss_function(nn);
            save_loss_function(nn->lossFunction, 'C');
        }
    }

    
}