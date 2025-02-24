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

#define ANSI_COLOR_BRIGHT_YELLOW "\x1b[93m"
#define ANSI_COLOR_BRIGHT_RED "\x1b[91m"

namespace config
{
    NEURAL_NETWORK *initialize(unsigned int id, LIST_INFO *numNeuronsPerLayerList, int argc, char *argv[])
    {
        int num_layers = 0;
        bool found_l = false, found_n = false, found_lr = false, found_lambda = false, found_epochs = false;

        double learning_rate = 0.01; // Default learning rate
        double lambda = 0.001;       // Default lambda
        int epochs = 1000;           // Default epochs
        bool render = false;

        for (int i = 1; i < argc; i++){
            if (strcmp(argv[i], "-l") == 0){
                num_layers = atoi(argv[i + 1]);
                i++;
                found_l = true;
            } else if (strcmp(argv[i], "-n") == 0){
                if (num_layers == 0)
                    utils::handle_error("-n must be preceded by -l", 1);

                for (int j = 0; j < num_layers; j++){
                    if (i + j + 1 >= argc || !isdigit(*argv[i + j + 1])){
                        utils::handle_error("Invalid number of neurons or missing value after -n", 1);
                    }

                    ds_list::push(numNeuronsPerLayerList, atoi(argv[i + j + 1]));
                }
                i += num_layers;
                found_n = true;
            } else if (strcmp(argv[i], "--lr") == 0){
                learning_rate = atof(argv[i + 1]);
                i++;
                found_lr = true;
            } else if (strcmp(argv[i], "--lambda") == 0){
                lambda = atof(argv[i + 1]);
                i++;
                found_lambda = true;
            } else if (strcmp(argv[i], "--epochs") == 0) {
                epochs = atoi(argv[i + 1]);
                i++;
                found_epochs = true;
            } else if (strcmp(argv[i], "--render") == 0) {
                render = true;
                i++;
            } 
        }

        if (!found_l || !found_n)
            utils::handle_error("-l and -n are required arguments.", 1);
        if (!found_lr)
            utils::handle_warning("--lr not specified, using default value: " + to_string(learning_rate));        
        if (!found_lambda)
            utils::handle_warning("--lambda not specified, using default value: " + to_string(lambda));
        if (!found_epochs)
            utils::handle_warning("--epochs not specified, using default value: " + to_string(epochs));

        NEURAL_NETWORK *newNeuralNetwork = create_neural_network(id, numNeuronsPerLayerList, learning_rate, lambda, epochs);
        return newNeuralNetwork;
    }

    void initialize_neurons(NEURAL_NETWORK *nn, LIST_INFO *inputList, LIST_INFO *targetList, string line){
        string tempStringTrain = "";
        double trainData;

        stringstream inputString(line);
        for (int i = 0; i < (nn->layersInfo->list->value + nn->layersInfo->lastBlock->value); i++){
            getline(inputString, tempStringTrain, ',');
            trainData = atof(tempStringTrain.c_str());
            if (i < nn->layersInfo->list->value)
                push(inputList, trainData);
            else
                push(targetList, trainData);
            

            string tempStringTrain = "";
        }

        int inputIndex = 0;

        for (NEURON *currentNeuron = nn->inputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
            double normalizedInput = (get_value_by_index(inputList->list, inputIndex) / 100.00);
            currentNeuron->neuronValue = normalizedInput;
            currentNeuron->activation = currentNeuron->neuronValue;
            inputIndex++;
        }

        int targetIndex = 0;
        for (NEURON *currentNeuron = nn->outputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
            currentNeuron->target = get_value_by_index(targetList->list, targetIndex);
            targetIndex++;
        }

        // utils::print_nn_io(nn);

        empty(inputList);
        empty(targetList);
    }

    void remove_file_train(){
        string current_directory = fs::current_path().string();

        fs::path log_path;

        fs::path train_log_path = fs::path(current_directory) / "visualize" / "loss_function.log";
        fs::path classify_log_path = fs::path(current_directory) / "visualize" / "classify.log";

        if (fs::exists(train_log_path))
            fs::remove(train_log_path);

        if (fs::exists(classify_log_path))
            fs::remove(classify_log_path);
    }

    void remove_file_predict(){
        string current_directory = fs::current_path().string();

        fs::path log_path;

        fs::path prediction_log_path = fs::path(current_directory) / "prediction" / "predict.log";

        if (fs::exists(prediction_log_path))
            fs::remove(prediction_log_path);
    }

    void save_loss_function(double lossFunction, char type)
    {
        string current_directory = fs::current_path().string();

        fs::path log_path;

        if (type == 'T'){
            log_path = fs::path(current_directory) / "visualize" / "loss_function.log";
        }
        else{
            log_path = fs::path(current_directory) / "visualize" / "classify.log";
        }

        ofstream file(log_path, ios::app);

        if (!file.is_open())
            handle_error("Could not open the file.", 1);

        file << lossFunction << "\n";

        file.close();
    }

    void save_predictions(double targetIndex, double predictionIndex)
    {
        string current_directory = fs::current_path().string();

        fs::path log_path;

        log_path = fs::path(current_directory) / "prediction" / "predict.log";

        ofstream file(log_path, ios::app);

        if (!file.is_open())
            handle_error("Could not open the file.", 1);


        file << targetIndex << "," << predictionIndex << "\n";

        file.close();
    }

    void train_with_epochs_randomly(NEURAL_NETWORK *nn, string filePath, bool saving_mode)
    {
        FILE_LIST_INFO *lines = new FILE_LIST_INFO();

        ifstream trainFile(filePath);

        if (!saving_mode)
            remove_file_train();

        if (!trainFile.is_open())
            handle_error("Could not open the file.", 1);


        string line = "";
        while (getline(trainFile, line))
        {
            file_list::push(lines, line);
            line = "";
        }

        trainFile.close();

        int index = 0;
        for (int epoch = 0; epoch < nn->epochs; ++epoch)
        {
            LIST_INFO *inputList = new LIST_INFO();
            LIST_INFO *targetList = new LIST_INFO();
            nn->learningRate *= 0.99; // Learning rate decay

            for (int i = 0; i < lines->size; i++)
            {
                int randomIndex = rand() % lines->size; // Randomly select a sample
                string currentLine = file_list::get_line_by_index(lines->file_list, randomIndex);

                initialize_neurons(nn, inputList, targetList, currentLine);
                neuralnets::feed_forward(nn);
                neuralnets::backpropagation(nn);
                save_loss_function(nn->lossFunction, 'T');

                print_train(index, nn->epochs * lines->size);
                index++;
            }
        }
    }

    void train_with_epochs(NEURAL_NETWORK *nn, string filePath, bool saving_mode)
    {
        FILE_LIST_INFO *lines = new FILE_LIST_INFO();

        ifstream trainFile(filePath);

        if (!saving_mode)
            remove_file_train();

        if (!trainFile.is_open())
            handle_error("Could not open the file.", 1);
            
        string line = "";

        while (getline(trainFile, line)){
            file_list::push(lines, line);
            line = "";
        }

        trainFile.close();

        int index = 0;
        for (int epoch = 0; epoch < nn->epochs; ++epoch){
            LIST_INFO *inputList = new LIST_INFO();
            LIST_INFO *targetList = new LIST_INFO();
            nn->learningRate *= 0.99;
            for (int i = 0; i < lines->size; i++){
                string currentLine = file_list::get_line_by_index(lines->file_list, i);
                initialize_neurons(nn, inputList, targetList, currentLine);
                neuralnets::feed_forward(nn);
                neuralnets::backpropagation(nn);
                save_loss_function(nn->lossFunction, 'T');
                // cout << currentLine << endl;
                print_train(index, nn->epochs * lines->size);
                index++;
            }
        }
    }

    void classify(NEURAL_NETWORK *nn, string filePath)
    {
        FILE_LIST_INFO *lines = new FILE_LIST_INFO();

        ifstream classifyFile(filePath);
        
        if (!classifyFile.is_open())
            handle_error("Could not open the file.", 1);

        string line = "";

        while (getline(classifyFile, line)){
            file_list::push(lines, line);
            line = "";
        }

        classifyFile.close();

        LIST_INFO *inputList = new LIST_INFO();
        LIST_INFO *targetList = new LIST_INFO();

        for (int i = 0; i < lines->size; i++){
            string currentLine = file_list::get_line_by_index(lines->file_list, i);
            initialize_neurons(nn, inputList, targetList, currentLine);
            neuralnets::feed_forward(nn);
            neuralnets::loss_function(nn);
            save_loss_function(nn->lossFunction, 'C');
            // cout << currentLine << endl;
        }
    }

    void predict(NEURAL_NETWORK *nn, string filePath){
        FILE_LIST_INFO *lines = new FILE_LIST_INFO();

        ifstream classifyFile(filePath);

        remove_file_predict();
        if (!classifyFile.is_open())
            handle_error("Could not open the file.", 1);

        string line = "";

        while (getline(classifyFile, line)){
            file_list::push(lines, line);
            line = "";
        }

        classifyFile.close();

        LIST_INFO *inputList = new LIST_INFO();
        LIST_INFO *targetList = new LIST_INFO();

        for (int i = 0; i < lines->size; i++){
            string currentLine = file_list::get_line_by_index(lines->file_list, i);
            initialize_neurons(nn, inputList, targetList, currentLine);
            neuralnets::feed_forward(nn);
            neuralnets::predict(nn);
            // cout << currentLine << endl;
        }
    }

    void save_neural_network(NEURAL_NETWORK *nn, const string &filePath){
        fs::path directoryPath = fs::path(filePath).parent_path();

        if (!fs::exists(directoryPath)) {
            try {
                fs::create_directories(directoryPath);
            } catch (const std::exception &e) {
                utils::handle_error("Failed to create directory: " + directoryPath.string() + " (" + e.what() + ")", 1);
            }
        }
        
        ofstream file(filePath);
        if (!file.is_open())
            utils::handle_error("Unable to open file for saving neural network.", 1);

        int NEURAL_NETWORK_FORMAT = 1;
        int LAYER_FORMAT = 2;
        int NEURON_FORMAT = 3;
        int CONNECTION_FORMAT = 4;

        // SAVE NEURAL NETWORK INITIALIZATION BASIC INFO
        file << "NEURAL_NETWORK " << nn->id << endl;
        file << tab_format(NEURAL_NETWORK_FORMAT) << "LAYERS_QUANTITY: " << nn->layersInfo->size << endl;

        for(LAYER* currentLayer = nn->inputLayer; currentLayer != NULL; currentLayer = currentLayer->next)
            file << tab_format(NEURAL_NETWORK_FORMAT) << "LAYER " << currentLayer->id << " NEURONS_QUANTITY " << currentLayer->numNeurons << endl;

        file << tab_format(NEURAL_NETWORK_FORMAT) << "LEARNING_RATE " << nn->learningRate << endl;
        file << tab_format(NEURAL_NETWORK_FORMAT) << "LAMBDA " << nn->lambda << endl;
        file << tab_format(NEURAL_NETWORK_FORMAT) << "EPOCHS " << nn->epochs << endl;
        file << endl;
        for(LAYER* currentLayer = nn->inputLayer; currentLayer != NULL; currentLayer = currentLayer->next){
            file << tab_format(NEURAL_NETWORK_FORMAT) << "LAYER " << currentLayer->id << endl;
            
            for(NEURON* currentNeuron = currentLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next){
                file << tab_format(LAYER_FORMAT) << "NEURON " << currentNeuron->id << endl;
                file << tab_format(NEURON_FORMAT) << "NEURON_BIAS: " << currentNeuron->bias << endl;
                file << tab_format(NEURON_FORMAT) << "NEURON_DELTA_LOSS: " << currentNeuron->deltaLoss << endl;

                for(CONNECTION* currentConnection = currentNeuron->connections; currentConnection != NULL; currentConnection = currentConnection->next){
                    file << tab_format(CONNECTION_FORMAT) << "CONNECTION " << currentConnection->backwardNeuron->id << " " << currentConnection->weight << " " << currentConnection->afterwardNeuron->id << endl;
                }

                /*
                for(CONNECTION* currentConnection = currentNeuron->previousConnections; currentConnection != NULL; currentConnection = currentConnection->nextAsPrevious){
                    file << tab_format(NEURON_FORMAT) << "PREVIOUS_CONNECTION " << currentConnection->id << endl;
                    file << tab_format(CONNECTION_FORMAT) << "PREVIOUS_CONNECTION_WEIGHT " << currentConnection->weight << endl;            
                }
                */

            }
        }

        file.close();
    }

    NEURAL_NETWORK* parse_neural_network(const string &filePath) {
        ifstream file(filePath);

        if (!file.is_open())
            utils::handle_error("Unable to open file for loading neural network.", 1);

        // Define an enum to map string keys to integer values
        enum Key {
            NEURAL_NETWORK_KEY,
            LAYERS_QUANTITY_KEY,
            LAYER_KEY,
            LEARNING_RATE_KEY,
            LAMBDA_KEY,
            EPOCHS_KEY,
            NEURON_KEY,
            NEURON_BIAS_KEY,
            NEURON_DELTA_LOSS_KEY,
            CONNECTION_KEY,
            UNKNOWN_KEY
        };

        // Function to convert string keys to enum values
        auto getKey = [](const string& key) -> Key {
            if (key == "NEURAL_NETWORK") return NEURAL_NETWORK_KEY;
            if (key == "LAYERS_QUANTITY:") return LAYERS_QUANTITY_KEY;
            if (key == "LAYER") return LAYER_KEY;
            if (key == "LEARNING_RATE") return LEARNING_RATE_KEY;
            if (key == "LAMBDA") return LAMBDA_KEY;
            if (key == "EPOCHS") return EPOCHS_KEY;
            if (key == "NEURON") return NEURON_KEY;
            if (key == "NEURON_BIAS:") return NEURON_BIAS_KEY;
            if (key == "NEURON_DELTA_LOSS:") return NEURON_DELTA_LOSS_KEY;
            if (key == "CONNECTION") return CONNECTION_KEY;
            return UNKNOWN_KEY;
        };

        // Variables to store parsed data
        unsigned int id = 0;
        double learningRate = 0.0;
        double lambda = 0.0;
        int epochs = 0;
        bool finishedInitializationFlag = false;

        LIST_INFO* layerSizesList = new LIST_INFO();
        layerSizesList->size = 0;

        string line;
        while (getline(file, line)) {
            stringstream ss(line);
            string key;
            ss >> key;

            // Get the enum value for the key
            Key currentKey = getKey(key);

            switch (currentKey) {
                case NEURAL_NETWORK_KEY: {
                    ss >> id;
                    break;
                }
                case LAYERS_QUANTITY_KEY: {
                    int layersQuantity;
                    ss >> layersQuantity;
                    layerSizesList->size = layersQuantity;
                    break;
                }
                case LAYER_KEY: {
                    unsigned int layerId;

                    string temp;
                    int neuronsQuantity;
                    
                    ss >> layerId >> temp >> neuronsQuantity;
                    
                    if (temp != "NEURONS_QUANTITY") {
                        utils::handle_error("Formato inválido na linha da camada", 1);
                    }

                    ds_list::push(layerSizesList, neuronsQuantity);
                    break;
                }
                case LEARNING_RATE_KEY: {
                    ss >> learningRate;
                    break;
                }
                case LAMBDA_KEY: {
                    ss >> lambda;
                    break;
                }
                case EPOCHS_KEY: {
                    ss >> epochs;
                    finishedInitializationFlag = true;
                    break;
                }
                default: {
                    break;
                }
            }

            if (finishedInitializationFlag)
                break;
        }

        file.close();

        ds_list::show(layerSizesList);
        // Use load_neural_network to initialize the neural network structure
        NEURAL_NETWORK* nn = load_neural_network(id, layerSizesList, learningRate, lambda, epochs);

        print_nn_io(nn);

        cout << endl;

        utils::handle_success("Initialization loaded");

        // Now, reopen the file to parse and update the weights, biases, and other parameters
        file.open(filePath);
        if (!file.is_open())
            utils::handle_error("Unable to reopen file for loading neural network parameters.", 1);

        LAYER* currentLayer = nn->inputLayer; // Reset currentLayer to inputLayer
        NEURON* currentNeuron = NULL;
        CONNECTION* currentConnection = NULL;

        while (getline(file, line)) {
            stringstream ss(line);
            string key;
            ss >> key;

            Key currentKey = getKey(key);

            switch (currentKey) {
                case LAYER_KEY: {
                    unsigned int layerId;
                    int neuronsQuantity;
                    ss >> layerId >> neuronsQuantity;

                    // Reset currentLayer to inputLayer before searching
                    currentLayer = nn->inputLayer;

                    // Move to the correct layer
                    while (currentLayer != NULL && currentLayer->id != layerId) {
                        currentLayer = currentLayer->next;
                    }
                    if (currentLayer == NULL) {
                        utils::handle_error("Layer " + to_string(layerId) + " not found.", 1);
                    }
                    utils::handle_success("LAYER " + to_string(layerId) + " loaded");
                    break;
                }
                case NEURON_KEY: {
                    unsigned int neuronId;
                    ss >> neuronId;

                    // Find the neuron in the current layer
                    if (currentLayer != NULL) {
                        currentNeuron = currentLayer->neurons;
                        while (currentNeuron != NULL && currentNeuron->id != neuronId) {
                            currentNeuron = currentNeuron->next;
                        }
                        if (currentNeuron == NULL) {
                            utils::handle_error("Neuron " + to_string(neuronId) + " not found in layer " + to_string(currentLayer->id), 1);
                        }
                        utils::handle_success("NEURON " + to_string(neuronId) + " loaded");
                    } else {
                        utils::handle_error("No current layer to find neuron.", 1);
                    }
                    break;
                }
                case NEURON_BIAS_KEY: {
                    double bias;
                    ss >> bias;
                    if (currentNeuron != NULL) {
                        currentNeuron->bias = bias;
                        utils::handle_success("NEURON BIAS " + to_string(bias) + " loaded");
                    } else {
                        utils::handle_error("No current neuron to set bias.", 1);
                    }
                    break;
                }
                case NEURON_DELTA_LOSS_KEY: {
                    double deltaLoss;
                    ss >> deltaLoss;
                    if (currentNeuron != NULL) {
                        currentNeuron->deltaLoss = deltaLoss;
                        utils::handle_success("NEURON DELTA LOSS " + to_string(deltaLoss) + " loaded");
                    } else {
                        utils::handle_error("No current neuron to set delta loss.", 1);
                    }
                    break;
                }
                case CONNECTION_KEY: {
                    unsigned int srcNeuronId, destNeuronId;
                    double weight;
                    
                    ss >> srcNeuronId >> weight >> destNeuronId;

                    if (currentLayer == nullptr) {
                        utils::handle_error("Cannot find source neuron: No active layer context.", 1);
                    }

                    // Find source and destination neurons
                    NEURON* srcNeuron = find_neuron(currentLayer, srcNeuronId);
                    NEURON* destNeuron = find_neuron(currentLayer->next, destNeuronId);

                    if (!srcNeuron || !destNeuron) {
                        utils::handle_error("Error: Could not find neurons for connection " + to_string(srcNeuronId) + " -> " + to_string(destNeuronId), 1);
                    }

                    // Create the connection with the parsed weight
                    neuralnets::create_connection(srcNeuron, weight, destNeuron);

                    utils::handle_success("CONNECTION " + to_string(srcNeuronId) + " -> " + to_string(destNeuronId) + " with weight " + to_string(weight) + " loaded.");
                    break;
                }
                default: {
                    // Handle unknown keys (if needed)
                    break;
                }
            }
        }

        file.close();

        return nn;
    }
}