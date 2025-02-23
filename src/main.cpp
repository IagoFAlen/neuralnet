#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdio>
#include <filesystem>

#include "neuralnetwork.hpp"
#include "config.hpp"
#include "list.hpp"
#include "utils.hpp"

using namespace std;
using namespace neuralnets;
using namespace ds_list;
using namespace config;
using namespace utils;

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    srand(time(NULL));
    LIST_INFO* numNeuronsPerLayerList = new LIST_INFO();

    string current_directory = fs::current_path().string();
    string TRAIN_FILE_PATH = current_directory + "/training/training.csv";
    string CLASSIFY_FILE_PATH = current_directory + "/classification/testing.csv";
    string PREDICT_FILE_PATH = current_directory + "/classification/testing.csv";
    string NETWORK_FILE_PATH = current_directory + "/state/network_state.txt"; // Default path for saving/loading the network

    NEURAL_NETWORK* nn = nullptr;

    bool load_from_file = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--load") == 0) {
            load_from_file = true;
            if (i + 1 < argc) {
                if (argv[i + 1][0] != ' ' && argv[i + 1][0] != '\t' && argv[i + 1][0] != '\0') {
                    NETWORK_FILE_PATH = argv[i + 1];
                } else {
                    handle_error("Invalid file path provided after --load.", 1);
                }
            }
            break;
        }
    }

    if (load_from_file) {
        nn = config::parse_neural_network(NETWORK_FILE_PATH);
        if (!nn)
            utils::handle_error("Failed to load neural network from file.", 1);
        
        utils::handle_success("Neural network loaded from file:" + NETWORK_FILE_PATH);

    } else {
        nn = config::initialize(1, numNeuronsPerLayerList, argc, argv);
        utils::clear_console();
        if (!nn)
            utils::handle_error("Failed to initialize neural network.", 1);
        
        utils::handle_success("Neural network initialized randomly.");

        bool saving_mode = false;
        config::train_with_epochs_randomly(nn, TRAIN_FILE_PATH, saving_mode);
        config::classify(nn, CLASSIFY_FILE_PATH);

        config::save_neural_network(nn, NETWORK_FILE_PATH);
        //utils::print_nn_io(nn);
        //utils::print_nn_io_previous(nn);
        utils::handle_success("Neural network saved to file: " + NETWORK_FILE_PATH);
    }

    config::predict(nn, PREDICT_FILE_PATH);

    return 0;
}