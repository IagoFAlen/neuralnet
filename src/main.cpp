#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdio>

#include "neuralnetwork.hpp"
#include "config.hpp"
#include "list.hpp"

using namespace std;
using namespace neuralnets;
using namespace ds_list;
using namespace config;


int main(int argc, char *argv[]){
    srand(time(NULL));
    string TRAIN_FILE_PATH = "/home/john/playground/neuralnets/training/training.csv";
    double LEARNING_RATE = 0.01;
    LIST_INFO* numNeuronsPerLayerList = new LIST_INFO();

    NEURAL_NETWORK* nn = new NEURAL_NETWORK();

    nn = config::initialize(1, numNeuronsPerLayerList, argc, argv, LEARNING_RATE);

    config::train(nn, TRAIN_FILE_PATH);
    return 0;
}