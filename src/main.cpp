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

    LIST_INFO* layersList = new LIST_INFO();

    int num_layers = 0;

    // Taking the parameters and setting it into layer list informations
    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "-l") == 0) {
            num_layers = atoi(argv[i+1]);
            i++;
        } else if(strcmp(argv[i], "-n") == 0) {
            for(int j = 0; j < num_layers; j++) {
                ds_list::push(layersList, atoi(argv[i+j+1]));
            }
            i += num_layers;
        }
    }

    NEURAL_NETWORK* nn = new NEURAL_NETWORK();

    nn = config::initialize(layersList);

    return 0;
}