#include "neuralnetwork.hpp"

using namespace neuralnets;
using namespace std;

namespace utils {
    void separator();
    void print_nn_io(NEURAL_NETWORK* nn);
    void print_nn_io_previous(NEURAL_NETWORK* nn);
    void print_train(int epoch, int epochs);
    void handle_error(const string& message, int exitCode);
    void handle_warning(const string& message);
    void handle_success(const string& message);
    string tab_format(int sizeFormat);
    void clear_console();
}