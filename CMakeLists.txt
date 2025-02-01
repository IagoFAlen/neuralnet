cmake_minimum_required(VERSION 3.10)

# Project name
project(neuralnet)

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp src/config.cpp src/neuralnetwork.cpp src/math.cpp)

# Add include directories
include_directories(/usr/local/include)

# Find the Quicksort and List libraries
find_library(LIST_LIBRARY list PATHS /usr/local/lib REQUIRED)

if (NOT LIST_LIBRARY)
    message(FATAL_ERROR "Could not find List library")
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${QUICKSORT_LIBRARY} ${LIST_LIBRARY})
