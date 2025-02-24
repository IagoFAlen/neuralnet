cmake_minimum_required(VERSION 3.10)

# Project name
project(neuralnet)

# Find OpenGL, GLFW, and GLEW
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

# Add the executable
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/config.cpp
    src/neuralnetwork.cpp
    src/math.cpp
    src/utils.cpp
    src/render.cpp
)

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR} # Add this to include globals.hpp
    /usr/local/include
    /usr/include
)

# Find the List and File List libraries
find_library(LIST_LIBRARY list PATHS /usr/local/lib REQUIRED)
find_library(FILE_LIST_LIBRARY file_list PATHS /usr/local/lib REQUIRED)

if (NOT LIST_LIBRARY)
    message(FATAL_ERROR "Could not find List library")
endif()

if (NOT FILE_LIST_LIBRARY)
    message(FATAL_ERROR "Could not find File List library")
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${LIST_LIBRARY}
    ${FILE_LIST_LIBRARY}
    glfw
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    pthread
)