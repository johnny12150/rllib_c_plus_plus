# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(CartPoleONNX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the path to the ONNX Runtime include directory and libraries
set(ONNXRUNTIME_DIR "onnxruntime-osx-arm64-1.19.2")  # Change this to your ONNX Runtime installation path

# Add the ONNX Runtime include directories
include_directories(${ONNXRUNTIME_DIR}/include)

# Link the ONNX Runtime library
link_directories(${ONNXRUNTIME_DIR}/lib)

# Specify the source file(s) for the project
add_executable(cartpole_onnx main.cpp)  # Replace 'main.cpp' with the actual C++ source file

# Link against ONNX Runtime
target_link_libraries(cartpole_onnx onnxruntime)

# Set any additional compile options if needed
target_compile_options(cartpole_onnx PRIVATE -Wall -Wextra)
