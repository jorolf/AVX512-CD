cmake_minimum_required(VERSION 3.16)
project(AVX512CD)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-ffast-math -mavx512cd")

add_executable(Histogram histogram.cpp)
add_executable(ConvolutionalHistogram convolutional_histogram.cpp)
add_executable(ConvolutionalBlurHistogram convolutional_blur_histogram.cpp)
add_executable(InstructionTest instruction_test.cpp)
add_executable(InstructionTiming instruction_timing.cpp)
