# AVX-512CD Example Code

This repository contains the code of a presentation I made for a university course.  
The presentation can be found here: [Link](https://docs.google.com/presentation/d/1AQOQchN-vUPKt3hO77HBq3Ye6_RVyF_5hwFgtzzfkgI/edit?usp=sharing)

- [instruction_test.cpp](./instruction_test.cpp) contains code demonstrating the basic functionality of the instruction set
- [histogram.cpp](./histogram.cpp) has different implementations of a basic histogram
- [convolutional_histogram.cpp](./convolutional_histogram.cpp) implements a histogram with a simple convolutional filter
- [convolutional_blur_histogram.cpp](./convolutional_blur_histogram.cpp) implements a histogram with a convolutional filter that averages the values
- [instruction_timing.cpp](./instruction_timing.cpp) measures the time that certain instructions take to complete

The code is licensed under the [MIT License](./LICENSE).

If you feel like improving the code feel free to create a pull request.