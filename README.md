# Parallel Image Search using CUDA

![Photo by <a href="https://unsplash.com/@rapol?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Rafael Pol</a> on <a href="https://unsplash.com/photos/6b5uqlWabB0?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  ](https://github.com/faridmmz/Image-Search-CUDA/blob/main/README_image.jpg "Photo by Rafael Pol on Unsplash")


## Contributors

- Faridreza Momtazzandi
- Mahya Ehsanimehr
- Ali Mojahed

## Table of Contents

- [Introduction](#introduction)
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [File Structure](#file-Structure)
- [Compilation and Execution](#compilation-and-Execution)
- [Results and Performance](#results-and-Performance)
- [Hardware Specifications](#hardware-Specifications)
- [Conclusion](#conclusion)

## Introduction

This project focuses on implementing a parallel image search program using CUDA, a parallel computing platform and API model created by NVIDIA. The objective is to efficiently find similar images within a dataset of random images using an image query. based on image features by utilizing the parallel processing capabilities of NVIDIA GPUs.

## Overview

The project involves rewriting an existing image search program from a previous phase (implemented in C++ with OpenMP) into a CUDA-based version. The program aims to find similar images by comparing various image features such as mean, median, standard deviation, Hu moments, and histograms. The CUDA implementation leverages the GPU's parallel architecture to significantly accelerate the image similarity calculation.

## Prerequisites

- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit
- OpenCV Library (for image processing)
- C++ Compiler (compatible with CUDA)

## File Structure

The repository includes the following files:

- `main.cu`: The main CUDA program that performs the parallel image search.
- `multicoreproject-CUDA-doc.pdf`: Detailed documentation about the project, including code explanation, execution results, and hardware specifications.

## Compilation and Execution

1. Ensure that you have the NVIDIA CUDA Toolkit and compatible C++ compiler installed.

2. Compile the CUDA program using the following command:
```
nvcc main.cu -o image_search -std=c++11 pkg-config --cflags --libs opencv4
```

3. Run the compiled program:
```
./image_search
```


## Results and Performance

The project demonstrates the efficiency of parallel programming using CUDA for image search. By utilizing the power of GPUs, the program is capable of significantly reducing execution times compared to the serial and OpenMP versions. The provided documentation (`multicoreproject-CUDA-doc.pdf`) contains detailed performance analysis, including execution times, for different configurations.

## Hardware Specifications

The program was executed and tested on the following hardware:

- Machine Model: MSI GE62 6QD
- Processor: Intel Core i7-6700HQ CPU @ 2.60GHz (4 cores, 8 threads)
- GPU: NVIDIA GTX 960M (4GB GDDR5)
- RAM: 8GB
- Operating System: Ubuntu 20.04

## Conclusion

The project demonstrates the effectiveness of CUDA in accelerating image processing tasks, showcasing the potential for significant speedup when dealing with large datasets and complex computations. The parallel image search program serves as a practical example of leveraging GPU capabilities for data-intensive applications.

For a comprehensive understanding of the project, refer to the documentation provided in `multicoreproject-CUDA-doc.pdf`.
