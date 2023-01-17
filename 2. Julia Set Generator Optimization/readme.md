# README

This program generates a set of images that depict the Julia set, and then uses FFmpeg to convert them into a video. This program was optimized using OpenMP and compiler optimizations to dramatically increase the computation rate by an order of magnitude. Customized Julia sets can be computed by inputting a single variable function and it's derivative, as well as a range for the variable `scale` to iterate over.

## Requirements
- CImg library
- FFmpeg

## Usage
1. Compile the program using a C++ compiler, with the flag `-lX11` and `-lpthread`
2. Run the program
3. The images will be saved in the "images3" folder, and the video file will be saved as "julia.mp4" in the working directory.

## Code Structure
The program is divided into 2 main parts:
- Generating the images
- Using FFmpeg to convert the images into a video

The class `VectorGenerator` is used to create a vector of doubles between a start and stop value, with a given step size. This is used to create a set of stretches for the Julia set.

The function `xy_to_complex` converts an (x, y) pixel location to a complex number.

The function `newt` uses the Newton-Raphson method to find the number of iterations it takes for a given complex number to diverge.

The function `make_mp4` uses the FFmpeg library to convert the set of images into a video file.