# Julia Set Image Generator

This program generates a series of images showing the Julia Set, an example of a fractal, for different stretching factors. The generated images are then combined into a video file.

## Libraries used
The following libraries are used in the program:
- `complex`: For working with complex numbers.
- `cstddef`: For working with sizes of objects.
- `vector`: For storing a list of values.
- `string`: For manipulating strings.
- `cstdlib`: For calling the command line.
- `CImg.h`: For creating and manipulating images.

## Constants
- `scale`: The size of the images to be generated, 2048 pixels.
- `origin`: The origin of the complex plane, (0.0, 0.0).
- `I`: Complex number i, the square root of -1.

## Functions
- `xy_to_complex`: Converts a screen coordinate to a complex number.
- `f`: Calculates the value of the Julia Set for a given complex number.
- `fp`: Calculates the derivative of the function `f`.
- `newt`: Implements the Newton-Raphson method to find a root of the function `f`.
- `VectorGenerator`: A class for generating and storing a list of values.
- `make_mp4`: Combines the generated images into a video file.

## Main Function
The `main` function generates the images and creates the video. It does this by:
1. Initializing the `CImg` object `julia` with white pixels and the dimensions `scale` x `scale`.
2. Initializing the `VectorGenerator` object `stretches` with a list of stretching factors.
3. Looping over the `stretches` list.
4. For each stretching factor, looping over the x and y coordinates of the image.
5. For each coordinate, converting the screen coordinate to a complex number and calculating the value of the Julia Set for that complex number using the `newt` function.
6. Colorizing the pixel at the current coordinate using the calculated value.
7. After all the pixels have been colorized, saving the image using `CImg`.
8. Repeat the process for the next stretching factor.
9. After all images have been generated, calling the `make_mp4` function to combine the images into a video file.

## Parallel Processing
The inner loops in the `main` function are parallelized using OpenMP to improve performance.
