//
// Created by danny on 1/12/2023.
//
//
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// Author: Daniel Leon
// some code borrowed from Andrew Lumsdaine's course on HPC at UW, 2020
//
#include <complex>
#include <cstddef>
#include <vector>
#include <string>

#define cimg_display 0
#include "CImg.h"

using namespace cimg_library;

const int scale = 2048;
std::complex<double> origin{0.0, 0.0};

std::complex<double> xy_to_complex(int ix, int iy, double stretch) {
    std::complex<double> v = {double(ix) / double(scale) - 0.5, double(iy) / double(scale) - 0.5};
    return v * stretch + origin;
}

std::complex<double> I = std::complex<double>(0,1);
std::complex<double> f(std::complex<double> x) {
    return (I / (std::exp(-x + I) + 1.0)   - 1.0 / (std::exp(-x - I) + 1.0) -
          1.0 / (std::exp(-x - 1.0) + 1.0) +   I / (std::exp(1.0 - x) + 1.0));
}

std::complex<double> fp(std::complex<double> x) {
    return (I * std::exp(-x + 1.0) / std::pow(std::exp(-x + 1.0) + 1.0, 2) -
                std::exp(-x - I  ) / std::pow(std::exp(-x - I  ) + 1.0, 2) -
            I * std::exp(-x - 1.0) / std::pow(std::exp(-x - 1.0) + 1.0, 2) +
                std::exp(-x + I  ) / std::pow(std::exp(-x + I  ) + 1.0, 2));
}

int newt(std::complex<double> x0) {
    for (int i = 0; i < 256; ++i) {
        std::complex<double> delta = -f(x0) / fp(x0);
        if (std::abs(delta) < 0.00001) {
            return i;
        }
        x0 += delta;
    }

    return 255;
}

class VectorGenerator {
private:
    std::vector<double> data;
public:
VectorGenerator(double start, double stop, double step) {
    int size = int((stop - start) / step) + 1;
    data.resize(size);
    double current_value = start;
    for(int i = 0; i < size; ++i) {
        data[i] = current_value;
        current_value += step;
    }
}
int size() {
    return data.size();
}

    __attribute__((unused)) void set_data(size_t index, double value) {
    data[index] = value;
}
double operator[](size_t index) {
    return data[index];
}
};

//void make_mp4(const std::string& image_folder) {
//    std::string output_file = image_folder + "/julia.mp4";
//    std::string command = "ffmpeg -i " + image_folder + "/julia_%d.bmp -c:v libx264 -r 5 -pix_fmt yuv420p " + output_file;
//    system(command.c_str());
//}


int main() {
    int                 depth = 1, planes = 3;
    unsigned char       white  = 255;
    CImg<unsigned char> julia(scale, scale, depth, planes, white);
    double start = 0.1, stop = 1.0, step = 0.1;
    VectorGenerator stretches(start, stop, step);
    CImg<unsigned char> frame(scale, scale, 1, 3);

    for (int n = 0; n <= stretches.size() - 1; ++n) {
        double stretch = stretches[n];
        //Parallelizing the inner loops using OpenMP
        #pragma omp parallel for
        for (int x = 0; x < scale; ++x) {
            for (int i = 0; i < scale; ++i) {
              std::complex<double> c = xy_to_complex(x, i, stretch);   // convert screen coord to complex number
              auto         color = (unsigned char)newt(c);    // get number of iterations
              julia(x, i, 0, 0)        = 1 * color;                  // red    13
              julia(x, i, 0, 1)        = 250 * color;                  // green  17
              julia(x, i, 0, 2)        = 195 * color;                  // blue   21
            }
        }

        // Save the image using CImg
        char filename[100];
        sprintf(filename,"images/julia_%s.bmp", std::to_string(stretches[n]).c_str());
        julia.save_bmp(filename);
    }

//// Use FFmpeg to convert the image files to a video file
//    std::string image_folder = "images";
////    void make_mp4(std::string image_folder);
//
//    make_mp4(image_folder);

    return 0;
}
