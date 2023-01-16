//
// Created by danny on 1/12/2023.
//

//
// This file is part of the course materials for AMATH483/583 at the University of Washington,
// Spring 2020
//
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// Author: Andrew Lumsdaine
//

#include <complex>
#include <cstddef>
#include <vector>
#include <string>
#include <cstdlib>
//#include <iostream>
//#include <omp.h>

#define cimg_display 0
#include "CImg.h"

using namespace cimg_library;
int                  scale   = 2048; //1024;
std::complex<double> origin{0.0, 0.0};

std::complex<double> xy_to_complex(int ix, int iy, double stretch) {
  std::complex<double> v = {double(ix) / double(scale) - 0.5, double(iy) / double(scale) - 0.5};
  return v * stretch + origin;
}

// f(x) = 1/(1+exp(-x+1)) - 1/(1+exp(-x+1))
std::complex<double> I = std::complex<double>(0,1);
std::complex<double> f(std::complex<double> x) { return (I / (std::exp(-x + I) + 1.0) - 1.0 / (std::exp(-x - I) + 1.0) -
                                                   1.0 / (std::exp(-x - 1.0) + 1.0) + I / (std::exp(1.0 - x) + 1.0));}

// f'(x) = exp(-x)/(1+exp(-x+1))^2 - exp(-x-1)/(1+exp(-x-1))^2
std::complex<double> fp(std::complex<double> x) { return
    (I * std::exp(-x + I) / std::pow(std::exp(-x + I) + 1.0, 2) - std::exp(-x - I) / std::pow(std::exp(-x - I) + 1.0, 2) -
      I * std::exp(-x - 1.0) / std::pow(std::exp(-x - 1.0) + 1.0, 2) + std::exp(1.0 - x) / std::pow(std::exp(1.0 - x) + 1.0, 2));}


//std::complex<double> f(std::complex<double> x)  { return (-1.0 / (std::exp(-x - 1.0) + 1.0) + 0.5 / (std::exp(-x - 2.0) + 1.0) - 0.5 / (std::exp(2.0 - x) + 1.0) + 1.0 / (std::exp(1.0 - x) + 1.0));}// f(x) = 1/(1+exp(-x))
//
//std::complex<double> df(std::complex<double> x) { return (-std::exp(-x - 1.0) / std::pow(std::exp(-x - 1.0) + 1.0, 2.0) + 0.5 * std::exp(-x - 2.0) / std::pow(std::exp(-x - 2.0) + 1.0, 2.0)
//                                                          - 0.5 * std::exp(2.0 - x) / std::pow(std::exp(2.0 - x) + 1.0, 2.0) + std::exp(1.0 - x) / std::pow(std::exp(1.0 - x) + 1.0, 2.0));} // f'(x) = exp(-x)/(1+exp(-x))^2

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
  void set_data(size_t index, double value) {
    data[index] = value;
  }
  double operator[](size_t index) {
    return data[index];
  }
};

void make_mp4(std::string image_folder) {
  std::string command = "ffmpeg -i " + image_folder + "/julia_%d.bmp -c:v libx264 -r 30 -pix_fmt yuv420p julia.mp4";
  system(command.c_str());
}


int main() {
  int                 depth = 1, planes = 3;
  unsigned char       white  = 255;
  CImg<unsigned char> julia(scale, scale, depth, planes, white);
  double start = 5.01, stop = 5.02, step = 0.01;
  VectorGenerator stretches(start, stop, step);
//  print stretches to console to check
//  for(int i = 0; i < stretches.size(); i++)
//    std::cout << stretches[i] << std::endl;

  for (int n = 0; n <= stretches.size() - 1; ++n) {
      double stretch = stretches[n];
      //Parallelizing the inner loops using OpenMP
     #pragma omp parallel for collapse(2)
      for (int i = 0; i < scale; ++i) {
        for (int j = 0; j < scale; ++j) {
          std::complex<double> x   = xy_to_complex(i, j, stretch);       // convert screen coord to complex number
          unsigned char        pix = (unsigned char)newt(x);    // get number of iterations
          julia(i, j, 0, 0)        = 13 * pix;                  // red
          julia(i, j, 0, 1)        = 17 * pix;                  // green
          julia(i, j, 0, 2)        = 21 * pix;                  // blue
        }
      }
      std::string file_name = "images/julia_" + std::to_string(stretches[n]) + ".bmp";
      julia.save_bmp(file_name.c_str());
    }
//  }
//    still getting ffmpeg error
//  // Use FFmpeg to convert the image files to a video file
//        make_mp4("images");
  return 0;
}
