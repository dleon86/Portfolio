# January 14, 2023
#
# This file was part of the course materials for AMATH483/583 at the University of Washington,
# 
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Re-Author: Daniel Leon
# Original Author: Andrew Lumsdaine
#

CXX       := c++

OPTS      := -g -O3# -DNDEBUG -O0
LANG      := -std=c++11
PICKY     := -Wall

CXXFLAGS  := $(LANG) $(OPTS) $(PICKY)

.PHONY	  : clean all

all		  : julia_video.exe 


LIBS		:= -lgomp
OS		:= $(shell uname -s)
ifeq ($(OS),Linux)
	LIBS	+= -lpthread
endif

julia_video.exe 	: julia_video.o
		  $(CXX) $(CXXFLAGS) julia_video.o -o julia_video $(LIBS)

julia_video.o 	: julia_video.cpp
	   	  $(CXX) $(CXXFLAGS) -Wno-char-subscripts -fopenmp -c julia_video.cpp -o julia_video.o


clean		:
		  /bin/rm -f *~ julia.bmp julia_video.exe julia_video.o \
