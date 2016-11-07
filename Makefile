CC:=g++

all:	gather.cc
	$(CC) -fopenmp -g -Wall -std=c++1y -O3 -march=native -mtune=native gather.cc -lbenchmark -lpthread  -o gather

debug:	gather.cc
	$(CC) -fopenmp -g -Wall -std=c++1y -O0 -march=native -mtune=native gather.cc -lbenchmark -lpthread  -o gather
