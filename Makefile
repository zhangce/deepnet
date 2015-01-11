CC=g++
CFLAGS=-O2 -std=c++0x
LIBFLAGSEND=-lrt
BUFFLAGS=`pkg-config --cflags --libs protobuf` -L/usr/lib/x86_64-linux-gnu -llmdb
BLASFLAGS=-lblas -lm -I/lfs/local/0/amir/software/openblas/OpenBLAS/

all: src/Main.cpp src/parser/cnn.pb.cc src/parser/parser.cpp
	g++ $(CFLAGS) -o main src/parser/cnn.pb.cc src/parser/parser.cpp src/Main.cpp $(BUFFLAGS) $(LIBFLAGSEND) $(BLASFLAGS)
