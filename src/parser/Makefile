all: runtest.cpp parser.cpp cnn.pb.cc
	g++ runtest.cpp cnn.pb.cc parser.cpp -o runtest `pkg-config --cflags --libs protobuf` -L/usr/lib/x86_64-linux-gnu -llmdb