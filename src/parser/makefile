all: test.cpp parser.cpp cnn.pb.cc
	g++ test.cpp cnn.pb.cc parser.cpp -o test `pkg-config --cflags --libs protobuf` -L/usr/lib/x86_64-linux-gnu -llmdb