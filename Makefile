CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

all: main.cpp
	g++ $(CFLAGS) *.cpp $(LDFLAGS)

.PHONY: run clean

run: all
	./a.out

clean:
	rm -f a.out
