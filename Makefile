CFLAGS = -std=c++23 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
NAME = Umwelt 

all: $(NAME)

$(NAME) : main.cpp
	g++ $(CFLAGS) *.cpp $(LDFLAGS) -o $(NAME)

.PHONY: run clean

run: $(NAME)
	./compile_shaders.sh
	mangohud --dlsym ./$(NAME)

clean:
	rm -f $(NAME)
	rm -f shaders/*.spv
