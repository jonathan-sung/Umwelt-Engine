CFLAGS = -std=c++2b -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
NAME = VulkanFun

all: VulkanFun

VulkanFun : main.cpp 
	g++ $(CFLAGS) *.cpp $(LDFLAGS) -o $(NAME)

.PHONY: run clean

run: VulkanFun 
	mangohud --dlsym ./$(NAME)

clean:
	rm -f $(NAME)
	rm -f shaders/*.spv
