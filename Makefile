CFLAGS = -std=c++23 -O2
SOURCES = main.cpp
# IMGUI_DIR = imgui
# SOURCES += $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp
# SOURCES += $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_vulkan.cpp
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
NAME = Umwelt 
CXXFLAGS = -std=c++23 -O2 -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends

all: $(NAME)

$(NAME) : $(SOURCES)
	g++ $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $(NAME)

.PHONY: run clean

run: $(NAME)
	./compile_shaders.sh
	mangohud --dlsym ./$(NAME)

clean:
	rm -f $(NAME)
	rm -f shaders/*.spv
