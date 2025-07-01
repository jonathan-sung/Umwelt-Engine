# Path Tracing with Vulkan

## Introduction

This project demonstrates a simple path tracing implementation using Vulkan. It consists of two main components:

1. A compute shader that performs path tracing and outputs the result to a `VkImage`.
2. A graphics pipeline that fills the screen with two triangles and samples a texture, which is the `VkImage` produced by the compute shader.

## Building

### Ubuntu/Debian

To build this project on Ubuntu or Debian, you need to have the following dependencies installed:

```bash
sudo apt install build-essential cmake vulkan-tools libvulkan-dev vulkan-utils vulkan-validationlayers-dev spirv-tools libglfw3-dev libglm-dev glslc
```

Then, you can build and run the project using the following command:

```bash
make run
```

## References

- [Ray Tracing in One Weekend Series](https://raytracing.github.io/)
- [Vulkan Programming Guide](https://a.co/d/duwHfyu)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [Vulkan Samples](https://github.com/SaschaWillems/Vulkan)
- [GLFW Documentation](https://www.glfw.org/docs/latest/)
- [Slang Documentation](https://shader-slang.github.io/)