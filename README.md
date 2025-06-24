# Path Tracing with Vulkan

This project demonstrates a simple path tracing implementation using Vulkan. It consists of two main components:

1. A compute shader that performs path tracing and outputs the result to a `VkImage`.
2. A graphics pipeline that fills the screen with two triangles and samples a texture, which is the `VkImage` produced by the compute shader.
