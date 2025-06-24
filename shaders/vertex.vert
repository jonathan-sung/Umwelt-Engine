#version 450 core

layout(location = 0) in vec2 inPosition;
// layout(set = 0, binding = 0) uniform VertexBuffer
// {
//     vec2 vertex;
// }
// vertexBuffer;

layout(location = 0) out vec3 fragColor;

void main()
{
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor = vec3(1.0, 0.0, 0.0);
}