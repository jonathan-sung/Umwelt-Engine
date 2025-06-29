#version 450 core
layout(set = 0, binding = 1) uniform sampler2D presentImage;
layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform PushConstants
{
    uint width;
    uint height;
}
pc;
void main()
{
    vec2 uv = gl_FragCoord.xy / vec2(pc.width, pc.height); // Adjust to your resolution
    vec4 texColor = texture(presentImage, uv);
    outColor = vec4(texColor.rgb, 1.0);
    // outColor = vec4(fragColor, 1.0);
}