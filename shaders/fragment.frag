#version 450 core
layout(set = 0, binding = 1) uniform sampler2D presentImage;
layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;
void main()
{
    vec2 uv = gl_FragCoord.xy / vec2(2560.0, 1440.0); // Adjust to your resolution
    vec4 texColor = texture(presentImage, uv);
    outColor = vec4(texColor.rgb, 1.0);
    // outColor = vec4(fragColor, 1.0);
}