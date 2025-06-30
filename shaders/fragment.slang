
layout(set = 0, binding = 1) Sampler2D presentImage;

struct PushConstants
{
    uint width;
    uint height;
    float time;
};

[vk::push_constant]
ConstantBuffer<PushConstants> pushConstants;

[shader("pixel")]
float4 main(float2 position: SV_Position, float3 fragColor: COLOR0) : SV_Target0
{
    float2 uv = position.xy / float2(pushConstants.width, pushConstants.height);
    float4 texColor = presentImage.Sample(uv);
    return texColor;
    // return float4(fragColor, 1.0);
}
