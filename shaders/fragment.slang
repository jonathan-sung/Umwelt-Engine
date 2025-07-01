
[vk::binding(0, 0)]
ConstantBuffer<float3> uniformBuffer;

[vk::binding(1, 0)]
Sampler2D presentImage;

struct PushConstants
{
    uint width;
    uint height;
    float time;
};

[vk::push_constant]
ConstantBuffer<PushConstants> pushConstants;

[shader("pixel")]
float4 fragmentMain(in float2 position: SV_Position, in float3 fragColor: COLOR0) : SV_Target0
{
    float2 uv = position.xy / float2(pushConstants.width, pushConstants.height);
    float4 texColor = presentImage.Sample(uv);
    return texColor;
    // return float4(uniformBuffer, 1.0);
    // return float4(fragColor, 1.0);
}
