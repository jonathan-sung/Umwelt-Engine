[shader("vertex")]
float4 main(float2 inPosition: POSITION0, float3 inColor: COLOR0, out float3 fragColor: COLOR0) : SV_Position
{
    fragColor = inColor;
    return float4(inPosition, 0.0, 1.0);
}
