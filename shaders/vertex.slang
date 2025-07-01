
// [shader("vertex")]
// float4 vertexMain(in float2 inPosition, in out float3 color: COLOR0) : SV_Position
// {
//     return float4(inPosition, 0.0, 1.0);
// }

[shader("vertex")]
void vertexMain(in float2 inPosition, in out float3 color: COLOR0, out float4 outPosition: SV_Position)
{
    outPosition = float4(inPosition, 0.0, 1.0);
}
