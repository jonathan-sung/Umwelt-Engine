#version 450

[SpecializationConstant]
[vk::constant_id(0)]
const uint special0;

[SpecializationConstant]
[vk::constant_id(1)]
const uint special1;

[vk::binding(0, 0)]
RWTexture2D<float4> outputImage;

struct PushConstants
{
    uint width;
    uint height;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;

struct Ray
{
    float3 origin;
    float3 direction;
};

struct HitRecord
{
    float3 position;
    float3 normal;
    float t;
};

float3 rayAt(Ray ray, float t)
{
    return ray.origin + (t * ray.direction);
}

float hitSphere(float3 center, float radius, Ray r)
{
    float3 rayToSphereDirection = (center - r.origin);
    float a = dot(r.direction, r.direction);
    float h = dot(r.direction, rayToSphereDirection);
    float c = dot(rayToSphereDirection, rayToSphereDirection) - (radius * radius);

    float discriminant = h * h - a * c;
    if (discriminant < 0)
        return -1;
    else
        return (h - sqrt(discriminant)) / a;
}

float3 getRayColor(Ray r)
{
    float3 sphereCenter = float3(0, 0, -1);
    float sphereRadius = 0.5;
    float t = hitSphere(sphereCenter, sphereRadius, r);
    if (t > 0)
    {
        float3 normal = normalize(rayAt(r, t) - sphereCenter);
        float3 sphereColor = 0.5 * (normal + float3(1, 1, 1));
        return sphereColor;
    }

    float3 unitDirection = normalize(r.direction);
    float a = 0.5 * (unitDirection.y + 1.0);
    float3 backgroundColor = ((1.0 - a) * float3(1, 1, 1)) + (a * float3(0.5, 0.7, 1.0));

    return backgroundColor;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 thread: SV_DispatchThreadID)
{
    float focalLength = 1.0;
    float viewportHeight = 2.0;
    float viewportWidth = viewportHeight * (float(pushConstants.width) / float(pushConstants.height));
    float3 cameraCenter = float3(0, 0, 0);

    float3 viewportU = float3(viewportWidth, 0, 0);
    float3 viewportV = float3(0, -viewportHeight, 0);

    float3 pixelU = viewportU / pushConstants.width;
    float3 pixelV = viewportV / pushConstants.height;

    float3 viewportUpperLeftCorner = cameraCenter - float3(0, 0, focalLength) - (viewportU / 2) - (viewportV / 2);
    float3 pixelUpperLeftCorner = viewportUpperLeftCorner + (0.5 * (pixelU + pixelV));

    // In Slang, use threadID instead of gl_GlobalInvocationID
    uint3 threadID = uint3(thread.xy, 0);
    float3 pixelCenter = pixelUpperLeftCorner + (float(threadID.x) * pixelU) + (float(threadID.y) * pixelV);
    float3 rayDirection = pixelCenter - cameraCenter;
    Ray r = { cameraCenter, rayDirection };
    float3 color = getRayColor(r);

    // Write to RWTexture2D in Slang
    // outputImage[threadID.xy] = float4(color, 1.0);

    outputImage[threadID.xy] = float4(special0, 0.0, special1, 1.0);
}
