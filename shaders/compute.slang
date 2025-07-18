[SpecializationConstant]
[vk::constant_id(0)]
const float special0 = 0.0;

[SpecializationConstant]
[vk::constant_id(1)]
const float special1 = 0.0;

[SpecializationConstant]
[vk::constant_id(2)]
const float special2 = 0.0;

[vk::binding(0, 0)]
RWTexture2D<float4> outputImage;

[vk::binding(1, 0)]
RWStructuredBuffer<float> bufferTest;

struct PushConstants
{
    uint width;
    uint height;
    float time;
    float alpha;
    float3 cameraPosition;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants; // 128 bytes

static const float infinity = float.maxValue;
static const float PI = 3.14159265359;
static uint3 g_thread;
static float randomCounter = 0;

static float degreesToRadians(float degrees)
{
    return (degrees * PI) / 180.0;
}

// pseudo random number generator
static float random()
{
    // Update randomCounter using a Linear Congruential Generator (LCG)

    float2 uv = float2(float(g_thread.x) + randomCounter, float(g_thread.y) + randomCounter);
    randomCounter++;
    return fract(sin(dot(uv.xy, float2(12.9898, 78.233))) * 43758.5453123);
}

static float random(float min, float max)
{
    return min + (max - min) * random();
}

static float3 randomFloat3()
{
    return float3(random(), random(), random());
}

static float3 randomFloat3(float min, float max)
{
    return float3(random(min, max), random(min, max), random(min, max));
}

static float3 randomUnitVectorFloat3()
{
    while (true)
    {
        float3 point = randomFloat3(-1, 1);
        float lengthSquared = dot(point, point);
        if (1e-16 < lengthSquared && lengthSquared <= 1)
            return point / sqrt(lengthSquared);
    }
}

static float3 randomOnHemisphere(const float3 normal)
{
    float3 onUnitSphere = randomUnitVectorFloat3();
    if (dot(normal, onUnitSphere) > 0.0)
        return onUnitSphere;
    else
        return -onUnitSphere;
}

struct Ray
{
    float3 origin;
    float3 direction;
    float3 at(float t)
    {
        return origin + (t * direction);
    }
};

struct HitRecord
{
    float3 position;
    float3 normal;
    float t;
    bool frontFace;

    [mutating]
    void setFaceNormal(const Ray ray, const float3 outwardNormal)
    {
        frontFace = dot(ray.direction, outwardNormal) < 0.0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

struct Interval
{
    float min;
    float max;

    __init()
    {
        min = float.maxValue;
        max = float.minValue;
    }

    __init(float min, float max)
    {
        this.min = min;
        this.max = max;
    }

    float size()
    {
        return max - min;
    }

    bool contains(float x)
    {
        return min <= x && x <= max;
    }

    bool surrounds(float x)
    {
        return min < x && x < max;
    }

    float clamp(float x)
    {
        if (x < min)
            return min;
        if (x > max)
            return max;
        return x;
    }

    static const Interval empty = Interval(float.maxValue, float.minValue);
    static const Interval universe = Interval(float.minValue, float.maxValue);
};

interface IHittable
{
    bool hit(const Ray ray, Interval rayT, inout HitRecord record);
};

struct HittableList : IHittable
{
    IHittable objects[2];

    __init(IHittable objects[2])
    {
        this.objects = objects;
    }

    override bool hit(const Ray ray, Interval rayT, inout HitRecord record)
    {
        HitRecord tempHitRecord;
        bool hasHitSomething = false;
        float closestSoFar = rayT.max;

        // return true;
        for (uint i = 0; i < objects.getCount(); i++)
        {
            if (objects[i].hit(ray, Interval(rayT.min, closestSoFar), tempHitRecord))
            {
                hasHitSomething = true;
                closestSoFar = tempHitRecord.t;
                record = tempHitRecord;
            }
        }

        return hasHitSomething;
    }
};

struct Sphere : IHittable
{
    float3 center;
    float radius;

    __init(float3 center, float radius)
    {
        this.center = center;
        this.radius = max(0, radius);
    }

    override bool hit(const Ray ray, Interval rayT, inout HitRecord record)
    {
        float3 rayToSphereDirection = (center - ray.origin);
        float a = dot(ray.direction, ray.direction);
        float h = dot(ray.direction, rayToSphereDirection);
        float c = dot(rayToSphereDirection, rayToSphereDirection) - (radius * radius);

        float discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        float sqrtd = sqrt(discriminant);

        float root = (h - sqrtd) / a;
        bool rootIsOutsideTheRangeOfTMinAndTMax = !(rayT.surrounds(root));
        if (rootIsOutsideTheRangeOfTMinAndTMax)
        {
            root = (h + sqrtd) / a;                                       // Use the other root
            rootIsOutsideTheRangeOfTMinAndTMax = !(rayT.surrounds(root)); // Check again to see if the new root is outside the range of tmin and tmax
            if (rootIsOutsideTheRangeOfTMinAndTMax)
                return false;
        }

        record.t = root;
        record.position = ray.at(record.t);
        float3 outwardNormal = (record.position - center) / radius;
        record.setFaceNormal(ray, outwardNormal);

        return true;
    }
};

struct Camera
{
    float3 cameraCenter;
    float3 pixelUpperLeftCorner;
    float3 pixelU;
    float3 pixelV;
    uint samplesPerPixel = 100;
    float pixelSamplesScale;
    uint3 thread;

    __init(uint3 thread)
    {
        float focalLength = 1.0;
        float viewportHeight = 2.0;
        float viewportWidth = viewportHeight * (float(pushConstants.width) / float(pushConstants.height));
        cameraCenter = pushConstants.cameraPosition;

        float3 viewportU = float3(viewportWidth, 0, 0);
        float3 viewportV = float3(0, -viewportHeight, 0);

        pixelU = viewportU / pushConstants.width;
        pixelV = viewportV / pushConstants.height;

        float3 viewportUpperLeftCorner = cameraCenter - float3(0, 0, focalLength) - (viewportU / 2) - (viewportV / 2);
        pixelUpperLeftCorner = viewportUpperLeftCorner + (0.5 * (pixelU + pixelV));
        pixelSamplesScale = 1.0 / samplesPerPixel;
        this.thread = thread;
    }

    [mutating]
    void render(const HittableList world)
    {
        float3 color = float3(0.0);
        for (uint sample = 0; sample < samplesPerPixel; sample++)
        {
            Ray ray = getRay(thread.xy);
            color += rayColor(ray, 5, world);
        }
        color *= pixelSamplesScale;

        outputImage[thread.xy] = float4(color, 1.0);
    }

    Ray getRay(uint2 thread)
    {
        float3 offset = sampleSquare();
        float3 pixelCenter = pixelUpperLeftCorner + ((float(thread.x) + offset.x) * pixelU) + ((float(thread.y) + offset.y) * pixelV);
        float3 rayDirection = pixelCenter - cameraCenter;

        Ray rayCameraToPixel = { cameraCenter, rayDirection };
        return rayCameraToPixel;
    }

    float3 sampleSquare()
    {
        return float3(random() - 0.5, random() - 0.5, 0);
    }

    float3 rayColor(const Ray ray, uint maxLightBounces, const HittableList world)
    {
        HitRecord hitRecord = HitRecord();
        Ray currentRay = ray;
        float3 color = float3(1.0);
        for (uint i = 0; i < maxLightBounces; i++)
        {
            bool objectWasHit = world.hit(currentRay, Interval(0.001, float.maxValue), hitRecord);
            if (objectWasHit)
            {
                float3 diffuseDirection = randomOnHemisphere(hitRecord.normal);
                currentRay = Ray(hitRecord.position, diffuseDirection);
                color *= 0.5;
            }
            else
            {
                float3 unitDirection = normalize(currentRay.direction);
                float a = 0.5 * (unitDirection.y + 1.0);
                float3 blueSky = float3(0.5, 0.7, 1.0);
                float3 backgroundColor = ((1.0 - a) * float3(1, 1, 1)) + (a * blueSky);
                color *= backgroundColor;
                return color;
            }
        }
        bool colorNotChanged = all(color == float3(1.0));
        if (colorNotChanged)
            return float3(0.0);
        else
            return color;
    }
};

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 thread: SV_DispatchThreadID)
{
    g_thread = thread;
    // World
    float3 center = float3(0, 0, -1);
    float radius = 0.5;

    Sphere sphere = Sphere(center, radius);
    Sphere sphere2 = Sphere(center + float3(0, -100 - radius, 0), 100);

    IHittable listOfObjects[] = { sphere, sphere2 };
    HittableList world = HittableList(listOfObjects);

    Camera camera = Camera(thread);
    camera.render(world);
}
