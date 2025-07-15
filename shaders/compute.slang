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

static float degreesToRadians(float degrees)
{
    return (degrees * PI) / 180.0;
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
            rootIsOutsideTheRangeOfTMinAndTMax = !(rayT.surrounds(root)); // Check if the new root is outside the range of tmin and tmax
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

    __init()
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
    }

    [mutating]
    void render(uint3 thread, const HittableList world)
    {
        float3 pixelCenter = pixelUpperLeftCorner + (float(thread.x) * pixelU) + (float(thread.y) * pixelV);
        float3 rayDirection = pixelCenter - cameraCenter;
        Ray rayCameraToPixel = { cameraCenter, rayDirection };
        float3 color = rayColor(rayCameraToPixel, world);
        outputImage[thread.xy] = float4(color, 1.0);
    }

    float3 rayColor(const Ray ray, const HittableList world)
    {
        HitRecord hitRecord = HitRecord();
        bool objectWasHit = world.hit(ray, Interval(0, float.maxValue), hitRecord);
        if (objectWasHit)
        {
            float3 colorOfHitPoint = 0.5 * (hitRecord.normal + float3(1.0));
            return colorOfHitPoint;
        }
        else
        {
            float3 unitDirection = normalize(ray.direction);
            float a = 0.5 * (unitDirection.y + 1.0);
            float3 blueSky = float3(0.5, 0.7, 1.0);
            float3 backgroundColor = ((1.0 - a) * float3(1, 1, 1)) + (a * blueSky);
            return backgroundColor;
        }
    }
};

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 thread: SV_DispatchThreadID)
{
    // World
    float3 center = float3(0, 0, -1);
    float radius = 0.5;

    Sphere sphere = Sphere(center, radius);
    Sphere sphere2 = Sphere(center + float3(0, -100 - radius, 0), 100);

    IHittable listOfObjects[] = { sphere, sphere2 };
    HittableList world = HittableList(listOfObjects);

    Camera camera = Camera();
    camera.render(thread, world);
}
