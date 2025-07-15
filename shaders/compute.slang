[require(SPV_KHR_non_semantic_info)]
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

static const float infinity = 1.0 / 0.0;
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

interface IHittable
{
    bool hit(const Ray ray, float tMin, float tMax, inout HitRecord record);
};

struct IHittableList : IHittable
{
    IHittable objects[2];

    __init(IHittable objects[2])
    {
        this.objects = objects;
    }

    override bool hit(const Ray ray, float tMin, float tMax, inout HitRecord record)
    {
        HitRecord tempHitRecord;
        bool hasHitSomething = false;
        float closestSoFar = tMax;

        // return true;
        for (uint i = 0; i < objects.getCount(); i++)
        {
            if (objects[i].hit(ray, tMin, closestSoFar, tempHitRecord))
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

    override bool hit(const Ray ray, float tMin, float tMax, inout HitRecord record)
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
        bool rootIsOutsideTheRangeOfTMinAndTMax = (root <= tMin || root >= tMax);
        if (rootIsOutsideTheRangeOfTMinAndTMax)
        {
            root = (h + sqrtd) / a;                                              // Use the other root
            rootIsOutsideTheRangeOfTMinAndTMax = (root <= tMin || root >= tMax); // Check if the new root is outside the range of tmin and tmax
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

float3 getRayColorSingleObject(Ray ray, const IHittable object)
{
    HitRecord hitRecord = HitRecord();
    bool isHit = object.hit(ray, 0.1, 1000.0, hitRecord);
    if (isHit)
    {
        float3 colorOfHitPoint = pushConstants.alpha * (hitRecord.normal + float3((1.0 / pushConstants.alpha) - 1));
        return colorOfHitPoint;
    }

    float3 unitDirection = normalize(ray.direction);
    float a = 0.5 * (unitDirection.y + 1.0);
    float3 backgroundColor = ((1.0 - a) * float3(1, 1, 1)) + (a * float3(0.5, 0.7, 1.0));

    return backgroundColor;
}

float3 getRayColorList(Ray ray, const IHittable world[])
{
    HitRecord hitRecord = HitRecord();
    for (uint i = 0; i < world.getCount(); i++)
    {
        bool isHit = world[i].hit(ray, 0.1, 1000.0, hitRecord);
        if (isHit)
        {
            float3 colorOfHitPoint = pushConstants.alpha * (hitRecord.normal + float3((1.0 / pushConstants.alpha) - 1));
            return colorOfHitPoint;
        }
    }

    float3 unitDirection = normalize(ray.direction);
    float a = 0.5 * (unitDirection.y + 1.0);
    float3 backgroundColor = ((1.0 - a) * float3(1, 1, 1)) + (a * float3(bufferTest[0], 0.7, 1.0));

    return backgroundColor;
}

float3 getRayColorWorld(Ray ray, const IHittableList world)
{
    HitRecord hitRecord = HitRecord();
    bool isHit = world.hit(ray, 0.1, 1000.0, hitRecord);
    if (isHit)
    {
        float3 colorOfHitPoint = pushConstants.alpha * (hitRecord.normal + float3((1.0 / pushConstants.alpha) - 1));
        // float3 colorOfHitPoint = float3(pushConstants.alpha);
        return colorOfHitPoint;
    }

    float3 unitDirection = normalize(ray.direction);
    float a = 0.5 * (unitDirection.y + 1.0);
    float3 backgroundColor = ((1.0 - a) * float3(1, 1, 1)) + (a * float3(0.5, 0.7, 1.0));

    return backgroundColor;
}

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
    IHittableList world = IHittableList(listOfObjects);

    // Camera
    float focalLength = 1.0;
    float viewportHeight = 2.0;
    float viewportWidth = viewportHeight * (float(pushConstants.width) / float(pushConstants.height));
    float3 cameraCenter = pushConstants.cameraPosition;

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
    Ray rayCameraToPixel = { cameraCenter, rayDirection };
    float3 color = getRayColorWorld(rayCameraToPixel, world);
    // float3 color = getRayColorList(rayCameraToPixel, listOfObjects);
    // float3 color = getRayColor2(rayCameraToPixel, sphere);

    // Write to RWTexture2D in Slang
    outputImage[threadID.xy] = float4(color, 1.0);

    // outputImage[threadID.xy] = float4(special0, special1, special2, 1.0);
}
