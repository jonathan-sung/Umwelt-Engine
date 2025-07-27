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
    float a;
    float vfov;
    uint samplesPerPixel;
    uint maxRayBounces;
    float focalLength;
    float4 cameraPosition;
    float4 cameraForward;
    float defocusAngle;
    float focusDistance;
    uint randomSeed;
    float dummy[3];
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants; // 128 bytes

// Global variables

static const float infinity = float.maxValue;
static const float PI = 3.14159265359;
static uint3 g_thread;
static float randomCounter = 0;
static uint g_seed = pushConstants.randomSeed; // (g_thread.x - g_thread.y) % uint(pushConstants.time);

// Global functions

static float3 reflect(float3 incident, float3 normal)
{
    return incident - ((2 * dot(incident, normal)) * normal);
}

// incident ray must be normalize before used as input
static float3 refract(float3 incident, float3 normal, float refractionIndexRatio)
{
    float cosTheta = min(dot(-incident, normal), 1.0);
    float3 refractedPerpendicular = refractionIndexRatio * (incident + cosTheta * normal);
    float3 refractedParallel = -sqrt(abs(1 - lengthSquaredFloat3(refractedPerpendicular))) * normal;
    float3 refracted = refractedPerpendicular + refractedParallel;
    return refracted;
}

static float lengthSquaredFloat3(float3 v)
{
    return dot(v, v);
}

static bool nearZeroFloat3(float3 v)
{
    float epsilon = 1e-8;
    return (abs(v.x) < epsilon) && (abs(v.y) < epsilon) && (abs(v.z) < epsilon);
}

static float3 linearToGammaColor(float3 linearColor)
{
    float3 gammaCorrectedColor = float3(0.0);
    if (linearColor.x > 0.0)
        gammaCorrectedColor.x = sqrt(linearColor.x);
    if (linearColor.y > 0.0)
        gammaCorrectedColor.y = sqrt(linearColor.y);
    if (linearColor.z > 0.0)
        gammaCorrectedColor.z = sqrt(linearColor.z);
    return gammaCorrectedColor;
}

// pseudo random number generator
static float random()
{
    // Update randomCounter using a Linear Congruential Generator (LCG)
    uint a = 1664525;
    uint c = 1013904223;
    // uint m = 1 << 16;
    g_seed = ((a * g_seed) + c); // % m;

    float2 uv = float2(float(g_thread.x) + float(g_seed) / 4294967296.0, float(g_thread.y) - float(g_seed) / 4294967296.0);

    // Hash-like transformation to produce a float in [0,1)
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
static float3 randomInUnitDisk()
{
    float r = sqrt(random());
    float theta = 2 * PI * random();
    return float3(r * cos(theta), r * sin(theta), 0.0);
    // while (true)
    // {
    //     float3 p = float3(random(-1, 1), random(-1, 1), 0);
    //     if (lengthSquaredFloat3(p) < 1)
    //         return p;
    // }
}

static float3 randomOnHemisphere(const float3 normal)
{
    float3 onUnitSphere = randomUnitVectorFloat3();
    if (dot(normal, onUnitSphere) > 0.0)
        return onUnitSphere;
    else
        return -onUnitSphere;
}

// Interfaces

interface IMaterial
{
    bool scatter(const Ray rayIncident, const HitRecord record, inout float3 attenuation, inout Ray scattered)
    {
        return false;
    }
};

interface IHittable
{
    bool hit(const Ray ray, Interval rayT, inout HitRecord record);
};

// Structs

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
    IMaterial material;
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

struct HittableList : IHittable
{
    IHittable objects[8];

    __init(IHittable objects[8])
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
    IMaterial material;

    __init(float3 center, float radius, IMaterial material)
    {
        this.center = center;
        this.radius = max(0, radius);
        this.material = material;
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
        record.material = material;

        return true;
    }
};

struct Camera
{
    float3 cameraCenter;
    float3 cameraForward = float3(0.0, 0.0, -1.0);
    float3 vectorUp = float3(0.0, 1.0, 0.0);
    float3 pixelUpperLeftCorner;
    float3 pixelU;
    float3 pixelV;
    uint samplesPerPixel;
    float pixelSamplesScale;
    float vfov;
    uint3 thread;
    float3 u, v, w; // basis vectors

    float defocusAngle = 0.0;
    float focusDistance = 10.0;
    float focalLength = 1.0;
    float3 defocusDiskU, defocusDiskV;

    __init(uint3 thread)
    {
        cameraCenter = pushConstants.cameraPosition.xyz;
        cameraForward = pushConstants.cameraForward.xyz;
        defocusAngle = pushConstants.defocusAngle;
        focusDistance = pushConstants.focusDistance;
        focalLength = pushConstants.focalLength; // length(lookFrom - cameraLookAt);

        // cameraLookAt = float3(0.0, 0.0, -1.0);
        samplesPerPixel = pushConstants.samplesPerPixel;
        vfov = pushConstants.vfov;
        float3 lookFrom = cameraCenter;

        // Determine viewport dimensions
        float theta = radians(pushConstants.vfov);
        float h = tan(theta / 2);
        float viewportHeight = 2.0 * h * focusDistance;
        float viewportWidth = viewportHeight * (float(pushConstants.width) / float(pushConstants.height));

        // Calculate basis vectors
        w = normalize(-cameraForward);
        u = normalize(cross(vectorUp, w));
        v = cross(w, u);

        float3 viewportU = viewportWidth * u;
        float3 viewportV = viewportHeight * -v;

        pixelU = viewportU / pushConstants.width;
        pixelV = viewportV / pushConstants.height;

        float3 viewportUpperLeftCorner = cameraCenter - (focusDistance * w) - (viewportU / 2) - (viewportV / 2);
        pixelUpperLeftCorner = viewportUpperLeftCorner + (0.5 * (pixelU + pixelV));

        // Calculate defocus stuff
        float defocusRadius = focusDistance * tan(radians(defocusAngle / 2));
        defocusDiskU = u * defocusRadius;
        defocusDiskV = v * defocusRadius;

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
            color += rayColor(ray, pushConstants.maxRayBounces, world);
        }
        color *= pixelSamplesScale;
        color = linearToGammaColor(color);
        outputImage[thread.xy] = float4(color, 1.0);
    }

    Ray getRay(uint2 thread)
    {
        float3 offset = sampleSquare();
        float3 pixelCenter = pixelUpperLeftCorner + ((float(thread.x) + offset.x) * pixelU) + ((float(thread.y) + offset.y) * pixelV);
        float3 rayOrigin = cameraCenter;
        if (defocusAngle > 0.0)
            rayOrigin = defocusDiskSample();
        float3 rayDirection = pixelCenter - rayOrigin;
        Ray rayCameraToPixel = Ray(rayOrigin, rayDirection);
        return rayCameraToPixel;
    }

    float3 sampleSquare()
    {
        return float3(random() - 0.5, random() - 0.5, 0);
    }

    float3 defocusDiskSample()
    {
        float3 p = randomInUnitDisk();
        return cameraCenter + (p.x * defocusDiskU) + (p.y * defocusDiskV);
    }

    float3 rayColor(const Ray ray, uint maxRayBounces, const HittableList world)
    {
        HitRecord hitRecord = HitRecord();
        Ray currentRay = ray;
        float3 color = float3(1.0);
        bool anyHit = false;
        for (uint i = 0; i < maxRayBounces; i++)
        {
            bool objectWasHit = world.hit(currentRay, Interval(0.001, float.maxValue), hitRecord);
            if (objectWasHit)
            {
                anyHit = true;
                Ray scattered;
                float3 attenuation = float3(1.0);
                if (hitRecord.material.scatter(currentRay, hitRecord, attenuation, scattered))
                    color *= attenuation;
                else
                {
                    color *= 0.0;
                    break;
                }

                currentRay = Ray(hitRecord.position, scattered.direction);
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
        if (!anyHit)
            return float3(0.0); // nothing hit
        return color;
    }
};

// Materials

struct Lambertian : IMaterial
{
    float3 albedo;

    __init(const float3 albedo)
    {
        this.albedo = albedo;
    }

    override bool scatter(const Ray rayIncident, const HitRecord record, inout float3 attenuation, inout Ray scattered)
    {
        float3 scatteredDirection = record.normal + randomUnitVectorFloat3();
        if (nearZeroFloat3(scatteredDirection))
            scatteredDirection = record.normal;
        scattered = Ray(record.position, scatteredDirection);
        attenuation = albedo;
        return true;
    }
};

struct Metal : IMaterial
{
    float3 albedo;
    float fuzz;

    __init(const float3 albedo, const float fuzz)
    {
        this.albedo = albedo;
        this.fuzz = fuzz;
    }

    override bool scatter(const Ray rayIncident, const HitRecord record, inout float3 attenuation, inout Ray scattered)
    {
        float3 reflected = reflect(rayIncident.direction, record.normal);
        reflected = normalize(reflected) + (fuzz * randomUnitVectorFloat3());
        scattered = Ray(record.position, reflected);
        attenuation = albedo;
        return (dot(scattered.direction, record.normal) > 0.0);
    }
};

struct Dielectric : IMaterial
{
    float refractionIndex;

    __init(float refractionIndex)
    {
        // ratio: refractive index of material / refractive index of surrounding medium
        // e.g. 1.5/1.0 would mean the material is made of glass (1.5), and it is surrounded by air since the refractive index of air is very close to 1.0
        this.refractionIndex = refractionIndex;
    }

    override bool scatter(const Ray rayIncident, const HitRecord record, inout float3 attenuation, inout Ray scattered)
    {
        attenuation = float3(1.0);
        float ri = record.frontFace ? (1.0 / refractionIndex) : refractionIndex;

        float3 unitDirection = normalize(rayIncident.direction);
        float cosTheta = min(dot(-unitDirection, record.normal), 1.0);
        float sinTheta = sqrt(1 - cosTheta * cosTheta);

        bool internalReflection = ri * sinTheta > 1.0;
        bool schlick = reflectance(cosTheta, ri) > random();
        float3 refractedDirection;

        // angle is very steep so instead of refracting the light, it reflects it
        if (schlick)
            refractedDirection = reflect(unitDirection, record.normal);
        else
            refractedDirection = refract(unitDirection, record.normal, ri);

        scattered = Ray(record.position, refractedDirection);
        return true;
    }

    // schlick approximation
    float reflectance(float cosine, float refractionIndex)
    {
        float r0 = (1 - refractionIndex) / (1 + refractionIndex);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
}

void renderCoverScene()
{
}

void renderTestScene()
{
    // World

    Lambertian materialGround = Lambertian(float3(0.8, 0.8, 0.0));
    Lambertian materialCenter = Lambertian(float3(0.1, 0.2, 0.5));
    Metal materialLeft = Metal(float3(0.8), pushConstants.a);
    Metal materialFront = Metal(float3(0.2, 0.1, 0.2), 0.0);
    Metal materialRight = Metal(float3(0.8, 0.6, 0.2), 1.0 - pushConstants.a);
    Dielectric materialGlass = Dielectric(1.5);
    Dielectric materialBubble = Dielectric(1.0 / 1.5);

    Sphere ground = Sphere(float3(0.0, -100.5, -1.0), 100.0, materialGround);
    Sphere sphereCenter = Sphere(float3(0.0, 0.0, -1.2), 0.5, materialCenter);
    Sphere sphereLeft = Sphere(float3(-1.0, 0.0, -1.0), 0.5, materialLeft);
    Sphere sphereRight = Sphere(float3(1.0, 0.0, -1.0), 0.5, materialRight);
    Sphere sphereGlass = Sphere(float3(1.0, 0.0, 2.0), 0.5, materialGlass);
    Sphere sphereBubble = Sphere(float3(1.0, 0.0, 2.0), 0.49, materialBubble);
    Sphere sphereGlass2 = Sphere(float3(-1.0, 0.0, 2.0), 0.5, materialGlass);
    Sphere sphereFront = Sphere(float3(0.0, 2.0, 1.0), 0.25, materialFront);

    IHittable listOfObjects[] = { ground, sphereCenter, sphereLeft, sphereRight, sphereGlass, sphereBubble, sphereGlass2, sphereFront };
    HittableList world = HittableList(listOfObjects);

    Camera camera = Camera(g_thread);
    camera.render(world);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 thread: SV_DispatchThreadID)
{
    g_thread = thread;
    renderTestScene();
}
