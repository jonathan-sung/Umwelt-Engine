#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>
#include <set>
#include <stdexcept>
#include <sys/types.h>
#include <vector>

#ifdef NDEBUG
bool validationLayersEnabled = false;
#else
bool validationLayersEnabled = true;
#endif

class Game
{
public:
    Game(const uint32_t width, const uint32_t height) : m_width(width), m_height(height)
    {
        // std::clog << "game object created!" << std::endl;
        initWindow();
        initVulkan();
    }

    ~Game()
    {
        // std::clog << "game object destroyed!" << std::endl;
        cleanup();
    }

    void run()
    {
        mainLoop();
    }

private:
    // Vulkan member variables
    VkInstance m_instance;
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    VkQueue m_graphicsQueue;
    VkQueue m_presentQueue;

    VkSurfaceKHR m_surface;

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete()
        {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    struct SwapChainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    const std::vector<const char *> validationLayers{
        "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<const char *> deviceExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    // General member variables
    uint32_t m_width;
    uint32_t m_height;

    // GLFW member variables
    GLFWwindow *m_window;

    void initWindow()
    {
        if (glfwInit() == GLFW_FALSE)
            throw std::runtime_error("GLFW initlization failed!");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        m_window = glfwCreateWindow(m_width, m_height, "Vulkan Fun 2", nullptr, nullptr);

        if (m_window == nullptr)
            throw std::runtime_error("failed to create GLFW window!");

        if (glfwVulkanSupported() == GLFW_FALSE)
            throw std::runtime_error("GLFW Vulkan not supported!");
    }

    void initVulkan()
    {
        createInstance();
        createSurface();
        choosePhysicalDevice();
        createLogicalDevice();
    }

    void createSurface()
    {
        if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create surface!");
        }
    }

    void createLogicalDevice()
    {
        if (m_physicalDevice == VK_NULL_HANDLE)
            throw std::runtime_error("logical device creation failed: physical device is null");
        QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);

        if (!indices.isComplete())
            throw std::runtime_error("not all required queue families can be found!");

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                nullptr,
                0,
                queueFamily,
                1,
                &queuePriority
            };
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkDeviceCreateInfo deviceInfo{
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(queueCreateInfos.size()),
            queueCreateInfos.data(),
            0,
            nullptr,
            static_cast<uint32_t>(deviceExtensions.size()),
            deviceExtensions.data(),
            nullptr
        };

        if (vkCreateDevice(m_physicalDevice, &deviceInfo, nullptr, &m_device) != VK_SUCCESS)
            throw std::runtime_error("failed to create logical device!");

        vkGetDeviceQueue(m_device, indices.graphicsFamily.value(), 0, &m_graphicsQueue);
        vkGetDeviceQueue(m_device, indices.presentFamily.value(), 0, &m_presentQueue);
    }

    void choosePhysicalDevice()
    {
        if (m_instance == VK_NULL_HANDLE)
            throw std::runtime_error("physical device selection failed: invalid instance");

        uint32_t physicalDeviceCount = 0;
        vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr);
        std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
        vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data());

        // Choose most powerful physical device
        float maxScore = 0;
        for (VkPhysicalDevice physicalDevice : physicalDevices)
        {
            float score = getPhysicalDeviceScore(physicalDevice);
            if (score > maxScore)
            {
                maxScore = score;
                m_physicalDevice = physicalDevice;
            }
        }

        if (m_physicalDevice == VK_NULL_HANDLE)
            throw std::runtime_error("failed to choose physical device!");

        if (!isDeviceSuitable(m_physicalDevice))
            throw std::runtime_error("physical device is not suitable!");
    }

    void createInstance()
    {
        // Get extensions
        std::vector<const char *> extensions;
        std::vector<std::string> extensionsStrings;

        extensionsStrings = getExtensions();
        extensions.resize(extensionsStrings.size());

        for (uint32_t i = 0; i < extensionsStrings.size(); i++)
            extensions[i] = extensionsStrings[i].c_str();

        // Get layers
        std::vector<const char *> layers;
        std::vector<std::string> layersStrings;

        layersStrings = getLayers();
        layers.resize(layersStrings.size());

        for (uint32_t i = 0; i < layersStrings.size(); i++)
            layers[i] = layersStrings[i].c_str();

        // App info
        VkApplicationInfo applicationInfo{
            VK_STRUCTURE_TYPE_APPLICATION_INFO,
            nullptr,
            "Application Name",
            VK_MAKE_VERSION(0, 1, 0),
            "Engine Name",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_3
        };

        VkInstanceCreateInfo instanceInfo{
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            nullptr,
            0,
            &applicationInfo,
            static_cast<uint32_t>(layers.size()),
            layers.data(),
            static_cast<uint32_t>(extensions.size()),
            extensions.data()
        };

        if (vkCreateInstance(&instanceInfo, nullptr, &m_instance) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance!");
        }
    }
    void mainLoop()
    {
        while (!glfwWindowShouldClose(m_window))
        {
            glfwPollEvents();
        }
    }
    void cleanup()
    {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        vkDestroyDevice(m_device, nullptr);
        vkDestroyInstance(m_instance, nullptr);
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    std::vector<std::string> getExtensions()
    {
        std::vector<std::string> extensions;
        uint32_t extensionsCount = 0;

        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionsCount);
        std::vector<const char *> requiredExtensions(glfwExtensions, glfwExtensions + extensionsCount);

        if (validationLayersEnabled)
            requiredExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        bool allRequiredExtensionsSupported = checkExtensionsAvailable(requiredExtensions);

        if (!allRequiredExtensionsSupported)
            throw std::runtime_error("not all requested extensions supported by vulkan!");

        extensions.resize(requiredExtensions.size());

        for (uint32_t i = 0; i < requiredExtensions.size(); i++)
        {
            extensions[i] = std::string(requiredExtensions[i]);
        }
        return extensions;
    }

    bool checkExtensionsAvailable(const std::vector<const char *> &requiredExtensions)
    {
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());
        for (const char *requiredExtension : requiredExtensions)
        {
            bool extensionFound = false;
            for (VkExtensionProperties currentExtension : availableExtensions)
            {
                // std::clog << requiredExtension << ":" <<  currentExtension.extensionName << std::endl;
                if (strcmp(requiredExtension, currentExtension.extensionName) == 0)
                {
                    // std::clog << "Extension: " << currentExtension.extensionName << " is supported" << std::endl;
                    extensionFound = true;
                    break;
                }
            }
            if (!extensionFound)
                return false;
        }

        return true;
    }

    std::vector<std::string> getLayers()
    {
        std::vector<std::string> layers;
        std::vector<const char *> requiredLayers(validationLayers);

        if (!checkAllLayersAvailable(requiredLayers))
            throw std::runtime_error("requested layer not available!");

        layers.resize(requiredLayers.size());
        for (uint32_t i = 0; i < requiredLayers.size(); i++)
        {
            layers[i] = std::string(requiredLayers[i]);
        }

        return layers;
    }

    bool checkAllLayersAvailable(const std::vector<const char *> &requiredLayers)
    {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char *requiredLayer : requiredLayers)
        {
            bool layerFound = false;
            for (VkLayerProperties layer : availableLayers)
            {
                // std::clog << requiredLayer << ":" << layer.layerName << std::endl;
                if (strcmp(requiredLayer, layer.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound)
                return false;
        }
        return true;
    }

    float getPhysicalDeviceScore(const VkPhysicalDevice &physicalDevice)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        // ChatGPTs scoring system
        uint32_t computeParallelism = properties.limits.maxComputeWorkGroupInvocations;
        uint32_t sharedMemoryKB = properties.limits.maxComputeSharedMemorySize / 1024; // Convert to KB
        float computeScore = (computeParallelism * sharedMemoryKB) / 1000.0f;

        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            computeScore *= 2.0f; // Arbitrary boost for dedicated GPUs
        }

        // std::clog << properties.deviceName << ":" << computeScore << std::endl;
        return computeScore;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice)
    {
        QueueFamilyIndices indices;
        uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, queueFamilyProperties.data());

        for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
        {
            // check for graphics queue family
            if (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                indices.graphicsFamily = i;

            // check for presentation queue family
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, m_surface, &presentSupport);
            if (presentSupport)
            {
                indices.presentFamily = i;
                // std::clog << "present supported: " << i << std::endl;
            }
            if (indices.isComplete())
                return indices;
        }

        return indices;
    }

    bool checkDeviceExtensionsSupport(VkPhysicalDevice physicalDevice)
    {
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        for (auto extension : availableExtensions)
        {
            requiredExtensions.erase(std::string(extension.extensionName));
        }

        return requiredExtensions.empty();
    }

    SwapChainSupportDetails querySwapChainSupportDetails(VkPhysicalDevice physicalDevice)
    {
        SwapChainSupportDetails details;

        // get capabilities
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, m_surface, &details.capabilities);

        // get formats
        uint32_t surfaceFormatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, m_surface, &surfaceFormatCount, nullptr);
        details.formats.resize(surfaceFormatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, m_surface, &surfaceFormatCount, details.formats.data());

        // get present modes
        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, m_surface, &presentModeCount, nullptr);
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, m_surface, &presentModeCount, details.presentModes.data());

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice physicalDevice)
    {
        bool extensionsSupported = checkDeviceExtensionsSupport(physicalDevice);

        if (!extensionsSupported)
            throw std::runtime_error("device extensions not supported!");

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        if (!indices.isComplete())
            throw std::runtime_error("requested queue families not available!");

        bool swapChainAdequate = false;
        if (extensionsSupported)
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupportDetails(physicalDevice);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && swapChainAdequate && extensionsSupported;
    }
};

int main()
{
    std::unique_ptr<Game> game = std::make_unique<Game>(800, 600);
    game->run();
    return EXIT_SUCCESS;
}