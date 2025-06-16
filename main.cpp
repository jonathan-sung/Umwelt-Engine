#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
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
        initWindow();
        initVulkan();
    }

    ~Game()
    {
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
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    // Pipeline
    std::vector<VkPipeline> computePipelines{};
    VkPipelineLayout computePipelineLayout;
    VkDescriptorSetLayout computeDescriptorSetLayout;

    VkSurfaceKHR m_surface;
    VkSwapchainKHR m_swapChain;
    std::vector<VkImage> m_swapChainImages;
    std::vector<VkImageView> m_swapChainImageViews;
    VkFormat m_swapChainFormat;
    VkExtent2D m_swapChainExtent;

    // Memory stuff
    VkBuffer m_frameBuffer;
    VkBufferView m_frameBufferView;
    VkDeviceMemory m_memoryBlock;

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;
        std::optional<uint32_t> computeFamily;

        bool isComplete()
        {
            return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value();
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
        createSwapChain();
        createImageViews();
        createManualFramebuffer();
        createComputePipeline();
        createCommandPool();
        testOutComputePipeline();
    }

    void testOutComputePipeline()
    {
        QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);

        VkCommandBufferBeginInfo beginInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            0,
            nullptr
        };

        vkBeginCommandBuffer(commandBuffers[0], &beginInfo);
        vkCmdBindPipeline(commandBuffers[0], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelines[0]);
        vkCmdDispatch(commandBuffers[0], 64, 1, 1); // a linear work group of 64x1x1 invocations
        vkEndCommandBuffer(commandBuffers[0]);

        vkDeviceWaitIdle(m_device);
    }

    void createCommandPool()
    {
        QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);

        VkCommandPoolCreateInfo commandPoolInfo{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            0,
            indices.computeFamily.value()
        };

        if (vkCreateCommandPool(m_device, &commandPoolInfo, nullptr, &commandPool) != VK_SUCCESS)
            throw std::runtime_error("failed to create command pool!");

        VkCommandBufferAllocateInfo allocInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            commandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            1
        };

        commandBuffers.resize(allocInfo.commandBufferCount);

        vkAllocateCommandBuffers(m_device, &allocInfo, commandBuffers.data());
    }

    VkShaderModule createShaderModule(const std::string &fileName)
    {
        std::vector<char> code = readFile(fileName);

        VkShaderModule shaderModule;

        // create shader module
        VkShaderModuleCreateInfo shaderModuleInfo{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            nullptr,
            0,                                              // flags
            code.size(),                                    // code size
            reinterpret_cast<const uint32_t *>(code.data()) // code data
        };

        if (vkCreateShaderModule(m_device, &shaderModuleInfo, nullptr, &shaderModule) != VK_SUCCESS)
            throw std::runtime_error("failed to create compute shader module!");

        return shaderModule;
    }

    void createComputePipeline()
    {
        VkShaderModule computeShaderModule = createShaderModule("shaders/test.spv");

        // create compute pipeline
        VkPipelineShaderStageCreateInfo computeShaderStageInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_COMPUTE_BIT,
            computeShaderModule,
            "main",
            nullptr // specialization info
        };

        // create descriptor sets

        VkDescriptorSetLayoutBinding descriptorSetBinding{
            0,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            1,
            VK_SHADER_STAGE_ALL,
            nullptr
        };

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &descriptorSetBinding
        };

        if (vkCreateDescriptorSetLayout(m_device, &descriptorSetLayoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create descriptor set layout!");

        VkPipelineLayoutCreateInfo computePipelineLayoutInfo{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,                           // set layout count
            &computeDescriptorSetLayout, // descriptor set layouts
            0,                           // push constant range
            nullptr                      // push constants
        };

        if (vkCreatePipelineLayout(m_device, &computePipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create pipeline layout!");

        VkComputePipelineCreateInfo computePipelineInfo{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            nullptr,
            0,
            computeShaderStageInfo,
            computePipelineLayout,
            VK_NULL_HANDLE,
            0
        };

        std::vector<VkComputePipelineCreateInfo> computePipelineInfos{ computePipelineInfo };

        // std::clog << "number of compute pipeline create infos: " << computePipelineInfos.size() << std::endl;

        computePipelines.resize(computePipelineInfos.size());

        if (vkCreateComputePipelines(m_device, nullptr, computePipelineInfos.size(), computePipelineInfos.data(), nullptr, computePipelines.data()) != VK_SUCCESS)
            throw std::runtime_error("failed to create compute pipeline!");

        vkDestroyShaderModule(m_device, computeShaderModule, nullptr);
    }

    bool createBlockOfMemory(uint32_t memoryType, VkDeviceSize size, VkDeviceMemory &memoryBlock)
    {
        bool success = false;

        VkMemoryAllocateInfo allocateInfo{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            size,
            memoryType
        };

        if (vkAllocateMemory(m_device, &allocateInfo, nullptr, &memoryBlock) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate memory!");

        success = true;

        return success;
    }

    void createManualFramebuffer()
    {
        // VkFormatProperties formatProperties;
        // vkGetPhysicalDeviceFormatProperties(m_physicalDevice, m_swapChainFormat, &formatProperties);
        // std::clog << "\t format properties: " << std::bitset<32>(formatProperties.bufferFeatures) << std::endl;

        QueueFamilyIndices queueFamilies = findQueueFamilies(m_physicalDevice);
        std::vector<uint32_t> queueFamilyIndices = { queueFamilies.graphicsFamily.value() };

        VkDeviceSize frameBufferSize = m_swapChainExtent.width * m_swapChainExtent.height * 8 * 4;

        // create buffer
        VkBufferCreateInfo bufferInfo{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            frameBufferSize,
            VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            static_cast<uint32_t>(queueFamilyIndices.size()),
            queueFamilyIndices.data()
        };

        if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_frameBuffer) != VK_SUCCESS)
            throw std::runtime_error("failed to create buffer!");

        // allocate memory
        // viewMemoryTypes();
        VkMemoryRequirements bufferMemoryRequirements{};
        vkGetBufferMemoryRequirements(m_device, m_frameBuffer, &bufferMemoryRequirements);

        VkMemoryPropertyFlags requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        std::optional<uint32_t> memoryTypeIndex = getMemoryTypeIndex(bufferMemoryRequirements, requiredFlags);

        if (!memoryTypeIndex.has_value())
            throw std::runtime_error("no suitable memory type found!");

        if (!createBlockOfMemory(memoryTypeIndex.value(), frameBufferSize, m_memoryBlock))
            throw std::runtime_error("failed to create block of memory!");

        // bind buffer
        if (vkBindBufferMemory(m_device, m_frameBuffer, m_memoryBlock, 0) != VK_SUCCESS)
            throw std::runtime_error("failed to bind buffer memory!");

        // write data into buffer from host

        // void *data;
        // vkMapMemory(m_device, bufferMemory, 0, bufferSize, 0, &data);

        // create buffer view
        VkBufferViewCreateInfo bufferViewInfo{
            VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            nullptr,
            0,
            m_frameBuffer,
            m_swapChainFormat,
            0,
            frameBufferSize
        };

        if (vkCreateBufferView(m_device, &bufferViewInfo, nullptr, &m_frameBufferView) != VK_SUCCESS)
            throw std::runtime_error("failed to create buffer view!");
    }

    void createImageViews()
    {
        m_swapChainImageViews.resize(m_swapChainImages.size());
        for (uint32_t i = 0; i < m_swapChainImages.size(); i++)
        {
            VkImageViewCreateInfo imageViewInfo{
                VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                nullptr,
                0,
                m_swapChainImages[i],
                VK_IMAGE_VIEW_TYPE_2D,
                m_swapChainFormat,
                VkComponentMapping{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
                VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
            };

            if (vkCreateImageView(m_device, &imageViewInfo, nullptr, &m_swapChainImageViews[i]) != VK_SUCCESS)
                throw std::runtime_error("failed to create image views!");
        }
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupportDetails(m_physicalDevice);

        VkSurfaceFormatKHR format = chooseSwapChainFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = choosePresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        VkSwapchainCreateInfoKHR swapChainInfo{
            VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            nullptr,
            0,
            m_surface,
            imageCount,
            format.format,
            format.colorSpace,
            extent,
            1,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            0,       // queue family count - changes depending on whether the graphics family is the same as the present family index
            nullptr, // same as above
            swapChainSupport.capabilities.currentTransform,
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode,
            VK_FALSE, // real-time rendering for video games should be set to VK_TRUE for performance, but if we're doing offline rendering then VK_FALSE
            nullptr
        };

        if (indices.graphicsFamily.value() != indices.presentFamily.value())
        {
            swapChainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            swapChainInfo.queueFamilyIndexCount = 2;
            swapChainInfo.pQueueFamilyIndices = queueFamilyIndices;
        }

        if (vkCreateSwapchainKHR(m_device, &swapChainInfo, nullptr, &m_swapChain) != VK_SUCCESS)
            throw std::runtime_error("failed to create swap chain!");

        vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, nullptr);
        m_swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, m_swapChainImages.data());

        m_swapChainFormat = format.format;
        m_swapChainExtent = extent;
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
            drawFrame();
        }
    }

    void drawFrame()
    {
    }

    void cleanup()
    {
        vkDeviceWaitIdle(m_device);
        vkDestroyDescriptorSetLayout(m_device, computeDescriptorSetLayout, nullptr);
        vkDestroyCommandPool(m_device, commandPool, nullptr);

        // pipeline objects
        for (auto pipeline : computePipelines)
            vkDestroyPipeline(m_device, pipeline, nullptr);
        vkDestroyPipelineLayout(m_device, computePipelineLayout, nullptr);

        // memory objects
        vkDestroyBufferView(m_device, m_frameBufferView, nullptr);
        vkFreeMemory(m_device, m_memoryBlock, nullptr);
        vkDestroyBuffer(m_device, m_frameBuffer, nullptr);

        for (VkImageView &imageView : m_swapChainImageViews)
            vkDestroyImageView(m_device, imageView, nullptr);

        vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);
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
            // std::clog << i << ":" << std::bitset<32>(queueFamilyProperties[i].queueFlags) << std::endl;
            // check for graphics queue family
            if (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                indices.graphicsFamily = i;

            if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                indices.computeFamily = i;

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
        // std::clog << "(" << details.capabilities.currentExtent.width << ", " << details.capabilities.currentExtent.height << ")" << std::endl;

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

    VkSurfaceFormatKHR chooseSwapChainFormat(std::vector<VkSurfaceFormatKHR> &availableFormats)
    {

        for (const auto &format : availableFormats)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return format;
        }

        return availableFormats[0];
    }

    VkPresentModeKHR choosePresentMode(std::vector<VkPresentModeKHR> &availablePresentModes)
    {
        for (const auto &presentMode : availablePresentModes)
        {
            if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
                return presentMode;
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;
        else
        {
            int width, height;
            glfwGetFramebufferSize(m_window, &width, &height);

            VkExtent2D actualExtent{
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    void viewMemoryTypes()
    {
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);

        std::clog << "memory types:\n";
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
        {
            std::clog << "\t heap index: " << memoryProperties.memoryTypes[i].heapIndex << std::endl;
            std::clog << "\t flag bits: " << std::bitset<32>(memoryProperties.memoryTypes[i].propertyFlags) << std::endl;
        }

        std::clog << "memory heaps:\n";
        for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++)
        {
            std::clog << "\t memory heap size: " << memoryProperties.memoryHeaps[i].size << std::endl;
            std::clog << "\t memory heap flag bits: " << std::bitset<32>(memoryProperties.memoryHeaps[i].flags) << std::endl;
        }
    }

    std::optional<uint32_t> getMemoryTypeIndex(const VkMemoryRequirements &memoryRequirements, const VkMemoryPropertyFlags &requiredFlags)
    {
        std::optional<uint32_t> selectedMemoryType;

        // std::clog << "memory requirements:\n";
        // std::clog << "\t suitable memory type index bits: " << std::bitset<32>(memoryRequirements.memoryTypeBits) << std::endl;
        // std::clog << "\t memory alignment: " << memoryRequirements.alignment << std::endl;
        // std::clog << "\t memory size: " << memoryRequirements.size << std::endl;

        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);

        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
        {
            bool currentMemoryTypeSupported = memoryRequirements.memoryTypeBits & (1 << i);
            bool allRequiredFlagsAvailable = (memoryProperties.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags;
            // std::clog << "suitable memory bits: \t" << std::bitset<32>(memoryRequirements.memoryTypeBits) << std::endl;
            // std::clog << "current bit: \t\t" << std::bitset<32>(1 << i) << std::endl;
            // std::clog << "memory property flags: \t" << std::bitset<32>(memoryProperties.memoryTypes[i].propertyFlags) << std::endl;
            // std::clog << "required flags: \t" << std::bitset<32>(requiredFlags) << std::endl;
            // std::clog << "memory heap flags: \t" << std::bitset<32>(memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].flags) << std::endl;
            // std::clog << "memory heap index: \t" << (memoryProperties.memoryTypes[i].heapIndex) << std::endl;
            // std::clog << "memory heap size: \t" << (memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size) << std::endl;
            // std::clog << std::endl;
            if (allRequiredFlagsAvailable && currentMemoryTypeSupported)
            {
                selectedMemoryType = i;
                break;
            }
        }

        return selectedMemoryType;
    }

    static std::vector<char> readFile(const std::string &fileName)
    {
        std::ifstream file(fileName, std::ios::ate | std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("failed to open file!");

        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }
};

int main()
{
    std::unique_ptr<Game> game = std::make_unique<Game>(800, 600);
    game->run();
    return EXIT_SUCCESS;
}