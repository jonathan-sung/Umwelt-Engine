#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstring>
#include <fstream>
#include <glm/detail/qualifier.hpp>
#include <glm/ext/vector_float2.hpp>
#include <limits>
#include <optional>
#include <string>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <glm/glm.hpp>
#include <iostream>
#include <memory>
#include <ostream>
#include <set>
#include <stdexcept>
#include <sys/types.h>
#include <vector>
#include <vulkan/vk_enum_string_helper.h>

#ifdef NDEBUG
bool validationLayersEnabled = false;
#else
bool validationLayersEnabled = true;
#endif

#define VK_CHECK(x, userMessage)                                                                             \
    do                                                                                                       \
    {                                                                                                        \
        VkResult err = x;                                                                                    \
        if (err)                                                                                             \
        {                                                                                                    \
            std::string errorMessage = "Detected Vulkan error: " + std::string(string_VkResult(err)) + "\t"; \
            throw std::runtime_error(errorMessage + std::string(userMessage));                               \
        }                                                                                                    \
    } while (0)

class Game
{
public:
    Game(const uint32_t width, const float aspectRatio) : m_width(width), m_height(uint32_t(float(width) / aspectRatio))
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

    // General Vulkan objects
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;

    // Graphics Pipeline
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_presentQueue = VK_NULL_HANDLE;

    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_commandBuffers;

    VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_graphicsPipelineLayout = VK_NULL_HANDLE;

    VkDescriptorPool m_graphicsDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_graphicsDescriptorSets;
    std::vector<VkDescriptorSetLayout> m_graphicsDescriptorSetLayouts{};

    VkRenderPass m_renderPass = VK_NULL_HANDLE;

    VkSemaphore imageAvailableSemaphore;
    VkSemaphore presentationReadySemaphore;
    VkFence drawingFinishedFence;
    uint32_t currentImageIndex;

    // // Swapchain
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> m_swapChainImages;
    std::vector<VkImageView> m_swapChainImageViews;
    std::vector<VkFramebuffer> m_framebuffers;
    VkFormat m_swapChainFormat;
    VkExtent2D m_swapChainExtent;

    // // Graphics Resources
    VkDeviceMemory m_vertexBufferMemory;
    VkBuffer m_vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_uniformBufferMemory;
    VkBuffer m_uniformBuffer = VK_NULL_HANDLE;

    VkDeviceMemory m_presentImageMemory = VK_NULL_HANDLE;
    VkImage m_presentImage = VK_NULL_HANDLE;
    VkImageView m_presentImageView = VK_NULL_HANDLE;
    VkSampler m_presentImageSampler = VK_NULL_HANDLE;

    // Compute Pipeline
    VkQueue m_computeQueue = VK_NULL_HANDLE;

    std::vector<VkPipeline> m_computePipelines{};
    VkPipelineLayout m_computePipelineLayout = VK_NULL_HANDLE;

    VkDescriptorPool m_computeDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_computeDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_computeDescriptorSetLayout = VK_NULL_HANDLE;

    VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantData) };
    VkPushConstantRange graphicsPushConstantRange{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantData) };

    // // Resources
    VkDeviceMemory m_computeImageMemory = VK_NULL_HANDLE;
    VkImage m_computeImage = VK_NULL_HANDLE;
    VkImageView m_computeImageView = VK_NULL_HANDLE;

    // General member variables
    uint32_t m_width;
    uint32_t m_height;
    bool enableConsoleOutput = false;
    std::array<uint32_t, 3> workGroupSize;

    // GLFW member variables
    GLFWwindow *m_window;

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

    struct Vertex
    {
        glm::vec2 position;
        glm::vec3 color;

        static VkVertexInputBindingDescription getBindingDescription()
        {
            VkVertexInputBindingDescription bindingDescription{};
            bindingDescription.binding = 0;
            bindingDescription.stride = sizeof(Vertex);
            bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            return bindingDescription;
        }

        static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
        {
            std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
            attributeDescriptions[0].binding = 0;
            attributeDescriptions[0].location = 0;
            attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
            attributeDescriptions[0].offset = offsetof(Vertex, position);
            attributeDescriptions[1].binding = 0;
            attributeDescriptions[1].location = 1;
            attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[1].offset = offsetof(Vertex, color);
            return attributeDescriptions;
        }
    };

    struct PushConstantData
    {
        VkExtent2D extent;
        float time;
    };

    const std::vector<const char *> validationLayers{
        "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<const char *> deviceExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    static void cursorPositionCallback(GLFWwindow *window, double xPos, double yPos)
    {
        // std::clog << "Cursor Position: (" << xPos << ", " << yPos << ")\n";
    }

    void initWindow()
    {
        if (glfwInit() == GLFW_FALSE)
            throw std::runtime_error("GLFW initlization failed!");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        m_window = glfwCreateWindow(m_width, m_height, "Umwelt Engine Demo", nullptr, nullptr);

        if (m_window == nullptr)
            throw std::runtime_error("failed to create GLFW window!");

        if (glfwVulkanSupported() == GLFW_FALSE)
            throw std::runtime_error("GLFW Vulkan not supported!");

        if (glfwRawMouseMotionSupported() == GLFW_FALSE)
            throw std::runtime_error("GLFW raw mouse motion not supported!");
        else
        {
            glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            glfwSetInputMode(m_window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
        }
        glfwSetCursorPosCallback(m_window, cursorPositionCallback);
    }

    void initVulkan()
    {
        createInstance();
        createSurface();
        choosePhysicalDevice();
        createLogicalDevice();

        createSwapChain();
        createImageViews();
        createSyncObjects();
        createRenderPass();
        createFramebuffers();

        // Compute
        setWorkGroupSize();
        createComputeResources();
        createComputeDescriptorSets();
        createComputePipeline();

        // Graphics
        createGraphicsResources();
        createGraphicsDescriptorSets();
        createGraphicsPipeline();

        createCommandPool();
    }

    void cleanupDescriptorSet(VkDevice device, VkDescriptorPool &descriptorPool, VkDescriptorSetLayout &descriptorSetLayout, uint32_t descriptorSetCount, VkDescriptorSet *descriptorSets)
    {
        vkFreeDescriptorSets(device, descriptorPool, descriptorSetCount, descriptorSets);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    void createComputeDescriptorSets()
    {
        VkDescriptorSetLayoutBinding bindings{
            0,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr
        };

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &bindings
        };
        VK_CHECK(vkCreateDescriptorSetLayout(m_device, &descriptorSetLayoutInfo, nullptr, &m_computeDescriptorSetLayout), "create descriptor set layout");

        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 };

        VkDescriptorPoolCreateInfo descriptorPoolInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            nullptr,
            0,
            1,
            1,
            &poolSize
        };

        VK_CHECK(vkCreateDescriptorPool(m_device, &descriptorPoolInfo, nullptr, &m_computeDescriptorPool), "create descriptor pool");

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            m_computeDescriptorPool,
            1,
            &m_computeDescriptorSetLayout
        };
        VK_CHECK(vkAllocateDescriptorSets(m_device, &descriptorSetAllocInfo, &m_computeDescriptorSet), "allocate descriptor sets");

        VkDescriptorImageInfo imageInfo{
            nullptr,
            m_computeImageView,
            VK_IMAGE_LAYOUT_GENERAL
        };

        VkWriteDescriptorSet descriptorWrites{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            m_computeDescriptorSet,
            0,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            &imageInfo,
            nullptr,
            nullptr
        };

        vkUpdateDescriptorSets(m_device, 1, &descriptorWrites, 0, nullptr);
    }

    void createGraphicsDescriptorSets()
    {
        VkDescriptorSetLayoutBinding uniformBufferDescriptorSetBinding{
            0,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            1,
            VK_SHADER_STAGE_ALL,
            nullptr
        };

        VkDescriptorSetLayoutBinding presentImageDescriptorSetBinding{
            1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            1,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            nullptr
        };

        std::vector<VkDescriptorSetLayoutBinding> graphicsDescriptorSetBindings = { uniformBufferDescriptorSetBinding, presentImageDescriptorSetBinding };

        VkDescriptorSetLayoutCreateInfo graphicsDescriptorSetLayoutInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(graphicsDescriptorSetBindings.size()), // 2; uniform buffer and input present image
            graphicsDescriptorSetBindings.data()
        };

        m_graphicsDescriptorSetLayouts.resize(1);
        VK_CHECK(vkCreateDescriptorSetLayout(m_device, &graphicsDescriptorSetLayoutInfo, nullptr, &m_graphicsDescriptorSetLayouts[0]), "create descriptor set layout");

        VkDescriptorPoolSize poolSizes[] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };

        VkDescriptorPoolCreateInfo graphicsDescriptorPoolInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            nullptr,
            0,
            1,
            2,
            poolSizes
        };
        VK_CHECK(vkCreateDescriptorPool(m_device, &graphicsDescriptorPoolInfo, nullptr, &m_graphicsDescriptorPool), "create graphics descriptor pool");

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            m_graphicsDescriptorPool,
            1,
            m_graphicsDescriptorSetLayouts.data()
        };

        m_graphicsDescriptorSets.resize(m_graphicsDescriptorSetLayouts.size());
        VK_CHECK(vkAllocateDescriptorSets(m_device, &descriptorSetAllocInfo, m_graphicsDescriptorSets.data()), "allocate graphics descriptor sets");

        // RESUME - COME BACK TO HERE AFTER CREATE THE VERTEX BUFFER
        VkDescriptorBufferInfo uniformBufferInfo{
            m_uniformBuffer,
            0,
            VK_WHOLE_SIZE
        };

        VkDescriptorImageInfo presentImageInfo{
            m_presentImageSampler,
            m_presentImageView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        };

        std::vector<VkWriteDescriptorSet> graphicsWriteDescriptorSets{ { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                                         nullptr,
                                                                         m_graphicsDescriptorSets[0],
                                                                         0,
                                                                         0,
                                                                         1,
                                                                         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                                         nullptr,
                                                                         &uniformBufferInfo,
                                                                         nullptr },
                                                                       { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                                         nullptr,
                                                                         m_graphicsDescriptorSets[0],
                                                                         1,
                                                                         0,
                                                                         1,
                                                                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                                         &presentImageInfo,
                                                                         nullptr,
                                                                         nullptr } };

        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(graphicsWriteDescriptorSets.size()), graphicsWriteDescriptorSets.data(), 0, nullptr);
    }

    void createUniformBuffer()
    {
        void *data;
        float stuff[] = { 69.0f };

        QueueFamilyIndices queueFamilies = findQueueFamilies(m_physicalDevice);

        VkBufferCreateInfo bufferInfo{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            sizeof(stuff),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            1,
            &queueFamilies.graphicsFamily.value()
        };

        VK_CHECK(vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_uniformBuffer), "create uniform buffer");

        VkMemoryRequirements uniformBufferMemoryRequirements{};
        vkGetBufferMemoryRequirements(m_device, m_uniformBuffer, &uniformBufferMemoryRequirements);

        VkMemoryAllocateInfo uniformBufferAllocInfo{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            uniformBufferMemoryRequirements.size,
            getMemoryTypeIndex(uniformBufferMemoryRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT).value()
        };

        VK_CHECK(vkAllocateMemory(m_device, &uniformBufferAllocInfo, nullptr, &m_uniformBufferMemory), "allocate uniform buffer memory");

        VK_CHECK(vkBindBufferMemory(m_device, m_uniformBuffer, m_uniformBufferMemory, 0), "bind uniform buffer memory");

        VK_CHECK(vkMapMemory(m_device, m_uniformBufferMemory, 0, VK_WHOLE_SIZE, 0, &data), "map uniform buffer data");
        std::memcpy(data, stuff, sizeof(stuff));
        vkUnmapMemory(m_device, m_uniformBufferMemory);
    }

    void createVertexBuffer()
    {
        void *data;
        Vertex vertices[] = { { { -1.0f, 1.0f }, { 1.0f, 0.0f, 0.0f } }, { { 1.0f, 1.0f }, { 0.0f, 1.0f, 0.0f } }, { { 1.0f, -1.0f }, { 0.0f, 0.0f, 1.0f } }, { { -1.0f, -1.0f }, { 1.0f, 0.0f, 1.0f } } };

        QueueFamilyIndices queueFamilies = findQueueFamilies(m_physicalDevice);

        VkBufferCreateInfo bufferInfo{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            sizeof(vertices),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            1,
            &queueFamilies.graphicsFamily.value()
        };

        VK_CHECK(vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_vertexBuffer), "create vertex buffer");

        VkMemoryRequirements vertexBufferMemoryRequirements{};
        vkGetBufferMemoryRequirements(m_device, m_vertexBuffer, &vertexBufferMemoryRequirements);

        VkMemoryAllocateInfo vertexBufferAllocInfo{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            vertexBufferMemoryRequirements.size,
            getMemoryTypeIndex(vertexBufferMemoryRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT).value()
        };

        VK_CHECK(vkAllocateMemory(m_device, &vertexBufferAllocInfo, nullptr, &m_vertexBufferMemory), "allocate vertex buffer memory");

        VK_CHECK(vkBindBufferMemory(m_device, m_vertexBuffer, m_vertexBufferMemory, 0), "bind vertex buffer memory");

        VK_CHECK(vkMapMemory(m_device, m_vertexBufferMemory, 0, VK_WHOLE_SIZE, 0, &data), "map vertex buffer memory");
        std::memcpy(data, vertices, sizeof(vertices));
        vkUnmapMemory(m_device, m_vertexBufferMemory);
    }

    void createImageSampler()
    {
        VkSamplerCreateInfo samplerInfo{
            VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            nullptr,
            0,
            VK_FILTER_NEAREST, // change this to linear if it looks blocky
            VK_FILTER_NEAREST,
            VK_SAMPLER_MIPMAP_MODE_NEAREST,
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            0.0f,
            VK_FALSE,
            1.0f,
            VK_FALSE,
            VK_COMPARE_OP_ALWAYS,
            0.0f,
            0.0f,
            VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
            VK_FALSE
        };
        VK_CHECK(vkCreateSampler(m_device, &samplerInfo, nullptr, &m_presentImageSampler), "create present image sampler");
    }

    void createGraphicsResources()
    {
        createVertexBuffer();
        createUniformBuffer();
        createImageSampler();
        createPresentImage();
    }

    void createPresentImage()
    {
        VkImageCreateInfo imageInfo{
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            nullptr,
            0,
            VK_IMAGE_TYPE_2D,
            VK_FORMAT_B8G8R8A8_UNORM,
            VkExtent3D{ m_width, m_height, 1 },
            1,
            1,
            VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            0,
            nullptr,
            VK_IMAGE_LAYOUT_UNDEFINED
        };

        VK_CHECK(vkCreateImage(m_device, &imageInfo, nullptr, &m_presentImage), "create present image");

        VkMemoryRequirements imageMemoryRequirements{};

        vkGetImageMemoryRequirements(m_device, m_presentImage, &imageMemoryRequirements);

        // std::clog << "REQUIRED SIZE: " << imageMemoryRequirements.size << std::endl;

        VkMemoryAllocateInfo memoryAllocInfo{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            imageMemoryRequirements.size,
            getMemoryTypeIndex(imageMemoryRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).value()
        };

        VK_CHECK(vkAllocateMemory(m_device, &memoryAllocInfo, nullptr, &m_presentImageMemory), "allocate present image memory");

        VK_CHECK(vkBindImageMemory(m_device, m_presentImage, m_presentImageMemory, 0), "bind present image memory");

        VkImageViewCreateInfo imageViewInfo{
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            nullptr,
            0,
            m_presentImage,
            VK_IMAGE_VIEW_TYPE_2D,
            VK_FORMAT_B8G8R8A8_UNORM,
            VkComponentMapping{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
            VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        };

        VK_CHECK(vkCreateImageView(m_device, &imageViewInfo, nullptr, &m_presentImageView), "create present image view");
    }

    void createSyncObjects()
    {
        VkSemaphoreCreateInfo imageAvailableSemaphoreInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0 };
        VkSemaphoreCreateInfo presentationReadySemaphoreInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0 };
        VK_CHECK(vkCreateSemaphore(m_device, &imageAvailableSemaphoreInfo, nullptr, &imageAvailableSemaphore), "create image available semaphore");
        VK_CHECK(vkCreateSemaphore(m_device, &presentationReadySemaphoreInfo, nullptr, &presentationReadySemaphore), "create present image semaphore");

        VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0 };
        VK_CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &drawingFinishedFence), "create drawing finished fence");
    }
    void createFramebuffers()
    {
        m_framebuffers.resize(m_swapChainImageViews.size());
        for (auto i = 0; i < m_swapChainImageViews.size(); i++)
        {
            VkImageView attachments[] = { m_swapChainImageViews[i] };

            VkFramebufferCreateInfo framebufferInfo{
                VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                nullptr,
                0,
                m_renderPass,
                1,
                attachments,
                m_swapChainExtent.width,
                m_swapChainExtent.height,
                1
            };

            VK_CHECK(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_framebuffers[i]), "create framebuffers");
        }
    }

    void createRenderPass()
    {
        std::vector<VkAttachmentDescription> attachments{ { 0,
                                                            VK_FORMAT_B8G8R8A8_UNORM,
                                                            VK_SAMPLE_COUNT_1_BIT,
                                                            VK_ATTACHMENT_LOAD_OP_CLEAR,
                                                            VK_ATTACHMENT_STORE_OP_STORE,
                                                            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                                            VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                                            VK_IMAGE_LAYOUT_UNDEFINED,
                                                            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR } };

        std::vector<VkAttachmentReference> attachmentReferences{ { 0,
                                                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL } };
        std::vector<VkSubpassDescription> subpasses{ { 0,
                                                       VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                       0,
                                                       nullptr,
                                                       1,
                                                       attachmentReferences.data(),
                                                       nullptr,
                                                       nullptr,
                                                       0,
                                                       nullptr } };
        VkRenderPassCreateInfo renderPassInfo{
            VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            nullptr,
            0,
            1,
            attachments.data(),
            1,
            subpasses.data(),
            0,
            nullptr
        };

        VK_CHECK(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass), "create render pass");
    }

    void createGraphicsPipeline()
    {
        VkShaderModuleCreateInfo vertexShaderInfo{};
        VkShaderModule vertexModule = createShaderModule("shaders/vertex.spv");
        VkShaderModuleCreateInfo fragmentShaderInfo{};
        VkShaderModule fragmentModule = createShaderModule("shaders/fragment.spv");

        VkPipelineShaderStageCreateInfo vertexStageInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_VERTEX_BIT,
            vertexModule,
            "main",
            nullptr
        };

        VkPipelineShaderStageCreateInfo fragmentStageInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            fragmentModule,
            "main",
            nullptr
        };

        std::vector<VkPipelineShaderStageCreateInfo> shaderStages{ vertexStageInfo, fragmentStageInfo };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescription = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            nullptr,
            0,
            1,
            &bindingDescription,
            static_cast<uint32_t>(attributeDescription.size()),
            attributeDescription.data()
        };

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
            VK_FALSE
        };
        VkViewport viewport{ 0.0f, 0.0f, static_cast<float>(m_swapChainExtent.width), static_cast<float>(m_swapChainExtent.height), 0.0f, 1.0f };
        VkRect2D scissor{ { 0, 0 }, m_swapChainExtent };

        VkPipelineViewportStateCreateInfo viewportStateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            nullptr,
            0,
            1,
            nullptr, // &viewport,
            1,
            nullptr, //&scissor
        };

        VkPipelineRasterizationStateCreateInfo rasterization{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_FALSE,
            VK_FALSE,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_BACK_BIT,
            VK_FRONT_FACE_COUNTER_CLOCKWISE,
            VK_FALSE,
            0.0f,
            0.0f,
            0.0f,
            1.0f
        };

        VkPipelineLayoutCreateInfo graphicsPipelineLayoutInfo{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(m_graphicsDescriptorSetLayouts.size()),
            m_graphicsDescriptorSetLayouts.data(),
            1,
            &graphicsPushConstantRange
        };

        VK_CHECK(vkCreatePipelineLayout(m_device, &graphicsPipelineLayoutInfo, nullptr, &m_graphicsPipelineLayout), "create graphics pipeline layout");

        VkPipelineColorBlendAttachmentState colorAttachments{
            VK_FALSE,
            VK_BLEND_FACTOR_SRC_COLOR,
            VK_BLEND_FACTOR_SRC_COLOR,
            VK_BLEND_OP_ADD,
            VK_BLEND_FACTOR_SRC_ALPHA,
            VK_BLEND_FACTOR_SRC_ALPHA,
            VK_BLEND_OP_ADD,
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
        };

        VkPipelineColorBlendStateCreateInfo colorBlendInfo{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_FALSE,
            VK_LOGIC_OP_COPY,
            1,
            &colorAttachments
        };

        std::vector<VkDynamicState> dynamicStates{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

        VkPipelineDynamicStateCreateInfo dynamicStateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(dynamicStates.size()),
            dynamicStates.data()
        };

        VkGraphicsPipelineCreateInfo graphicsPipelineInfo{
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(shaderStages.size()),
            shaderStages.data(),
            &vertexInputStateCreateInfo,
            &inputAssemblyStateInfo,
            nullptr,
            &viewportStateInfo,
            &rasterization,
            nullptr,
            nullptr,
            &colorBlendInfo,
            &dynamicStateInfo, // nullptr,
            m_graphicsPipelineLayout,
            m_renderPass,
            0,
            VK_NULL_HANDLE,
            0,
        };
        VK_CHECK(vkCreateGraphicsPipelines(m_device, nullptr, 1, &graphicsPipelineInfo, nullptr, &m_graphicsPipeline), "create graphics pipeline");

        vkDestroyShaderModule(m_device, vertexModule, nullptr);
        vkDestroyShaderModule(m_device, fragmentModule, nullptr);
    }

    void createComputeResources()
    {
        createComputeImage();
    }

    void createComputeImage()
    {
        // Check for suitable formats
        // for (uint32_t i = 0; i < 184; i++)
        // {
        //     VkFormatProperties formatProperties{};
        //     vkGetPhysicalDeviceFormatProperties(m_physicalDevice, (VkFormat)i, &formatProperties);
        //     if (formatProperties.linearTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
        //     {
        //         std::clog << "suitable format: " << i << std::endl;
        //     }
        // }
        VkFormatProperties formatProperties{};
        vkGetPhysicalDeviceFormatProperties(m_physicalDevice, VK_FORMAT_B8G8R8A8_UNORM, &formatProperties);

        // std::clog << "linear tiling features: " << std::bitset<32>(formatProperties.linearTilingFeatures) << std::endl;
        // std::clog << "optimal tiling features: " << std::bitset<32>(formatProperties.optimalTilingFeatures) << std::endl;

        VkImageFormatProperties imageFormatProperties{};
        vkGetPhysicalDeviceImageFormatProperties(m_physicalDevice, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_STORAGE_BIT, 0, &imageFormatProperties);

        // std::clog << "image max array layers: " << imageFormatProperties.maxArrayLayers << std::endl;
        // std::clog << "image max mip levels: " << imageFormatProperties.maxMipLevels << std::endl;
        // std::clog << "image supported sample count bits: " << std::bitset<32>(imageFormatProperties.sampleCounts) << std::endl;
        // std::clog << "image max extent: (" << imageFormatProperties.maxExtent.width << ", " << imageFormatProperties.maxExtent.height << ", " << imageFormatProperties.maxExtent.depth << ")" << std::endl;

        VkImageCreateInfo imageInfo{
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            nullptr,
            0,
            VK_IMAGE_TYPE_2D,
            VK_FORMAT_B8G8R8A8_UNORM,
            VkExtent3D{ m_width, m_height, 1 },
            1,
            1,
            VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_TILING_OPTIMAL, // CHANGE THIS BACK TO OPTIMAL IF IT WORKS
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_SHARING_MODE_EXCLUSIVE, // change this if the graphics queue and compute queue are different
            0,
            nullptr,
            VK_IMAGE_LAYOUT_UNDEFINED
        };

        VK_CHECK(vkCreateImage(m_device, &imageInfo, nullptr, &m_computeImage), "create compute image");

        VkMemoryRequirements imageMemoryRequirements{};

        vkGetImageMemoryRequirements(m_device, m_computeImage, &imageMemoryRequirements);

        // std::clog << "REQUIRED SIZE: " << imageMemoryRequirements.size << std::endl;

        VkMemoryAllocateInfo memoryAllocInfo{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            imageMemoryRequirements.size,
            getMemoryTypeIndex(imageMemoryRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).value()
        };

        VK_CHECK(vkAllocateMemory(m_device, &memoryAllocInfo, nullptr, &m_computeImageMemory), "allocate compute image memory");

        VK_CHECK(vkBindImageMemory(m_device, m_computeImage, m_computeImageMemory, 0), "bind compute image memory");

        VkImageViewCreateInfo imageViewInfo{
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            nullptr,
            0,
            m_computeImage,
            VK_IMAGE_VIEW_TYPE_2D,
            VK_FORMAT_B8G8R8A8_UNORM,
            VkComponentMapping{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
            VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        };

        VK_CHECK(vkCreateImageView(m_device, &imageViewInfo, nullptr, &m_computeImageView), "create compute image view");
    }

    void createCommandPool()
    {
        QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);

        VkCommandPoolCreateInfo commandPoolInfo{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            0,
            indices.graphicsFamily.value()
        };

        if (vkCreateCommandPool(m_device, &commandPoolInfo, nullptr, &m_commandPool) != VK_SUCCESS)
            throw std::runtime_error("failed to create command pool!");

        VkCommandBufferAllocateInfo allocInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            m_commandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            1
        };

        m_commandBuffers.resize(allocInfo.commandBufferCount);

        vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data());
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
        VkShaderModule computeModule = createShaderModule("shaders/compute.spv");

        std::array<VkSpecializationMapEntry, 2> mapEntries;
        mapEntries[0] = { 0, 0, sizeof(uint32_t) };
        mapEntries[1] = { 1, sizeof(uint32_t) * 1, sizeof(uint32_t) };

        void *specializationData = reinterpret_cast<void *>(workGroupSize.data());

        VkSpecializationInfo specializationInfo{
            2,
            mapEntries.data(),
            sizeof(uint32_t) * mapEntries.size(),
            &specializationData
        };

        VkPipelineShaderStageCreateInfo stageInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_COMPUTE_BIT,
            computeModule,
            "main",
            &specializationInfo
        };

        VkPipelineLayoutCreateInfo computePipelineLayoutInfo{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &m_computeDescriptorSetLayout,
            1,
            &computePushConstantRange
        };

        VK_CHECK(vkCreatePipelineLayout(m_device, &computePipelineLayoutInfo, nullptr, &m_computePipelineLayout), "create compute pipeline layout");

        VkComputePipelineCreateInfo computePipelineInfo{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            nullptr,
            0,
            stageInfo,
            m_computePipelineLayout,
            VK_NULL_HANDLE,
            -1
        };

        m_computePipelines.resize(1);
        VK_CHECK(vkCreateComputePipelines(m_device, nullptr, 1, &computePipelineInfo, nullptr, m_computePipelines.data()), "create compute pipeline");

        vkDestroyShaderModule(m_device, computeModule, nullptr);
    }

    bool createBlockOfMemory(uint32_t memoryType, VkDeviceSize size, VkDeviceMemory &memoryBlock)
    {

        VkMemoryAllocateInfo allocateInfo{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            size,
            memoryType
        };

        VK_CHECK(vkAllocateMemory(m_device, &allocateInfo, nullptr, &memoryBlock), "allocate block of memory");

        return true;
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
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, // not rendering into directly using graphics pipeline; using compute shaders; directly into the swap chain images using the compute shaders
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

        // std::clog << "number of images in swap chain: " << imageCount << std::endl;

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
        vkGetDeviceQueue(m_device, indices.computeFamily.value(), 0, &m_computeQueue);
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

    void setWorkGroupSize()
    {
        // Set work group size
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(m_physicalDevice, &properties);
        uint32_t size = 1;
        do
        {
            size = size << 1;

        } while (size * size <= properties.limits.maxComputeWorkGroupInvocations);

        size = size >> 1; // choose the largest size supported by the device
        size = 16; // hard-coding it on 16 just to be safe; comment this out later
        workGroupSize[0] = size;
        workGroupSize[1] = size;
        workGroupSize[2] = 1;

        // std::clog << size * size << std::endl;
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
            // TEST_updateUniformBuffer();

            acquireImage();
            drawFrame();
            present();
        }
    }
    void TEST_updateUniformBuffer()
    {
        void *data;
        float stuff[] = { (float)glfwGetTime() };
        VK_CHECK(vkMapMemory(m_device, m_uniformBufferMemory, 0, VK_WHOLE_SIZE, 0, &data), "map uniform buffer memory");
        std::memcpy(data, stuff, sizeof(stuff));
        vkUnmapMemory(m_device, m_uniformBufferMemory);

        // update vertex buffers

        Vertex vertices[] = { { { -1.0f, 1.0f }, { 1.0f, 0.0f, 0.0f } }, { { 1.0f, 1.0f }, { 0.0f, 1.0f, 0.0f } }, { { 1.0f, -1.0f }, { 0.0f, 0.0f, 1.0f } }, { { -1.0f, -1.0f }, { 1.0f, 0.0f, 1.0f } } };
        VK_CHECK(vkMapMemory(m_device, m_vertexBufferMemory, 0, VK_WHOLE_SIZE, 0, &data), "map vertex buffer memory");
        std::memcpy(data, vertices, sizeof(vertices));
        vkUnmapMemory(m_device, m_vertexBufferMemory);
    }

    void acquireImage()
    {
        VK_CHECK(vkAcquireNextImageKHR(m_device, m_swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &currentImageIndex), "acquire next image");
        // std::clog << "swap chain image index: " << currentImageIndex << std::endl;
    }

    void drawFrame()
    {
        /*
        - draw stuff into computeImage using compute shader
            - move image from undefined to general layout, and from none to shader write
            - write data into computeImage (temporary - cmdfillimage)
                - set up cmd dispatch workgroup sizes and shizz
                - write the GLSL shader code which actually does this stuff
            - computeImage: move from general to transfer src layout, and from shader write to transfer read
            - presentImage: move from undefined to transfer dst layout, and from none to transfer write // BUT WHAT ABOUT THE PREVIOUS FRAME? we're assuming that the images will start with undefined layout
            - transfer computeImage into presentImage
        - present image to screen
            - presentImage: move from transfer dst layout to present layout
        - reset everything back to normal
            - set computeImage to undefined
            - set presentImage to undefined
            - reset command buffer individual using that special flag at creation

        */
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0, nullptr };
        VkImageSubresourceRange subresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkClearValue clearValue = { 1.0f, 1.0f, 1.0f, 1.0f };
        VkClearColorValue redClearColorValue = { 1.0f, 0.0f, 0.0f, 1.0f };
        VkDeviceSize offsets[] = { 0 };

        VkImageMemoryBarrier computeImageBarrier{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_NONE,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            m_computeImage,
            subresourceRange
        };

        VkImageMemoryBarrier transferComputeImageToPresentImageComputeBarrier{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            m_computeImage,
            subresourceRange
        };

        VkImageMemoryBarrier transferComputeImageToPresentImagePresentBarrier{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_NONE,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            m_presentImage,
            subresourceRange
        };

        VkImageMemoryBarrier presentImageToShaderRead{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            m_presentImage,
            subresourceRange
        };

        VkRenderPassBeginInfo renderPassBeginInfo{
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            nullptr,
            m_renderPass,
            m_framebuffers[currentImageIndex],
            { { 0, 0 }, { m_swapChainExtent.width, m_swapChainExtent.height } },
            1,
            &clearValue
        };

        VkImageSubresourceLayers subresourceLayers{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        VkOffset3D offset3D{ 0, 0, 0 };
        VkExtent3D extent{};

        VkImageCopy imageCopyInfo{
            subresourceLayers,
            offset3D,
            subresourceLayers,
            offset3D,
            VkExtent3D{ m_width, m_height, 1 }
        };

        // Commands TODO
        // 1. compute image
        // 2. transfer image from Compute VkImage to Present VkImage
        // 3. Sample the Present VkImage as a texture onto the full screen quad
        VK_CHECK(vkBeginCommandBuffer(m_commandBuffers[0], &beginInfo), "begin command buffer");

        // update push constants

        PushConstantData pushConstantData{ m_swapChainExtent, static_cast<float>(glfwGetTime()) };

        vkCmdPushConstants(m_commandBuffers[0], m_computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantData), reinterpret_cast<void *>(&pushConstantData));
        vkCmdPushConstants(m_commandBuffers[0], m_graphicsPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantData), reinterpret_cast<void *>(&pushConstantData));

        // set up compute pipeline
        vkCmdBindPipeline(m_commandBuffers[0], VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelines[0]);
        vkCmdBindDescriptorSets(m_commandBuffers[0], VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineLayout, 0, 1, &m_computeDescriptorSet, 0, nullptr);

        // execute compute work
        vkCmdPipelineBarrier(m_commandBuffers[0], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &computeImageBarrier);
        vkCmdDispatch(m_commandBuffers[0], m_width / workGroupSize[0], m_height / workGroupSize[1], 1);

        // transfer compute image to present image
        vkCmdPipelineBarrier(m_commandBuffers[0], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &transferComputeImageToPresentImageComputeBarrier);
        vkCmdPipelineBarrier(m_commandBuffers[0], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &transferComputeImageToPresentImagePresentBarrier);
        vkCmdCopyImage(m_commandBuffers[0], m_computeImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_presentImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopyInfo);

        // set up graphics pipeline
        vkCmdPipelineBarrier(m_commandBuffers[0], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &presentImageToShaderRead);
        vkCmdBindPipeline(m_commandBuffers[0], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
        vkCmdBindDescriptorSets(m_commandBuffers[0], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineLayout, 0, m_graphicsDescriptorSets.size(), m_graphicsDescriptorSets.data(), 0, nullptr);

        // update dynamic states of graphics pipeline
        VkViewport viewport{ 0.0f, 0.0f, static_cast<float>(m_swapChainExtent.width), static_cast<float>(m_swapChainExtent.height), 0.0f, 1.0f };
        VkRect2D scissor{ { 0, 0 }, m_swapChainExtent };
        vkCmdSetViewport(m_commandBuffers[0], 0, 1, &viewport);
        vkCmdSetScissor(m_commandBuffers[0], 0, 1, &scissor);

        // render pass
        vkCmdBeginRenderPass(m_commandBuffers[0], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindVertexBuffers(m_commandBuffers[0], 0, 1, &m_vertexBuffer, offsets);
        vkCmdDraw(m_commandBuffers[0], 4, 1, 0, 0);
        vkCmdEndRenderPass(m_commandBuffers[0]);

        VK_CHECK(vkEndCommandBuffer(m_commandBuffers[0]), "end command buffer");

        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

        VkSubmitInfo submitInfo{
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            1,
            &imageAvailableSemaphore,
            waitStages,
            1,
            m_commandBuffers.data(),
            1,
            &presentationReadySemaphore
        };
        VK_CHECK(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, drawingFinishedFence), "queue submit");
    }

    void present()
    {

        VkPresentInfoKHR presentInfo{
            VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            nullptr,
            1,
            &presentationReadySemaphore,
            1,
            &m_swapChain,
            &currentImageIndex,
            nullptr
        };

        VkResult result = vkQueuePresentKHR(m_presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
            recreateSwapChain();
        else
            VK_CHECK(result, "queue present");
        currentImageIndex = (currentImageIndex + 1) % m_framebuffers.size();

        VK_CHECK(vkWaitForFences(m_device, 1, &drawingFinishedFence, VK_TRUE, INT64_MAX), "wait for fences");
        VK_CHECK(vkResetFences(m_device, 1, &drawingFinishedFence), "reset fences");
        VK_CHECK(vkResetCommandPool(m_device, m_commandPool, 0), "reset command pool");
        // VK_CHECK(vkResetDescriptorPool(m_device, m_graphicsDescriptorPool, 0), "reset descriptor pool");
    }

    void destroyDescriptorSets()
    {
    }

    void destroyImage(VkDevice device, VkDeviceMemory memory, VkImageView imageView, VkImage image)
    {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImageView(m_device, imageView, nullptr);
        vkDestroyImage(device, image, nullptr);
    }

    void cleanup()
    {
        vkDeviceWaitIdle(m_device);

        vkDestroySemaphore(m_device, imageAvailableSemaphore, nullptr);
        vkDestroySemaphore(m_device, presentationReadySemaphore, nullptr);
        vkDestroyFence(m_device, drawingFinishedFence, nullptr);

        vkDestroyPipelineLayout(m_device, m_graphicsPipelineLayout, nullptr);
        vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

        for (auto descriptorSetLayout : m_graphicsDescriptorSetLayouts)
            vkDestroyDescriptorSetLayout(m_device, descriptorSetLayout, nullptr);

        vkDestroyDescriptorPool(m_device, m_graphicsDescriptorPool, nullptr);

        cleanupSwapChain();

        // for (auto framebuffer : m_framebuffers)
        //     vkDestroyFramebuffer(m_device, framebuffer, nullptr);
        // for (VkImageView &imageView : m_swapChainImageViews)
        //     vkDestroyImageView(m_device, imageView, nullptr);
        // vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);

        vkDestroyRenderPass(m_device, m_renderPass, nullptr);

        // Buffers
        vkFreeMemory(m_device, m_vertexBufferMemory, nullptr);
        vkFreeMemory(m_device, m_uniformBufferMemory, nullptr);
        vkDestroyBuffer(m_device, m_vertexBuffer, nullptr);
        vkDestroyBuffer(m_device, m_uniformBuffer, nullptr);

        // Images

        vkFreeMemory(m_device, m_presentImageMemory, nullptr);
        vkDestroyImageView(m_device, m_presentImageView, nullptr);
        vkDestroyImage(m_device, m_presentImage, nullptr);
        vkDestroySampler(m_device, m_presentImageSampler, nullptr);

        vkFreeMemory(m_device, m_computeImageMemory, nullptr);
        vkDestroyImageView(m_device, m_computeImageView, nullptr);
        vkDestroyImage(m_device, m_computeImage, nullptr);

        vkDestroyDescriptorPool(m_device, m_computeDescriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_computeDescriptorSetLayout, nullptr);

        // pipeline objects
        for (auto pipeline : m_computePipelines)
            vkDestroyPipeline(m_device, pipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_computePipelineLayout, nullptr);

        vkDestroyCommandPool(m_device, m_commandPool, nullptr);

        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        vkDestroyDevice(m_device, nullptr);
        vkDestroyInstance(m_instance, nullptr);
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    VkExtent2D getCurrentWindowExtent()
    {
        int width, height;
        glfwGetFramebufferSize(m_window, &width, &height);
        return VkExtent2D{ static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
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
        // std::clog << "available swap chain image usages: " << std::bitset<32>(details.capabilities.supportedUsageFlags) << std::endl;
        // std::clog << "available swap chain image usages: " << std::bitset<32>(details.capabilities.supportedUsageFlags) << std::endl;

        // std::vector<uint32_t> imageUsages = { VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT };

        // for (auto i : imageUsages)
        // {
        //     if ((i & details.capabilities.supportedUsageFlags) == i)
        //         std::clog << "\t" << i << std::endl;
        // }

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
        // std::clog << "Available swapchain image formats: " << std::endl;
        // for (const auto &format : availableFormats)
        // {
        //     std::clog << "\t" << format.format << std::endl;
        //     std::clog << "\t" << format.colorSpace << std::endl;
        // }

        for (const auto &format : availableFormats)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return format;
            // if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            //     return format;
        }

        return availableFormats[0];
    }

    VkPresentModeKHR choosePresentMode(std::vector<VkPresentModeKHR> &availablePresentModes)
    {
        for (const auto &presentMode : availablePresentModes)
        {
            if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
                // if (presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
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

        if (enableConsoleOutput)
        {
            std::clog << "memory requirements:\n";
            std::clog << "\t suitable memory type index bits: " << std::bitset<32>(memoryRequirements.memoryTypeBits) << std::endl;
            std::clog << "\t memory alignment: " << memoryRequirements.alignment << std::endl;
            std::clog << "\t memory size: " << memoryRequirements.size << std::endl;
        }

        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);

        // for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
        for (uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; i++)
        {
            bool currentMemoryTypeSupported = memoryRequirements.memoryTypeBits & (1 << i);
            bool allRequiredFlagsAvailable = (memoryProperties.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags;
            if (enableConsoleOutput)
            {
                std::clog << "memory type count: \t" << memoryProperties.memoryTypeCount << std::endl;
                std::clog << "suitable memory bits: \t" << std::bitset<32>(memoryRequirements.memoryTypeBits) << std::endl;
                std::clog << "current bit: \t\t" << std::bitset<32>(1 << i) << std::endl;
                std::clog << "memory property flags: \t" << std::bitset<32>(memoryProperties.memoryTypes[i].propertyFlags) << std::endl;
                std::clog << "required flags: \t" << std::bitset<32>(requiredFlags) << std::endl;
                std::clog << "memory heap flags: \t" << std::bitset<32>(memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].flags) << std::endl;
                std::clog << "memory heap index: \t" << (memoryProperties.memoryTypes[i].heapIndex) << std::endl;
                std::clog << "memory heap size: \t" << (memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size) << "\n\n";
            }
            if (allRequiredFlagsAvailable && currentMemoryTypeSupported)
            {
                selectedMemoryType = i;
                if (enableConsoleOutput)
                    std::clog << "\t\tSelected Memory Type: " << selectedMemoryType.value() << std::endl;
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

    void recreateSwapChain()
    {
        // check if window is minimised
        int width = 0, height = 0;
        glfwGetFramebufferSize(m_window, &width, &height);
        while (width == 0 || height == 0)
        {
            if (enableConsoleOutput)
                std::clog << "minimised!" << std::endl;
            glfwGetFramebufferSize(m_window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(m_device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createFramebuffers();

        m_width = m_swapChainExtent.width;
        m_height = m_swapChainExtent.height;

        // recreate present image
        destroyImage(m_device, m_presentImageMemory, m_presentImageView, m_presentImage);
        createPresentImage();

        // recreate compute image
        destroyImage(m_device, m_computeImageMemory, m_computeImageView, m_computeImage);
        createComputeImage();

        // recreate compute descriptor sets
        // vkFreeDescriptorSets(m_device, m_computeDescriptorPool, 1, &m_computeDescriptorSet);
        vkDestroyDescriptorPool(m_device, m_computeDescriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_computeDescriptorSetLayout, nullptr);
        createComputeDescriptorSets();

        // recreate graphics descriptor sets
        // vkFreeDescriptorSets(m_device, m_graphicsDescriptorPool, m_graphicsDescriptorSets.size(), m_graphicsDescriptorSets.data());
        vkDestroyDescriptorPool(m_device, m_graphicsDescriptorPool, nullptr);
        for (auto &layout : m_graphicsDescriptorSetLayouts)
            vkDestroyDescriptorSetLayout(m_device, layout, nullptr);
        createGraphicsDescriptorSets();
    }

    void cleanupSwapChain()
    {
        for (size_t i = 0; i < m_framebuffers.size(); i++)
            vkDestroyFramebuffer(m_device, m_framebuffers[i], nullptr);

        for (size_t i = 0; i < m_swapChainImageViews.size(); i++)
            vkDestroyImageView(m_device, m_swapChainImageViews[i], nullptr);

        vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);
    }
};

int main()
{
    std::unique_ptr<Game> game = std::make_unique<Game>(2560, 16.0f / 9.0f);
    // std::unique_ptr<Game> game = std::make_unique<Game>(1920, 16.0f / 9.0f);
    // std::unique_ptr<Game> game = std::make_unique<Game>(2048, 4.0f / 3.0f);
    // std::unique_ptr<Game> game = std::make_unique<Game>(1024, 4.0f / 3.0f);
    // std::unique_ptr<Game> game = std::make_unique<Game>(640, 4.0f / 3.0f);
    // std::unique_ptr<Game> game = std::make_unique<Game>(320, 4.0f / 3.0f);
    game->run();
    return EXIT_SUCCESS;
}