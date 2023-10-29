#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_core.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <set>
#include <cstring>
#include <optional>
#include <algorithm>
#include <limits>
#include <fstream>

using i32 = int;
using u32 = unsigned int;
using f32 = float;

// Window Specs
const u32 WIDTH = 800;
const u32 HEIGHT = 600;

const std::vector<const char*> VALIDATION_LAYERS = { "VK_LAYER_KHRONOS_validation" };
const std::vector<const char*> DEVICE_EXTENSIONS = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_portability_subset" };

#ifdef NDEBUG
const bool ENABLE_VALIDATION_LAYERS = false;
#else
const bool ENABLE_VALIDATION_LAYERS = true;
#endif

VkResult CreateDebugUtilsMessengerExt(
        VkInstance instance,
        const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkDebugUtilsMessengerEXT* pDebugMessenger
        )
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else 
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerExt(
        VkInstance instance,
        VkDebugUtilsMessengerEXT debugMessenger,
        const VkAllocationCallbacks* pAllocator
        )
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices
{
    std::optional<u32> m_GraphicsFamily;
    std::optional<u32> m_PresentFamily;

    bool IsComplete()
    {
        return m_GraphicsFamily.has_value()
            && m_PresentFamily.has_value();
    }
};

struct SwapChainSupportInfo
{
    VkSurfaceCapabilitiesKHR m_Capabilities;
    std::vector<VkSurfaceFormatKHR> m_Formats;
    std::vector<VkPresentModeKHR> m_PresentModes;
};

class HelloTriangle
{
    public:
        void Run();

    private:
        void Init();
        void InitWindow();
        void InitVulkan();
        bool QueryValidationLayerSupport();
        std::vector<const char*> GetRequiredExtensions();
        void CreateDebugMessengerCreationInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
        void CreateDebugMessenger();
        void CreateVulkanInstance();
        void CreateSurface();
        QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice gpu);
        bool CheckDeviceExtensionSupport(VkPhysicalDevice gpu);
        bool IsGPUSuitable(VkPhysicalDevice gpu);
        void SelectGPU();
        void CreateLogicalDevice();
        VkSurfaceFormatKHR SelectSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats);
        VkPresentModeKHR SelectSwapPresentMode(const std::vector<VkPresentModeKHR>& presentModes);
        VkExtent2D SelectSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
        void CreateSwapChain();
        void CreateImageViews();
        VkShaderModule CreateShaderModule(const std::vector<char>& code);
        void CreateGraphicsPipeline();
        void Update();
        void Cleanup();
        
        SwapChainSupportInfo QuerySwapChainSupport(VkPhysicalDevice gpu);

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
                VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                void* pUserData
        );

        static std::vector<char> ReadFile(const std::string& fileName);

        GLFWwindow* p_Window;
        VkInstance m_Instance;
        VkDebugUtilsMessengerEXT m_DebugMessenger;
        VkPhysicalDevice m_GPU = VK_NULL_HANDLE;
        VkDevice m_LogicalDevice;
        VkQueue m_GraphicsQueue;
        VkSurfaceKHR m_Surface;
        VkQueue m_PresentQueue;
        VkSwapchainKHR m_SwapChain;
        std::vector<VkImage> m_SwapChainImages;
        VkFormat m_SwapChainImageFormat;
        VkExtent2D m_SwapChainExtent;
        std::vector<VkImageView> m_SwapChainImageViews;
        VkPipelineLayout m_PipelineLayout;
};

void HelloTriangle::Run()
{
    Init();
    
    Update();

    Cleanup();
}

void HelloTriangle::Init()
{
    InitWindow();
    InitVulkan();
}

void HelloTriangle::InitWindow()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    p_Window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
}

void HelloTriangle::InitVulkan()
{
    CreateVulkanInstance();
    CreateDebugMessenger();
    CreateSurface();
    SelectGPU();
    CreateLogicalDevice();
    CreateSwapChain();
}

bool HelloTriangle::QueryValidationLayerSupport()
{
    u32 layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (auto layerName : VALIDATION_LAYERS)
    {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers)
        {
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }

        }

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}

std::vector<const char*> HelloTriangle::GetRequiredExtensions()
{
    u32 glfwExtensionCount = 0;
    const char** glfwExtensions;

    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (ENABLE_VALIDATION_LAYERS)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

void HelloTriangle::CreateDebugMessengerCreationInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = 
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr;
}

void HelloTriangle::CreateDebugMessenger()
{
    if (!ENABLE_VALIDATION_LAYERS) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    CreateDebugMessengerCreationInfo(createInfo);

    if (CreateDebugUtilsMessengerExt(m_Instance, &createInfo, nullptr, &m_DebugMessenger) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to set up debug messenger!");
    }
}

void HelloTriangle::CreateVulkanInstance()
{
    if (ENABLE_VALIDATION_LAYERS && !QueryValidationLayerSupport())
    {
        throw std::runtime_error("Validation layers requested, but none are available!");
    }

    VkApplicationInfo appInfo{};

    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0); 
    appInfo.pEngineName = "Bobo";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    auto extensions = GetRequiredExtensions();

    extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.emplace_back("VK_KHR_get_physical_device_properties2");

    VkInstanceCreateInfo createInfo{};
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    createInfo.enabledExtensionCount = static_cast<u32>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = 0;
    createInfo.pNext = nullptr;

    if (ENABLE_VALIDATION_LAYERS)
    {
        createInfo.enabledLayerCount = static_cast<u32>(VALIDATION_LAYERS.size());
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        
        CreateDebugMessengerCreationInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    }

    if (vkCreateInstance(&createInfo, nullptr, &m_Instance) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan instance...");
    }
}

void HelloTriangle::CreateSurface()
{
    if (glfwCreateWindowSurface(m_Instance, p_Window, nullptr, &m_Surface) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create window surface.");
    }
}

QueueFamilyIndices HelloTriangle::FindQueueFamilies(VkPhysicalDevice gpu)
{
    QueueFamilyIndices indices;
    
    u32 queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queueFamilyCount, queueFamilies.data());

    i32 idx = 0;
    for (const auto& queueFamily : queueFamilies)
    {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            indices.m_GraphicsFamily = idx;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(gpu, idx, m_Surface, &presentSupport);

        if (presentSupport)
        {
            indices.m_PresentFamily = idx;
        }

        if (indices.IsComplete()) break;

        idx++;
    }

    return indices;
}

bool HelloTriangle::CheckDeviceExtensionSupport(VkPhysicalDevice gpu)
{
    u32 extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());

    for (const auto& extension : availableExtensions)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

SwapChainSupportInfo HelloTriangle::QuerySwapChainSupport(VkPhysicalDevice gpu)
{
    SwapChainSupportInfo info;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, m_Surface, &info.m_Capabilities);

    u32 formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, m_Surface, &formatCount, nullptr);

    if (formatCount)
    {
        info.m_Formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, m_Surface, &formatCount, info.m_Formats.data()); 
    }

    u32 presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, m_Surface, &presentModeCount, nullptr);

    if (presentModeCount)
    {
        info.m_PresentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, m_Surface, &presentModeCount, info.m_PresentModes.data());
    }

    return info;
}

bool HelloTriangle::IsGPUSuitable(VkPhysicalDevice gpu)
{
    QueueFamilyIndices indices = FindQueueFamilies(gpu);

    bool extensionsSupported = CheckDeviceExtensionSupport(gpu);

    bool swapChainAdequate = false;
    if (extensionsSupported)
    {
        SwapChainSupportInfo swapChainSupport = QuerySwapChainSupport(gpu);
        swapChainAdequate = 
            !swapChainSupport.m_Formats.empty() &&
            !swapChainSupport.m_PresentModes.empty();
    } 

    return indices.IsComplete() 
        && extensionsSupported
        && swapChainAdequate;
}

void HelloTriangle::SelectGPU()
{
    u32 deviceCount = 0;
    vkEnumeratePhysicalDevices(m_Instance, &deviceCount, nullptr);

    if (!deviceCount)
    {
        throw std::runtime_error("A GPU that supports Vulkan could not be found.");
    }

    std::vector<VkPhysicalDevice> gpus(deviceCount);
    vkEnumeratePhysicalDevices(m_Instance, &deviceCount, gpus.data());

    for (const auto& gpu : gpus)
    {
        if (IsGPUSuitable(gpu))
        {
            m_GPU = gpu;
            break;
        }
    }

    if (m_GPU == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Failed to find a suitable GPU.");
    }
}

void HelloTriangle::CreateLogicalDevice()
{
    QueueFamilyIndices indices = FindQueueFamilies(m_GPU);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<u32> uniqueQueueFamilies = { indices.m_GraphicsFamily.value(), indices.m_PresentFamily.value() };

    f32 queuePriority = 1.0f;
    for (u32 queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};

        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures gpuFeatures{};
    VkDeviceCreateInfo createInfo{};

    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<u32>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &gpuFeatures;
    createInfo.enabledExtensionCount = static_cast<u32>(DEVICE_EXTENSIONS.size());
    createInfo.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
    createInfo.enabledLayerCount = 0;

    if (ENABLE_VALIDATION_LAYERS)
    {
        createInfo.enabledLayerCount = static_cast<u32>(VALIDATION_LAYERS.size());
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }

    if (vkCreateDevice(m_GPU, &createInfo, nullptr, &m_LogicalDevice) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create logical device!");
    }

    vkGetDeviceQueue(m_LogicalDevice, indices.m_GraphicsFamily.value(), 0, &m_GraphicsQueue);
    vkGetDeviceQueue(m_LogicalDevice, indices.m_PresentFamily.value(), 0, &m_PresentQueue);
}

VkSurfaceFormatKHR HelloTriangle::SelectSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats)
{
    for (const auto& format : formats)
    {
        if (format.format == VK_FORMAT_B8G8R8_SRGB &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return format;
        }
    }

    return formats[0];
}

VkPresentModeKHR HelloTriangle::SelectSwapPresentMode(const std::vector<VkPresentModeKHR>& presentModes)
{
    for (const auto& presentMode : presentModes)
    {
        if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return presentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D HelloTriangle::SelectSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<u32>::max())
    {
        return capabilities.currentExtent;
    }

    i32 width, height;
    glfwGetFramebufferSize(p_Window, &width, &height);

    VkExtent2D actualExtent =
    {
        static_cast<u32>(width),
        static_cast<u32>(height)
    };

    actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return actualExtent;
}

void HelloTriangle::CreateSwapChain()
{
    SwapChainSupportInfo swapChainSupport = QuerySwapChainSupport(m_GPU);

    auto surfaceFormat = SelectSwapSurfaceFormat(swapChainSupport.m_Formats);
    auto presentMode = SelectSwapPresentMode(swapChainSupport.m_PresentModes);
    auto extent = SelectSwapExtent(swapChainSupport.m_Capabilities);

    u32 imageCount = swapChainSupport.m_Capabilities.minImageCount + 1;

    if (swapChainSupport.m_Capabilities.maxImageCount > 0
            && imageCount > swapChainSupport.m_Capabilities.maxImageCount)
    {
        imageCount = swapChainSupport.m_Capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = m_Surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = swapChainSupport.m_Capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;

    QueueFamilyIndices indices = FindQueueFamilies(m_GPU);
    u32 queueFamilyIndices[] = { indices.m_GraphicsFamily.value(), indices.m_PresentFamily.value() };

    if (indices.m_GraphicsFamily != indices.m_PresentFamily)
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = static_cast<u32>(sizeof(queueFamilyIndices) / sizeof(u32));
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }

    if (vkCreateSwapchainKHR(m_LogicalDevice, &createInfo, nullptr, &m_SwapChain) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(m_LogicalDevice, m_SwapChain, &imageCount, nullptr);
    m_SwapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_LogicalDevice, m_SwapChain, &imageCount, m_SwapChainImages.data());

    m_SwapChainImageFormat = surfaceFormat.format;
    m_SwapChainExtent = extent;
}

void HelloTriangle::CreateImageViews()
{
    m_SwapChainImageViews.resize(m_SwapChainImages.size());

    for (size_t i = 0; i < m_SwapChainImages.size(); i++)
    {
        VkImageViewCreateInfo createInfo{};

        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = m_SwapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = m_SwapChainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_LogicalDevice, &createInfo, nullptr, &m_SwapChainImageViews[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create image views!");
        }
    }

}

VkShaderModule HelloTriangle::CreateShaderModule(const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{};

    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const u32*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_LogicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader modules!");
    }

    return shaderModule;
}

void HelloTriangle::CreateGraphicsPipeline()
{
    auto vertShaderCode = ReadFile("shaders/basic/shader.vert.spv");
    auto fragShaderCode = ReadFile("shaders/basic/shader.frag.spv");

    VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    std::vector<VkDynamicState> dynamicStates =
    {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<u32>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<f32>(m_SwapChainExtent.width);
    viewport.height = static_cast<f32>(m_SwapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = m_SwapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = 
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_TRUE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(m_LogicalDevice, &pipelineLayoutInfo, nullptr, &m_PipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Pipeline Layout!");
    }

    vkDestroyShaderModule(m_LogicalDevice, fragShaderModule, nullptr);
    vkDestroyShaderModule(m_LogicalDevice, vertShaderModule, nullptr);
}

void HelloTriangle::Update()
{
    while (!glfwWindowShouldClose(p_Window))
    {
        glfwPollEvents();
    }
}

void HelloTriangle::Cleanup()
{
    vkDestroyPipelineLayout(m_LogicalDevice, m_PipelineLayout, nullptr);
    for (auto imageView : m_SwapChainImageViews)
    {
        vkDestroyImageView(m_LogicalDevice, imageView, nullptr);
    }
    vkDestroySwapchainKHR(m_LogicalDevice, m_SwapChain, nullptr);
    vkDestroyDevice(m_LogicalDevice, nullptr);
    vkDestroySurfaceKHR(m_Instance, m_Surface, nullptr);
    if (ENABLE_VALIDATION_LAYERS)
    {
        DestroyDebugUtilsMessengerExt(m_Instance, m_DebugMessenger, nullptr);
    }
    vkDestroyInstance(m_Instance, nullptr);
    glfwDestroyWindow(p_Window);
    glfwTerminate();
}


VKAPI_ATTR VkBool32 VKAPI_CALL HelloTriangle::debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData
        )
{
    std::cerr << "Validation Layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

std::vector<char> HelloTriangle::ReadFile(const std::string& fileName)
{
    std::ifstream file{ fileName, std::ios::ate | std::ios::binary };

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}





