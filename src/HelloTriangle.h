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
#include <cstring>
#include <optional>

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

    bool IsComplete()
    {
        return m_GraphicsFamily.has_value();
    }
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
        QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice gpu);
        bool IsGPUSuitable(VkPhysicalDevice gpu);
        void SelectGPU();
        void CreateLogicalDevice();
        void Loop();
        void Cleanup();

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
                VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                void* pUserData
        );

        GLFWwindow* p_Window;
        VkInstance m_Instance;
        VkDebugUtilsMessengerEXT m_DebugMessenger;
        VkPhysicalDevice m_GPU = VK_NULL_HANDLE;
        VkDevice m_LogicalDevice;
        VkQueue m_GraphicsQueue;
};

void HelloTriangle::Run()
{
    Init();
    
    Loop();

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
    SelectGPU();
    CreateLogicalDevice();
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

        if (indices.IsComplete()) break;

        idx++;
    }

    return indices;
}

bool HelloTriangle::IsGPUSuitable(VkPhysicalDevice gpu)
{
    QueueFamilyIndices indices = FindQueueFamilies(gpu);

    return indices.IsComplete();
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

    VkDeviceQueueCreateInfo queueCreateInfo{};
    
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = indices.m_GraphicsFamily.value();
    queueCreateInfo.queueCount = 1;

    f32 queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures gpuFeatures{};
    VkDeviceCreateInfo createInfo{};

    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
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
}

void HelloTriangle::Loop()
{
    while (!glfwWindowShouldClose(p_Window))
    {
        glfwPollEvents();
    }
}

void HelloTriangle::Cleanup()
{
    vkDestroyDevice(m_LogicalDevice, nullptr);
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

