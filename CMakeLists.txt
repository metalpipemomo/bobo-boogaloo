cmake_minimum_required(VERSION 3.25)

if(NOT "${CMAKE_BINARY_DIR}" MATCHES "/build$")
    message(FATAL_ERROR "Please create a 'build' directory and run CMake from there. This helps to keep the source directory clean.")
endif()

project(cross_platform_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Disable GLFW's unnecessary build targets
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)

# Add GLFW submodule and specify GLM include directory
add_subdirectory(libs/glfw)
include_directories(libs/glm)

# Find Vulkan package
find_package(Vulkan REQUIRED)

# Define your executable target
add_executable(cross_platform_test src/main.cpp)

# Link your target against GLFW and Vulkan
target_link_libraries(cross_platform_test glfw Vulkan::Vulkan)

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
                    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
endif()

