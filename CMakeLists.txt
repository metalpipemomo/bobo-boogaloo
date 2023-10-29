cmake_minimum_required(VERSION 3.25)

if(NOT "${CMAKE_BINARY_DIR}" MATCHES "/build$")
    message(FATAL_ERROR "Please create a 'build' directory and run CMake from there. This helps to keep the source directory clean.")
endif()

project(cross_platform_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Disable GLFW's unnecessary build targets
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)

# Add GLFW submodule and specify GLM include directory
add_subdirectory(libs/glfw)
include_directories(libs/glm)
add_subdirectory(libs/Jolt/Build)

# Find Vulkan package
find_package(Vulkan REQUIRED)

# Function to compile GLSL to SPIR-V
function(compile_glsl source_file target_file)
    string(REPLACE "/" "_" SANITIZED_TARGET_NAME ${target_file})
    add_custom_command(
        OUTPUT ${target_file}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${source_file} -o ${target_file}
        DEPENDS ${source_file}
        COMMENT "Compiling GLSL ${source_file} to SPIR-V"
    )
    add_custom_target(
        compile_shaders_${SANITIZED_TARGET_NAME} ALL
        DEPENDS ${target_file}
    )
endfunction(compile_glsl)

# Glob shader files recursively
file(GLOB_RECURSE SHADER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/*.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/*.frag"
)

# Initialize an empty list to collect all shader targets
set(ALL_SHADER_TARGETS "")

# Loop through an empty list to collect all shader targets 
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER} EXT)
    get_filename_component(SHADER_PATH ${SHADER} DIRECTORY)
    get_filename_component(SUB_DIR_NAME ${SHADER_PATH} NAME)

    if(SHADER_EXT STREQUAL ".vert")
        set(SPIRV "${SHADER_PATH}/${SHADER_NAME}.vert.spv")
    elseif(SHADER_EXT STREQUAL ".frag")
        set(SPIRV "${SHADER_PATH}/${SHADER_NAME}.frag.spv")
    else()
        message(FATAL_ERROR "Unsupported shader extension: ${SHADER_EXT}")
    endif()

    string(REPLACE "/" "_" SANITIZED_TARGET_NAME ${SPIRV})
    set(TARGET_NAME "compile_shaders_${SANITIZED_TARGET_NAME}")
    compile_glsl(${SHADER} ${SPIRV})
    
    list(APPEND ALL_SHADER_TARGETS ${TARGET_NAME})
endforeach(SHADER)

# Define your executable target
add_executable(cross_platform_test 
    src/HelloTriangle.h
    src/main.cpp)

# Link your target against GLFW and Vulkan
target_link_libraries(cross_platform_test glfw Jolt Vulkan::Vulkan)

add_dependencies(cross_platform_test ${ALL_SHADER_TARGETS})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT cross_platform_test)

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
                    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
endif()

