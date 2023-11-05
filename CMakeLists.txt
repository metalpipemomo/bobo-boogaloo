cmake_minimum_required(VERSION 3.25)

if(NOT "${CMAKE_BINARY_DIR}" MATCHES "/build$")
    message(FATAL_ERROR "Please create a 'build' directory and run CMake from there. This helps to keep the source directory clean.")
endif()

project(bobo-boogaloo)

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

# Add externals
add_subdirectory(libs/glfw)
include_directories(libs/glm)
add_subdirectory(libs/Jolt/Build)
include_directories(libs/stb)
include_directories(libs/tinyobjloader)

# Find Vulkan package
find_package(Vulkan REQUIRED)

# Glob shader files recursively
file(GLOB_RECURSE SHADER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/*.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/*.frag"
)


# Function to compile GLSL to SPIR-V
function(compile_glsl source_file target_file target_name)

    add_custom_command(
        OUTPUT ${target_file}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${source_file} -o ${target_file}
        DEPENDS ${source_file}
        COMMENT "Compiling GLSL ${source_file} to SPIR-V"
    )
    add_custom_target(
        compile_shaders_${target_name} ALL
        DEPENDS ${target_file}
    )
endfunction(compile_glsl)


# Initialize an empty list to collect all shader targets
set(ALL_SHADER_TARGETS "")

# Loop through shader sources to collect all shader targets
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER} EXT)
    get_filename_component(SHADER_PATH ${SHADER} DIRECTORY)

    file(RELATIVE_PATH RELATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders" ${SHADER_PATH})

    # Set the output SPIR-V path using the relative path inside the build directory
    if(SHADER_EXT STREQUAL ".vert")
        set(SPIRV "${CMAKE_CURRENT_BINARY_DIR}/${RELATIVE_PATH}/${SHADER_NAME}.vert.spv")
    elseif(SHADER_EXT STREQUAL ".frag")
        set(SPIRV "${CMAKE_CURRENT_BINARY_DIR}/${RELATIVE_PATH}/${SHADER_NAME}.frag.spv")
    else()
        message(FATAL_ERROR "Unsupported shader extension: ${SHADER_EXT}")
    endif()

    # Ensure the destination directory exists
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${RELATIVE_PATH}")

    # Generate a sanitized target name using the relative path and shader name
    string(REPLACE "\\" "_" TEMP_SANITIZED_DIR_NAME ${RELATIVE_PATH})
    string(REPLACE "/" "_" TEMP_SANITIZED_DIR_NAME_2 ${TEMP_SANITIZED_DIR_NAME})
    string(REPLACE ":" "_" SANITIZED_DIR_NAME ${TEMP_SANITIZED_DIR_NAME_2})

    
    set(SANITIZED_TARGET_NAME "${SANITIZED_DIR_NAME}_${SHADER_NAME}${SHADER_EXT}")
    set(TARGET_NAME "compile_shaders_${SANITIZED_TARGET_NAME}")

    compile_glsl(${SHADER} ${SPIRV} ${SANITIZED_TARGET_NAME})

    list(APPEND ALL_SHADER_TARGETS ${TARGET_NAME})
endforeach(SHADER)

# Define your executable target
add_executable(bobo-boogaloo
    src/HelloTriangle.h
    src/main.cpp)

# Preprocessor Macro
if (NOT MSVC)
    target_compile_definitions(bobo-boogaloo PRIVATE MAC_TARGET)
endif()

# Handle linking
target_link_libraries(bobo-boogaloo glfw Jolt Vulkan::Vulkan)

add_dependencies(bobo-boogaloo ${ALL_SHADER_TARGETS})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT bobo-boogaloo)

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
                    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
endif()

