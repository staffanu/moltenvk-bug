cmake_minimum_required(VERSION 3.20)
project(vulkan_flush_invalidate_repro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)

# Compile compute shader to SPIR-V at build time.
find_program(GLSLC glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES bin)
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found; install LunarG Vulkan SDK")
endif()

set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shader.comp)
set(SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shader.comp.spv)

add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND ${GLSLC} -O ${SHADER_SRC} -o ${SHADER_SPV}
    DEPENDS ${SHADER_SRC}
    VERBATIM
)
add_custom_target(shader DEPENDS ${SHADER_SPV})

add_executable(repro main.cpp)
add_dependencies(repro shader)
target_link_libraries(repro PRIVATE Vulkan::Vulkan)
target_compile_definitions(repro PRIVATE
    SHADER_SPV_PATH="${SHADER_SPV}"
)

add_executable(compute_repro compute_repro.cpp)
add_dependencies(compute_repro shader)
target_link_libraries(compute_repro PRIVATE Vulkan::Vulkan)
target_compile_definitions(compute_repro PRIVATE
    SHADER_SPV_PATH="${SHADER_SPV}"
)

# minimal_repro: no shader, no compute, no descriptors -- just vkCmdCopyBuffer.
add_executable(minimal_repro minimal_repro.cpp)
target_link_libraries(minimal_repro PRIVATE Vulkan::Vulkan)

if(APPLE)
    # MoltenVK requires the portability enumeration extension at instance level.
    target_compile_definitions(repro PRIVATE USE_PORTABILITY_ENUMERATION=1)
endif()
