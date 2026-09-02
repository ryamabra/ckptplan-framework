cmake_minimum_required(VERSION 3.20)

project(ckptplan_native LANGUAGES CXX)

option(CKPTPLAN_BUILD_TESTS "Build the native planner tests" ON)
option(CKPTPLAN_ENABLE_CUDA "Build CUDA memory-accounting tools" OFF)

add_library(ckptplan_planner STATIC
    src/planner.cpp
)
add_library(ckptplan::planner ALIAS ckptplan_planner)

target_include_directories(ckptplan_planner
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(ckptplan_planner PUBLIC cxx_std_20)
set_target_properties(ckptplan_planner PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
    target_compile_options(ckptplan_planner PRIVATE /W4 /permissive-)
else()
    target_compile_options(ckptplan_planner PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(CKPTPLAN_BUILD_TESTS)
    enable_testing()
    add_executable(ckptplan_planner_test tests/planner_test.cpp)
    target_link_libraries(ckptplan_planner_test PRIVATE ckptplan::planner)
    add_test(NAME ckptplan_planner_test COMMAND ckptplan_planner_test)
endif()

if(CKPTPLAN_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(ckptplan_cuda STATIC cuda/memory_accounting.cu)
    target_include_directories(ckptplan_cuda PUBLIC include)
    target_link_libraries(ckptplan_cuda PUBLIC CUDA::cudart)
    target_compile_features(ckptplan_cuda PUBLIC cuda_std_17)
    set_target_properties(ckptplan_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    add_executable(ckptplan_cuda_demo examples/cuda_memory_demo.cu)
    target_link_libraries(ckptplan_cuda_demo PRIVATE ckptplan_cuda)
endif()
