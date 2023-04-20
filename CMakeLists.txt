cmake_minimum_required(VERSION 3.22)

# ----------------------------------------------------------------------------
# Defining the CUDA Neural Network project
# ----------------------------------------------------------------------------

project(cuda-nn
	VERSION 0.1.0
	DESCRIPTION ""
	LANGUAGES CXX CUDA
)

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------

include(FetchContent)
FetchContent_Declare(
	spdlog
	GIT_REPOSITORY https://github.com/gabime/spdlog.git
	GIT_TAG        v1.10.0
)
FetchContent_MakeAvailable(spdlog)

set(THREADS_PREFER_PTHREAD_FLAG ON)  # Set -pthreads
find_package(Threads REQUIRED)
find_package(CUDAToolkit)

add_subdirectory(src/mnist)

# ----------------------------------------------------------------------------
# Building
# ----------------------------------------------------------------------------

add_executable(cuda-nn
	src/main.cpp
	src/matrix2d.cu
)

target_compile_options(cuda-nn PRIVATE
	$<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -Wpedantic $<$<CONFIG:RELEASE>:-O2 -flto>>
	$<$<COMPILE_LANGUAGE:CUDA>:-g -lineinfo -use_fast_math>
)
target_compile_features(cuda-nn PRIVATE cxx_std_20 cuda_std_17)

target_include_directories(cuda-nn
PRIVATE
	src
)

target_link_libraries(cuda-nn
PUBLIC
	Threads::Threads
	spdlog::spdlog
	CUDA::cudart_static
	CUDA::cuda_driver
	mnist
)
