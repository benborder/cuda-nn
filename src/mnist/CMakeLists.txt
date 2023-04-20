# ----------------------------------------------------------------------------
# Download dataset
# ----------------------------------------------------------------------------

include(FetchContent)
FetchContent_Declare(
	mnist_train_images
	URL            http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	URL_HASH       MD5=f68b3c2dcbeaaa9fbdd348bbdeb94873
	DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_GetProperties(mnist_train_images)
if(NOT mnist_train_images_POPULATED)
    FetchContent_Populate(mnist_train_images)
    execute_process(
			COMMAND gzip -dk ${mnist_train_images_SOURCE_DIR}/train-images-idx3-ubyte.gz
			COMMAND ln -sf ${mnist_train_images_SOURCE_DIR}/train-images-idx3-ubyte ${mnist_train_images_BINARY_DIR}/train-images-idx3-ubyte
    )
endif()

FetchContent_Declare(
	mnist_train_labels
	URL            http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	URL_HASH       MD5=d53e105ee54ea40749a09fcbcd1e9432
	DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_GetProperties(mnist_train_labels)
if(NOT mnist_train_labels_POPULATED)
    FetchContent_Populate(mnist_train_labels)
    execute_process(
			COMMAND gzip -dk ${mnist_train_labels_SOURCE_DIR}/train-labels-idx1-ubyte.gz
			COMMAND ln -sf ${mnist_train_labels_SOURCE_DIR}/train-labels-idx1-ubyte ${mnist_train_labels_BINARY_DIR}/train-labels-idx1-ubyte
    )
endif()

FetchContent_Declare(
	mnist_test_images
	URL            http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	URL_HASH       MD5=9fb629c4189551a2d022fa330f9573f3
	DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_GetProperties(mnist_test_images)
if(NOT mnist_test_images_POPULATED)
    FetchContent_Populate(mnist_test_images)
    execute_process(
			COMMAND gzip -dk ${mnist_test_images_SOURCE_DIR}/t10k-images-idx3-ubyte.gz
			COMMAND ln -sf ${mnist_test_images_SOURCE_DIR}/t10k-images-idx3-ubyte ${mnist_test_images_BINARY_DIR}/t10k-images-idx3-ubyte
    )
endif()

FetchContent_Declare(
	mnist_test_labels
	URL            http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	URL_HASH       MD5=ec29112dd5afa0611ce80d1b7f02629c
	DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_GetProperties(mnist_test_labels)
if(NOT mnist_test_labels_POPULATED)
    FetchContent_Populate(mnist_test_labels)
    execute_process(
			COMMAND gzip -dk ${mnist_test_labels_SOURCE_DIR}/t10k-labels-idx1-ubyte.gz
			COMMAND ln -sf ${mnist_test_labels_SOURCE_DIR}/t10k-labels-idx1-ubyte ${mnist_test_labels_BINARY_DIR}/t10k-labels-idx1-ubyte
    )
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)  # Set -pthreads
find_package(Threads REQUIRED)

# ----------------------------------------------------------------------------
# Build MNIST dataset loader
# ----------------------------------------------------------------------------

add_library(mnist
	mnist.cpp
)

target_compile_options(mnist PRIVATE -Wall -Wextra -Werror -Wpedantic $<$<CONFIG:RELEASE>:-O2 -flto>)

target_compile_features(mnist PRIVATE cxx_std_20)

target_include_directories(mnist
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(mnist
PUBLIC
  Threads::Threads
PRIVATE
  $<BUILD_INTERFACE:spdlog::spdlog>
)

add_custom_target(mnist_dataset ALL
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${mnist_train_images_BINARY_DIR} ./mnist
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${mnist_train_labels_BINARY_DIR} ./mnist
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${mnist_test_images_BINARY_DIR} ./mnist
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${mnist_test_labels_BINARY_DIR} ./mnist
	COMMENT "Copying dataset"
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_dependencies(mnist mnist_dataset)

# ----------------------------------------------------------------------------
# Install MNIST dataset
# ----------------------------------------------------------------------------

install(
	DIRECTORY ${CMAKE_BINARY_DIR}/mnist/
	DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/mnist
)
