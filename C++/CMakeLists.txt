cmake_minimum_required(VERSION 3.10)

project(NEURAL_NET VERSION 1.0 LANGUAGES CXX CUDA)
enable_testing()

find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})

# Set include directories
message("--Source directory: ${NEURAL_NET_SOURCE_DIR}")
include_directories(${NEURAL_NET_SOURCE_DIR}/include)

# Get CPP files
file(GLOB SRC ${NEURAL_NET_SOURCE_DIR}/src/*cpp ${NEURAL_NET_SOURCE_DIR}/src/*cu)
message("--src: ${SRC}")

# Get executable files
file(GLOB EXECLIST ${NEURAL_NET_SOURCE_DIR}/bin/*.cu ${NEURAL_NET_SOURCE_DIR}/bin/*.cpp)
message("--Exec List: ${EXECLIST}")

####Google Test #####
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
# determine build type based on directory name. Do not allow for in source building
#
if(${NEURAL_NET_SOURCE_DIR} STREQUAL ${NEURAL_NET_BINARY_DIR})
  message(FATAL_ERROR " *** In-source building not allowed. Please create a subdir 'Release' or 'Debug', and run cmake from within this directory 'cmake ..' ***")
else()
  get_filename_component(TYPE ${NEURAL_NET_BINARY_DIR} NAME)
  string(TOUPPER "${TYPE}" TYPE)
  if(${TYPE} STREQUAL "RELEASE")
    set(CMAKE_BUILD_TYPE Release)
  else()
    set(CMAKE_BUILD_TYPE Debug)
 endif()
 message("-- Build type set to: ${TYPE}")
endif()

add_executable(
  hello_test
  tests/test.cpp
)
target_link_libraries(
  hello_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(hello_test)

# src libraries
if(NOT SRC STREQUAL "")
  set(LIBNAME "Neural_Net")
  add_library(${LIBNAME} ${SRC})
endif()

target_link_libraries(Neural_Net ${CUDA_LIBRARIES} cublas)
# executables
foreach(EXEC ${EXECLIST})
  get_filename_component(EXECNAME ${EXEC} NAME_WE)
  add_executable(${EXECNAME} ${EXEC})

  if(NOT SRC STREQUAL "")
    message("EXEC ${EXECNAME}")
    target_link_libraries(${EXECNAME} LINK_PUBLIC ${LIBNAME} ${CUDA_LIBRARIES} cublas)

    set_property(TARGET ${EXECNAME} PROPERTY CUDA_ARCHITECTURES 50 72)
  endif()

  message("-- Adding executable: ${EXECNAME}")
endforeach(EXEC)
set_property(TARGET Neural_Net PROPERTY CUDA_ARCHITECTURES 50 72)
target_compile_options(Neural_Net PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
set_target_properties(Neural_Net PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)