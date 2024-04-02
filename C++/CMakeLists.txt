cmake_minimum_required(VERSION 3.10)

enable_testing()

project(NEURAL_NET VERSION 1.0 LANGUAGES CXX CUDA)

###############
# C++ Options #
###############
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Set include directories
message("--Source directory: ${NEURAL_NET_SOURCE_DIR}")
include_directories(${NEURAL_NET_SOURCE_DIR}/include)

# Get CPP files
file(GLOB SRC ${NEURAL_NET_SOURCE_DIR}/src/*cpp ${NEURAL_NET_SOURCE_DIR}/src/*cu)
message("--src: ${SRC}")

# Get executable files
file(GLOB EXECLIST ${NEURAL_NET_SOURCE_DIR}/bin/main.cu)
message("--Exec List: ${EXECLIST}")

####Google Test #####
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


# src libraries
if(NOT SRC STREQUAL "")
  set(LIBNAME "Neural_Net")
  add_library(${LIBNAME} ${SRC})
endif()

# executables
foreach(EXEC ${EXECLIST})
  get_filename_component(EXECNAME ${EXEC} NAME_WE)
  add_executable(${EXECNAME} ${EXEC})

  if(NOT SRC STREQUAL "")
    message("SRC ${SRC}")
    message("${EXECNAME}")
    message("${LIBNAME}")
    target_link_libraries(${EXECNAME} LINK_PUBLIC ${LIBNAME})

    set_target_properties(${EXECNAME} PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

    set_property(TARGET ${EXECNAME} PROPERTY CUDA_ARCHITECTURES 50 72)
  endif()

  message("-- Adding executable: ${EXECNAME}")
endforeach(EXEC)
set_property(TARGET Neural_Net PROPERTY CUDA_ARCHITECTURES 50 72)

add_executable(
  example2_suite
  tests/example2_tests.cu
  tests/test_utils/example2.cu
)
add_executable(
  mono_threaded
  tests/mono_thread.cu
)

target_link_libraries(
  example2_suite
  GTest::gtest_main ${LIBNAME}
)
target_link_libraries(
  mono_threaded
  GTest::gtest_main ${LIBNAME}
)

add_executable(
  multi_threaded
  tests/multi_thread.cu
)

target_link_libraries(
  multi_threaded
  GTest::gtest_main ${LIBNAME}
)

add_executable(
  main_tests
  tests/test.cu
)

target_link_libraries(
  main_tests
  GTest::gtest_main ${LIBNAME}
)
set_property(TARGET mono_threaded PROPERTY CUDA_ARCHITECTURES 50 72)
set_property(TARGET multi_threaded PROPERTY CUDA_ARCHITECTURES 50 72)
set_property(TARGET example2_suite PROPERTY CUDA_ARCHITECTURES 50 72)
set_property(TARGET main_tests PROPERTY CUDA_ARCHITECTURES 50 72)
include(GoogleTest)
gtest_discover_tests(mono_threaded)
gtest_discover_tests(multi_threaded)
gtest_discover_tests(main_tests)
gtest_discover_tests(example2_suite)