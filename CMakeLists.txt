cmake_minimum_required(VERSION 3.10)
project(MLFramework)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(MY_DEBUG_OPTIONS "-ggdb;-pedantic-errors;-Wall;-Weffc++;-Wextra;-Wconversion;-Wsign-conversion")

# Add include and source directories
include_directories(include)
include_directories(/usr/include/gtest)

# Add the Google Test subdirectory if needed
# You can remove or comment out this line if you use find_package
# add_subdirectory(path/to/googletest)


# Enable testing
enable_testing()
find_package(GTest REQUIRED)
add_subdirectory(src)
add_subdirectory(tests)
