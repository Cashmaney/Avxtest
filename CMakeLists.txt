cmake_minimum_required(VERSION 3.7)
project(avxsse)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mno-vzeroupper -O3 -save-temps")

add_executable(avxsse ${SOURCE_FILES})