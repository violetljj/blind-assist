cmake_minimum_required(VERSION 3.22.1)
project(dav2_preprocess_native LANGUAGES CXX)

find_package(OpenCV REQUIRED CONFIG)

add_library(dav2_preprocess_native SHARED dav2_preprocess_native.cpp)
target_compile_features(dav2_preprocess_native PRIVATE cxx_std_17)
target_compile_options(dav2_preprocess_native PRIVATE -O3 -ffast-math)
target_link_libraries(
    dav2_preprocess_native
    PRIVATE
    OpenCV::opencv_java4
    android
    log
)
