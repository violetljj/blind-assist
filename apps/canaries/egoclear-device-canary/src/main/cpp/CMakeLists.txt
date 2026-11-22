cmake_minimum_required(VERSION 3.22.1)
project(egoclear_snpe_jni LANGUAGES CXX)

add_library(SNPE SHARED IMPORTED)
set_target_properties(SNPE PROPERTIES IMPORTED_LOCATION "${QAIRT_ROOT}/lib/aarch64-android/libSNPE.so")

add_library(egoclear_snpe_jni SHARED egoclear_snpe_jni.cpp)
target_include_directories(egoclear_snpe_jni PRIVATE "${QAIRT_ROOT}/include/SNPE")
target_compile_features(egoclear_snpe_jni PRIVATE cxx_std_17)
target_compile_options(egoclear_snpe_jni PRIVATE -O3 -fno-fast-math -ffp-contract=off)
target_link_libraries(egoclear_snpe_jni SNPE)
target_link_options(egoclear_snpe_jni PRIVATE "-Wl,-z,max-page-size=16384" "-Wl,-z,common-page-size=16384")
