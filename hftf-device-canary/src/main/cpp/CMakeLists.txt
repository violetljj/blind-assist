cmake_minimum_required(VERSION 3.22.1)
project(dav2_preprocess_native LANGUAGES CXX)

find_package(OpenCV REQUIRED CONFIG)

set(QAIRT_INCLUDE_DIR "${QAIRT_ROOT}/include/QNN")
if(EXISTS "${QAIRT_INCLUDE_DIR}/QnnInterface.h")
    set(DAV2_QNN_AVAILABLE 1)
else()
    set(DAV2_QNN_AVAILABLE 0)
    message(WARNING "QAIRT headers unavailable; cached-context JNI will fail closed")
endif()

add_library(
    dav2_preprocess_native
    SHARED
    dav2_preprocess_native.cpp
    dav2_yuv420_rgb.cpp
    dav2_qnn_cached_context.cpp
)
target_compile_features(dav2_preprocess_native PRIVATE cxx_std_17)
target_compile_options(dav2_preprocess_native PRIVATE -O3 -fno-fast-math -ffp-contract=off)
target_compile_definitions(dav2_preprocess_native PRIVATE DAV2_QNN_AVAILABLE=${DAV2_QNN_AVAILABLE})
if(DAV2_QNN_AVAILABLE)
    target_include_directories(dav2_preprocess_native PRIVATE "${QAIRT_INCLUDE_DIR}")
endif()
target_link_libraries(
    dav2_preprocess_native
    PRIVATE
    OpenCV::opencv_java4
    android
    dl
    log
)
