cmake_minimum_required(VERSION 3.4.1)

include_directories(${CMAKE_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/include)

set(xgen_path ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libxgen.so)
add_library (xgen SHARED IMPORTED)
set_target_properties(xgen PROPERTIES IMPORTED_LOCATION "${xgen_path}")

add_library(
        inference_api_jni
        SHARED
        src/main/cpp/inference_api.cc
)

message(STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})


find_library(
        log-lib
        log
)


find_library(
        android-lib
        android
)


if(ANDROID_ABI MATCHES "arm64-v8a")
    target_link_libraries(
            inference_api_jni
            xgen
            ${android-lib}
            ${log-lib}
    )
endif()
