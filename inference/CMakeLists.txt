cmake_minimum_required(VERSION 3.10)
project(inference)

set(TVM_COMMON_INCLUDE  ${CMAKE_SOURCE_DIR}/tvm_common/include)
file(GLOB TVM_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tvm_common/src/*.cpp)

include_directories(
    ${TVM_COMMON_INCLUDE}
)

file(GLOB CPP_FILES "*.cpp")
foreach(CPP_FILE ${CPP_FILES})
    get_filename_component(TARGET_NAME ${CPP_FILE} NAME_WE)
    message(STATUS ${TARGET_NAME})
    message(STATUS ${CPP_FILE})
    message(STATUS ${TVM_COMMON_SOURCE})
    add_executable(${TARGET_NAME} ${CPP_FILE} ${TVM_COMMON_SOURCE})
    target_link_libraries(${TARGET_NAME} tvm_runtime gtest gtest_main)
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/install)
endforeach()

file(GLOB OUTPUT_FILES "${CMAKE_SOURCE_DIR}/output_*")
file(COPY ${OUTPUT_FILES} DESTINATION ${CMAKE_BINARY_DIR}/install)
