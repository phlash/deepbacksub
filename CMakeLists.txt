cmake_minimum_required(VERSION 3.16)
project(deepbacksub CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video videoio highgui)
find_package(PkgConfig REQUIRED)
pkg_check_modules(Minizip minizip REQUIRED IMPORTED_TARGET)

# generate Metadata schema with flatc
# - for which we provide a custom command and a custom target
set(GEN_HEADER_PATH ${CMAKE_CURRENT_LIST_DIR})
add_custom_command(OUTPUT ${GEN_HEADER_PATH}/metadata_schema_generated.h
    COMMAND flatc -c --scoped-enums -o ${GEN_HEADER_PATH}/ ${GEN_HEADER_PATH}/metadata_schema.fbs
    COMMENT "Compiling schema ${GEN_HEADER_PATH}/metadata_schema.fbs"
    DEPENDS ${GEN_HEADER_PATH}/metadata_schema.fbs)
add_custom_target(gen-headers ALL
    DEPENDS ${GEN_HEADER_PATH}/metadata_schema_generated.h)

# This assumes that tensorflow got checked out as submodule.
# True, if `git clone` had the additional option `--recursive`.
add_subdirectory(tensorflow/tensorflow/lite 
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

# add dependency from tensorflow-lite library to generated headers custom target
add_dependencies(tensorflow-lite gen-headers)

add_executable(deepseg
  deepseg.cc
  loopback.cc 
  transpose_conv_bias.cc
)

target_link_libraries(deepseg 
      tensorflow-lite ${CMAKE_DL_LIBS}
      opencv_core 
      opencv_video
      opencv_videoio
      opencv_imgproc
      opencv_imgcodecs
      opencv_highgui
      PkgConfig::Minizip
)


install(TARGETS deepseg)
install(DIRECTORY models 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/deepbacksub 
    FILES_MATCHING PATTERN "*.tflite" PATTERN "*.md")
