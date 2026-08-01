# executable
set( EXE_NAME EncoderApp )

# get source files
file( GLOB SRC_FILES "*.cpp" )

# get include files
file( GLOB INC_FILES "*.h" )

# get additional libs for gcc on Ubuntu systems
if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
  if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
    if( USE_ADDRESS_SANITIZER )
      set( ADDITIONAL_LIBS asan )
    endif()
  endif()
endif()

# NATVIS files for Visual Studio
if( MSVC )
  file( GLOB NATVIS_FILES "../../VisualStudio/*.natvis" )
  # extend the stack size on windows to 2MB
  set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /STACK:0x200000" )
endif()

# add executable
add_executable( ${EXE_NAME} ${SRC_FILES} ${INC_FILES} ${NATVIS_FILES} )
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if( DEFINED ENABLE_TRACING )
  if( ENABLE_TRACING )
    target_compile_definitions( ${EXE_NAME} PUBLIC ENABLE_TRACING=1 )
  else()
    target_compile_definitions( ${EXE_NAME} PUBLIC ENABLE_TRACING=0 )
  endif()
endif()

if( DEFINED ENABLE_HIGH_BITDEPTH )
  if( ENABLE_HIGH_BITDEPTH )
    target_compile_definitions( ${EXE_NAME} PUBLIC RExt__HIGH_BIT_DEPTH_SUPPORT=1 )
  else()
    target_compile_definitions( ${EXE_NAME} PUBLIC RExt__HIGH_BIT_DEPTH_SUPPORT=0 )
  endif()
endif()

if( CMAKE_COMPILER_IS_GNUCC AND BUILD_STATIC )
  set( ADDITIONAL_LIBS ${ADDITIONAL_LIBS} -static -static-libgcc -static-libstdc++ )
  target_compile_definitions( ${EXE_NAME} PUBLIC ENABLE_WPP_STATIC_LINK=1 )
endif()

target_link_libraries( ${EXE_NAME} PRIVATE CommonLib EncoderLib DecoderLib Utilities CudaBackend ${ADDITIONAL_LIBS} )

if( EXTENSION_360_VIDEO )
  target_link_libraries( ${EXE_NAME} PRIVATE Lib360 AppEncHelper360 )
endif()

if( EXTENSION_HDRTOOLS )
  target_link_libraries( ${EXE_NAME} PRIVATE HDRLib )
endif()

# lldb custom data formatters
if( XCODE )
  add_dependencies( ${EXE_NAME} Install${PROJECT_NAME}LldbFiles )
endif()

if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
  add_custom_command( TARGET ${EXE_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
                                                          $<$<CONFIG:Debug>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}/EncoderApp>
                                                          $<$<CONFIG:Release>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}/EncoderApp>
                                                          $<$<CONFIG:RelWithDebInfo>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO}/EncoderApp>
                                                          $<$<CONFIG:MinSizeRel>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL}/EncoderApp>
                                                          $<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/bin/EncoderAppStaticd>
                                                          $<$<CONFIG:Release>:${CMAKE_SOURCE_DIR}/bin/EncoderAppStatic>
                                                          $<$<CONFIG:RelWithDebInfo>:${CMAKE_SOURCE_DIR}/bin/EncoderAppStaticp>
                                                          $<$<CONFIG:MinSizeRel>:${CMAKE_SOURCE_DIR}/bin/EncoderAppStaticm> )
endif()

# example: place header files in different folders
source_group( "Natvis Files" FILES ${NATVIS_FILES} )

# set the folder where to place the projects
set_target_properties( ${EXE_NAME}  PROPERTIES FOLDER app LINKER_LANGUAGE CXX )

if( VTM_BUILD_CUDA_BACKEND_TESTS )
  set( ENCODER_BACKEND_SMOKE_INPUT "${CMAKE_CURRENT_BINARY_DIR}/encoder-backend-smoke.yuv" )
  set( ENCODER_BACKEND_SMOKE_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/encoder-backend-smoke.vvc" )
  file( GENERATE OUTPUT "${ENCODER_BACKEND_SMOKE_INPUT}" CONTENT "0" )

  add_test( NAME EncoderApp.Help COMMAND ${EXE_NAME} --help )
  add_test( NAME EncoderApp.InvalidBackend COMMAND ${EXE_NAME} --GPUBackend=bogus )
  add_test( NAME EncoderApp.InvalidDevice COMMAND ${EXE_NAME} --GPUDevice=-1 )
  set_tests_properties(
    EncoderApp.Help
    EncoderApp.InvalidBackend
    EncoderApp.InvalidDevice
    PROPERTIES WILL_FAIL TRUE
  )

  if( ENABLE_CUDA )
    set( ENCODER_BACKEND_FAILURE_TEST EncoderApp.InvalidCudaDeviceAfterCreate )
    set( ENCODER_BACKEND_FAILURE_ARGS --GPUBackend=cuda --GPUDevice=999 )

    # Deterministic raw 4:2:0 inputs for real process-level fail-fast coverage. Any byte pattern is valid raw YUV;
    # the sizes are exactly two 64x64 frames for inter SAD and one 512x512 frame for QPA.
    set( ENCODER_CUDA_SAD_INPUT "${CMAKE_CURRENT_BINARY_DIR}/encoder-cuda-sad-failfast.yuv" )
    set( ENCODER_CUDA_QPA_INPUT "${CMAKE_CURRENT_BINARY_DIR}/encoder-cuda-qpa-failfast.yuv" )
    string( REPEAT "0123456789abcdef" 768 ENCODER_CUDA_SAD_CONTENT )
    string( REPEAT "0123456789abcdef" 24576 ENCODER_CUDA_QPA_CONTENT )
    file( GENERATE OUTPUT "${ENCODER_CUDA_SAD_INPUT}" CONTENT "${ENCODER_CUDA_SAD_CONTENT}" )
    file( GENERATE OUTPUT "${ENCODER_CUDA_QPA_INPUT}" CONTENT "${ENCODER_CUDA_QPA_CONTENT}" )
    unset( ENCODER_CUDA_SAD_CONTENT )
    unset( ENCODER_CUDA_QPA_CONTENT )

    add_test(
      NAME EncoderApp.CudaSadFailFast
      COMMAND ${CMAKE_COMMAND}
        "-DTEST_EXECUTABLE=$<TARGET_FILE:${EXE_NAME}>"
        "-DTEST_KIND=SAD"
        "-DTEST_INPUT=${ENCODER_CUDA_SAD_INPUT}"
        "-DTEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/encoder-cuda-sad-failfast.vvc"
        "-DTEST_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/RunCudaFailFastTest.cmake"
    )
    set_tests_properties( EncoderApp.CudaSadFailFast PROPERTIES
      RUN_SERIAL TRUE LABELS gpu
    )

    add_test(
      NAME EncoderApp.CudaQpaFailFast
      COMMAND ${CMAKE_COMMAND}
        "-DTEST_EXECUTABLE=$<TARGET_FILE:${EXE_NAME}>"
        "-DTEST_KIND=QPA"
        "-DTEST_INPUT=${ENCODER_CUDA_QPA_INPUT}"
        "-DTEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/encoder-cuda-qpa-failfast.vvc"
        "-DTEST_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/RunCudaFailFastTest.cmake"
    )
    set_tests_properties( EncoderApp.CudaQpaFailFast PROPERTIES
      RUN_SERIAL TRUE LABELS gpu
    )
  else()
    set( ENCODER_BACKEND_FAILURE_TEST EncoderApp.CudaUnavailableAfterCreate )
    set( ENCODER_BACKEND_FAILURE_ARGS --GPUBackend=cuda )
  endif()
  add_test(
    NAME ${ENCODER_BACKEND_FAILURE_TEST}
    COMMAND ${EXE_NAME}
      -c "${CMAKE_SOURCE_DIR}/cfg/encoder_intra_vtm.cfg"
      -i "${ENCODER_BACKEND_SMOKE_INPUT}"
      -b "${ENCODER_BACKEND_SMOKE_OUTPUT}"
      --SourceWidth=16 --SourceHeight=16 --FrameRate=1 --FramesToBeEncoded=1
      --InputBitDepth=8 --InputChromaFormat=420
      ${ENCODER_BACKEND_FAILURE_ARGS}
  )
  set_tests_properties( ${ENCODER_BACKEND_FAILURE_TEST} PROPERTIES WILL_FAIL TRUE RUN_SERIAL TRUE )
endif()

