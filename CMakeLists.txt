# minimum required cmake version
cmake_minimum_required( VERSION 3.10 FATAL_ERROR )

option( ENABLE_CUDA "Build the optional CUDA compute backend" OFF )
option( VTM_BUILD_CUDA_BACKEND_TESTS "Build the opt-in CUDA backend contract tests" OFF )
option( VTM_ENABLE_DECODER_BATCH_PROFILING
        "Build the opt-in decoder batching profiler (diagnostic builds only)" OFF )

if( ENABLE_CUDA AND CMAKE_VERSION VERSION_LESS 3.24 )
  message( FATAL_ERROR "ENABLE_CUDA requires CMake 3.24 or newer (found ${CMAKE_VERSION})" )
endif()

# project name
if( EXTENSION_360_VIDEO )
  project( NextSoftware360 )
else()
  project( NextSoftware )
endif()

if( ENABLE_CUDA )
  set( VTM_CUDA_ARCHITECTURES "native" CACHE STRING
       "CUDA architectures (for example native, 120, or sm_120)" )
  set( _vtm_cuda_architectures "${VTM_CUDA_ARCHITECTURES}" )
  string( REGEX REPLACE "sm_([0-9]+)" "\\1" _vtm_cuda_architectures "${_vtm_cuda_architectures}" )
  set( CMAKE_CUDA_ARCHITECTURES "${_vtm_cuda_architectures}" )
  string( SHA256 _vtm_cuda_build_hash "${CMAKE_BINARY_DIR}" )
  string( SUBSTRING "${_vtm_cuda_build_hash}" 0 8 _vtm_cuda_build_hash )
  set( VTM_CUDA_OUTPUT_SUFFIX "cuda-${_vtm_cuda_architectures}-${_vtm_cuda_build_hash}" CACHE STRING
       "Unique output directory suffix for this CUDA build tree" )
  # Disable nvcc's implicit runtime and link the single explicit CUDA::cudart target below.
  set( CMAKE_CUDA_RUNTIME_LIBRARY None )
  enable_language( CUDA )
  find_package( CUDAToolkit REQUIRED )
endif()

if( VTM_BUILD_CUDA_BACKEND_TESTS )
  include( CTest )
  enable_testing()
endif()

# use ccache
find_program( CCACHE_FOUND ccache )
if( CCACHE_FOUND )
  message( STATUS "ccache found. using it." )
  set_property( GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache )
  set_property( GLOBAL PROPERTY RULE_LAUNCH_LINK ccache )
endif()

# set default CMAKE_BUILD_TYPE to Release if not set
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE )
endif()

if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
  if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
    set( USE_ADDRESS_SANITIZER OFF CACHE BOOL "Compiles with -sanitize=address and links to libasan" )
  endif()
endif()

set( EXTENSION_360_VIDEO OFF CACHE BOOL "If EXTENSION_360_VIDEO is on, 360Lib will be added" )
set( EXTENSION_HDRTOOLS OFF CACHE BOOL "If EXTENSION_HDRTOOLS is on, HDRLib will be added" )

if (DEFINED ENABLE_TRACING)
  set( ENABLE_TRACING OFF CACHE BOOL "ENABLE_TRACING will be set to this value" )
endif()
if (DEFINED ENABLE_HIGH_BITDEPTH)
  set( ENABLE_HIGH_BITDEPTH OFF CACHE BOOL "ENABLE_HIGH_BITDEPTH will be set to this value" )
endif()

set( ENABLE_SEARCH_OPENSSL ON CACHE BOOL "ENABLE_SEARCH_OPENSSL will be set to this value" )

if( CMAKE_COMPILER_IS_GNUCC )
  set( BUILD_STATIC OFF CACHE BOOL "Build static executables" )
endif()

# set c++17
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

# compile everything position independent (even static libraries)
set( CMAKE_POSITION_INDEPENDENT_CODE TRUE )

# set verbose compile options
#set( CMAKE_VERBOSE_MAKEFILE ON )

# use folders in IDEs for projects (e.g. lib sample app test)
set_property( GLOBAL PROPERTY USE_FOLDERS ON )

# Include a utility module providing functions, macros, and settings
include( ${CMAKE_SOURCE_DIR}/cmake/CMakeBuild/cmake/modules/BBuildEnv.cmake )

# BBuildEnv normally makes every build tree share the same source-tree bin/lib
# directories. Keep those legacy locations for CPU builds, but isolate every
# CUDA artifact in a sibling directory so cleaning a CUDA tree can never remove
# files or directories owned by a CPU tree.
if( ENABLE_CUDA )
  foreach( output_kind ARCHIVE LIBRARY RUNTIME PDB COMPILE_PDB )
    foreach( output_config "" DEBUG RELEASE RELWITHDEBINFO MINSIZEREL )
      if( output_config STREQUAL "" )
        set( output_variable "CMAKE_${output_kind}_OUTPUT_DIRECTORY" )
      else()
        set( output_variable "CMAKE_${output_kind}_OUTPUT_DIRECTORY_${output_config}" )
      endif()
      if( DEFINED ${output_variable} AND NOT "${${output_variable}}" STREQUAL "" )
        set( ${output_variable} "${${output_variable}}-${VTM_CUDA_OUTPUT_SUFFIX}" )
      endif()
    endforeach()
  endforeach()
endif()

if( VTM_ENABLE_DECODER_BATCH_PROFILING )
  foreach( output_kind ARCHIVE LIBRARY RUNTIME PDB COMPILE_PDB )
    foreach( output_config "" DEBUG RELEASE RELWITHDEBINFO MINSIZEREL )
      if( output_config STREQUAL "" )
        set( output_variable "CMAKE_${output_kind}_OUTPUT_DIRECTORY" )
      else()
        set( output_variable "CMAKE_${output_kind}_OUTPUT_DIRECTORY_${output_config}" )
      endif()
      if( DEFINED ${output_variable} AND NOT "${${output_variable}}" STREQUAL "" )
        set( ${output_variable} "${${output_variable}}-decoder-batch-profile" )
      endif()
    endforeach()
  endforeach()
endif()

# Enable warnings for some generators and toolsets.
# bb_enable_warnings( gcc warnings-as-errors -Wno-sign-compare )
# bb_enable_warnings( gcc -Wno-unused-variable )
# bb_enable_warnings( gcc-4.8 warnings-as-errors -Wno-unused-variable )
# for gcc 8.2:
bb_enable_warnings( gcc warnings-as-errors -Wno-sign-compare)

if( XCODE )
  bb_enable_warnings( clang warnings-as-errors
                            -Wno-deprecated-declarations
                            -Wno-unknown-attributes
                            -Wno-pessimizing-move
                            -Wno-absolute-value
                            -Wno-unused-const-variable
                            -Wno-unused-command-line-argument )
else()
  bb_enable_warnings( clang warnings-as-errors
                            -Wno-unknown-attributes
                            -Wno-pessimizing-move
                            -Wno-absolute-value
                            -Wno-unused-const-variable )
endif()
#bb_enable_warnings( clang warnings-as-errors )

# enable warnings
bb_enable_warnings( msvc warnings-as-errors "/wd4996" )

# enable sse4.1 build for all source files for gcc and clang
if( (UNIX OR MINGW) AND NOT (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64") )
  add_compile_options( "$<$<COMPILE_LANGUAGE:CXX>:-msse4.1>" )
endif()

# enable parallel build for Visual Studio
if( MSVC )
  add_compile_options( "$<$<COMPILE_LANGUAGE:CXX>:/MP>" )
  add_compile_options( "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>" )
endif()

# set address sanitizer compiler arguments
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
  if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
    if( USE_ADDRESS_SANITIZER )
      # add compile options
      add_compile_options( "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>" )
      add_link_options( "-fsanitize=address" )
    endif()
  endif()

  if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0 )
    add_compile_options( "$<$<COMPILE_LANGUAGE:CXX>:-fabi-version=6>" )
  endif()
endif()

# modify .lldbinit for lldb custom data formatters
if( XCODE )
  set( LLDB_INSTALL_ROOT "$ENV{HOME}/.lldb.d" )
  set( LLDBINIT_FILE "$ENV{HOME}/.lldbinit" )
  set( ENABLE_LLDBINIT_UPDATE ON )
  # add custom target to install LLDB files.
  add_subdirectory( "lldb" )
endif()

# add needed subdirectories
add_subdirectory( "source/Lib/CommonLib" )
add_subdirectory( "source/Lib/CommonAnalyserLib" )
add_subdirectory( "source/Lib/CudaBackend" )
if( EXTENSION_360_VIDEO )
  add_subdirectory( "source/Lib/Lib360" )
  add_subdirectory( "source/Lib/AppEncHelper360" )
endif()
if ( EXTENSION_HDRTOOLS )
  add_subdirectory( "source/Lib/HDRLib")
endif()
add_subdirectory( "source/Lib/DecoderAnalyserLib" )
add_subdirectory( "source/Lib/DecoderLib" )
add_subdirectory( "source/Lib/EncoderLib" )
add_subdirectory( "source/Lib/Utilities" )

add_subdirectory( "source/App/DecoderAnalyserApp" )
add_subdirectory( "source/App/DecoderApp" )
add_subdirectory( "source/App/EncoderApp" )
add_subdirectory( "source/App/SEIRemovalApp" )
add_subdirectory( "source/App/SEIFilmGrainApp" )
add_subdirectory( "source/App/Parcat" )
add_subdirectory( "source/App/StreamMergeApp" )
add_subdirectory( "source/App/BitstreamExtractorApp" )
add_subdirectory( "source/App/SubpicMergeApp" )
if( EXTENSION_360_VIDEO )
  add_subdirectory( "source/App/utils/360ConvertApp" )
endif()
