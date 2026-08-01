if( NOT DEFINED DECODER_EXECUTABLE OR NOT EXISTS "${DECODER_EXECUTABLE}" )
  message( FATAL_ERROR "Decoder executable is missing: ${DECODER_EXECUTABLE}" )
endif()
if( NOT DEFINED DECODER_MANIFEST )
  message( FATAL_ERROR "DECODER_MANIFEST is required" )
endif()
if( NOT DECODER_PROFILE_MODE STREQUAL "ON" AND NOT DECODER_PROFILE_MODE STREQUAL "OFF" )
  message( FATAL_ERROR "Invalid decoder profiling mode: ${DECODER_PROFILE_MODE}" )
endif()
if( NOT DEFINED DECODER_CMAKE_CACHE OR NOT EXISTS "${DECODER_CMAKE_CACHE}" )
  message( FATAL_ERROR "Decoder CMake cache is missing: ${DECODER_CMAKE_CACHE}" )
endif()

file( SHA256 "${DECODER_EXECUTABLE}" DECODER_EXECUTABLE_SHA256 )
file( SHA256 "${DECODER_CMAKE_CACHE}" DECODER_CMAKE_CACHE_SHA256 )
get_filename_component( DECODER_EXECUTABLE_NAME "${DECODER_EXECUTABLE}" NAME )

file( WRITE "${DECODER_MANIFEST}"
  "{\n"
  "  \"schema\": 1,\n"
  "  \"artifact\": \"${DECODER_EXECUTABLE_NAME}\",\n"
  "  \"executable_sha256\": \"${DECODER_EXECUTABLE_SHA256}\",\n"
  "  \"decoder_batch_profiling\": \"${DECODER_PROFILE_MODE}\",\n"
  "  \"configuration\": \"${DECODER_CONFIGURATION}\",\n"
  "  \"cmake_cache_sha256\": \"${DECODER_CMAKE_CACHE_SHA256}\"\n"
  "}\n"
)
