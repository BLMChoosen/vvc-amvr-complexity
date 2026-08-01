if( NOT DEFINED TEST_MODE OR NOT DEFINED TEST_WORK_DIR )
  message( FATAL_ERROR "CUDA loop-filter chain process test is missing TEST_MODE or TEST_WORK_DIR" )
endif()

file( MAKE_DIRECTORY "${TEST_WORK_DIR}" )
set( TEST_INPUT "${TEST_WORK_DIR}/input-1920x1080-8bit.yuv" )
set( TEST_BITSTREAM "${TEST_WORK_DIR}/loop-filter-chain-ai-8bit.vvc" )

if( TEST_MODE STREQUAL "SETUP" )
  if( NOT DEFINED TEST_GENERATOR OR NOT DEFINED TEST_ENCODER OR NOT DEFINED TEST_SOURCE_DIR )
    message( FATAL_ERROR "CUDA loop-filter chain fixture setup is missing an executable or source directory" )
  endif()
  set( TEST_ENCODER_RECON "${TEST_WORK_DIR}/encoder-reconstruction.yuv" )
  file( REMOVE "${TEST_INPUT}" "${TEST_BITSTREAM}" "${TEST_ENCODER_RECON}" )
  execute_process(
    COMMAND "${TEST_GENERATOR}" --write-alf-yuv "${TEST_INPUT}"
    RESULT_VARIABLE GENERATOR_RESULT OUTPUT_VARIABLE GENERATOR_STDOUT ERROR_VARIABLE GENERATOR_STDERR
  )
  if( NOT GENERATOR_RESULT EQUAL 0 )
    message( FATAL_ERROR "CUDA loop-filter chain input generation failed (${GENERATOR_RESULT}):\n${GENERATOR_STDOUT}\n${GENERATOR_STDERR}" )
  endif()
  file( SIZE "${TEST_INPUT}" TEST_INPUT_SIZE )
  if( NOT TEST_INPUT_SIZE EQUAL 3110400 )
    message( FATAL_ERROR "CUDA loop-filter chain input has ${TEST_INPUT_SIZE} bytes; expected 3110400" )
  endif()
  execute_process(
    COMMAND "${TEST_ENCODER}"
      -c "${TEST_SOURCE_DIR}/cfg/encoder_intra_vtm.cfg"
      -i "${TEST_INPUT}" -b "${TEST_BITSTREAM}" -o "${TEST_ENCODER_RECON}"
      --SourceWidth=1920 --SourceHeight=1080 --FrameRate=1 --FramesToBeEncoded=1
      --InputBitDepth=8 --InternalBitDepth=8 --InputChromaFormat=420 --QP=42
      --CTUSize=128 --MaxCUWidth=128 --MaxCUHeight=128 --DualITree=0
      --TransformSkip=0 --MTS=0 --LFNST=0 --ISP=0 --MIP=0 --MRL=0
      --ALF=1 --CCALF=0 --SAO=1 --LMCSEnable=1
      --DepQuant=0 --RDOQ=0 --RDOQTS=0 --SEIDecodedPictureHash=1 --Verbosity=0
    WORKING_DIRECTORY "${TEST_SOURCE_DIR}"
    RESULT_VARIABLE ENCODER_RESULT OUTPUT_VARIABLE ENCODER_STDOUT ERROR_VARIABLE ENCODER_STDERR
  )
  file( REMOVE "${TEST_ENCODER_RECON}" )
  if( NOT ENCODER_RESULT EQUAL 0 OR NOT EXISTS "${TEST_BITSTREAM}" )
    message( FATAL_ERROR "CUDA loop-filter chain fixture encode failed (${ENCODER_RESULT}):\n${ENCODER_STDOUT}\n${ENCODER_STDERR}" )
  endif()
  return()
endif()

if( NOT DEFINED TEST_DECODER OR NOT EXISTS "${TEST_BITSTREAM}" )
  message( FATAL_ERROR "CUDA loop-filter chain process test requires the decoder and prepared fixture bitstream" )
endif()

if( TEST_MODE STREQUAL "PARITY" )
  set( TEST_CPU_YUV "${TEST_WORK_DIR}/decoded-cpu.yuv" )
  set( TEST_CUDA_YUV "${TEST_WORK_DIR}/decoded-cuda.yuv" )
  file( REMOVE "${TEST_CPU_YUV}" "${TEST_CUDA_YUV}" )
  execute_process(
    COMMAND "${TEST_DECODER}" -b "${TEST_BITSTREAM}" -o "${TEST_CPU_YUV}"
      --GPUBackend=cpu --SEIDecodedPictureHash=1
    RESULT_VARIABLE CPU_RESULT OUTPUT_VARIABLE CPU_STDOUT ERROR_VARIABLE CPU_STDERR
  )
  execute_process(
    COMMAND "${TEST_DECODER}" -b "${TEST_BITSTREAM}" -o "${TEST_CUDA_YUV}"
      --GPUBackend=cuda --GPUDevice=0 --GPUExperimentalLoopFilterChain=1 --SEIDecodedPictureHash=1
    RESULT_VARIABLE CUDA_RESULT OUTPUT_VARIABLE CUDA_STDOUT ERROR_VARIABLE CUDA_STDERR
  )
  if( NOT CPU_RESULT EQUAL 0 OR NOT CUDA_RESULT EQUAL 0 )
    message( FATAL_ERROR "CUDA loop-filter chain parity decode failed (CPU=${CPU_RESULT}, CUDA=${CUDA_RESULT}):\n${CPU_STDOUT}\n${CPU_STDERR}\n${CUDA_STDOUT}\n${CUDA_STDERR}" )
  endif()
  file( SHA256 "${TEST_CPU_YUV}" CPU_SHA256 )
  file( SHA256 "${TEST_CUDA_YUV}" CUDA_SHA256 )
  if( NOT CPU_SHA256 STREQUAL CUDA_SHA256 )
    message( FATAL_ERROR "CUDA loop-filter chain output differs: CPU=${CPU_SHA256}, CUDA=${CUDA_SHA256}" )
  endif()
  string( REGEX MATCH "CUDA loop-filter chain frames/pixels/DBF tasks/SAO CTUs/ALF CTUs: 1/2073600/[1-9][0-9]*/[1-9][0-9]*/[1-9][0-9]*" CHAIN_STATS "${CUDA_STDOUT}${CUDA_STDERR}" )
  if( CHAIN_STATS STREQUAL "" )
    message( FATAL_ERROR "CUDA loop-filter chain collector/runtime was not exercised:\n${CUDA_STDOUT}\n${CUDA_STDERR}" )
  endif()
  return()
endif()

if( TEST_MODE STREQUAL "FAILFAST" )
  set( TEST_FAILURE_YUV "${TEST_WORK_DIR}/decoded-failure.yuv" )
  file( REMOVE "${TEST_FAILURE_YUV}" )
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env VTM_CUDA_LOOP_FILTER_CHAIN_TEST_FAILURE=sao-launch
      "${TEST_DECODER}" -b "${TEST_BITSTREAM}" -o "${TEST_FAILURE_YUV}"
      --GPUBackend=cuda --GPUDevice=0 --GPUExperimentalLoopFilterChain=1 --SEIDecodedPictureHash=1
    RESULT_VARIABLE FAILURE_RESULT OUTPUT_VARIABLE FAILURE_STDOUT ERROR_VARIABLE FAILURE_STDERR
  )
  if( NOT FAILURE_RESULT EQUAL 1 )
    message( FATAL_ERROR "CUDA loop-filter chain fatal runtime failure returned ${FAILURE_RESULT}; expected process exit 1" )
  endif()
  string( FIND "${FAILURE_STDOUT}${FAILURE_STDERR}" "CUDA loop-filter chain execution failed after selection:" FAILURE_PREFIX )
  string( FIND "${FAILURE_STDOUT}${FAILURE_STDERR}" "Injected CUDA loop-filter chain SAO launch failure" FAILURE_DETAIL )
  if( FAILURE_PREFIX EQUAL -1 OR FAILURE_DETAIL EQUAL -1 )
    message( FATAL_ERROR "CUDA loop-filter chain fatal error was not reported clearly:\n${FAILURE_STDOUT}\n${FAILURE_STDERR}" )
  endif()
  return()
endif()

message( FATAL_ERROR "Unknown CUDA loop-filter chain process test mode: ${TEST_MODE}" )
