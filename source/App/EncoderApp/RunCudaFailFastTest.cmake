if( NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED TEST_KIND OR NOT DEFINED TEST_INPUT
    OR NOT DEFINED TEST_OUTPUT OR NOT DEFINED TEST_SOURCE_DIR )
  message( FATAL_ERROR "CUDA EncoderApp fail-fast test arguments are incomplete" )
endif()

if( TEST_KIND STREQUAL "SAD" )
  set( ENV{VTM_CUDA_SAD_TEST_FAILURE} kernel )
  set( TEST_CONFIG "${TEST_SOURCE_DIR}/cfg/encoder_lowdelay_P_vtm.cfg" )
  set( TEST_EXPECTED "CUDA SAD execution failed after selection" )
  set( TEST_ARGUMENTS
    --SourceWidth=64 --SourceHeight=64 --FrameRate=1 --FramesToBeEncoded=2
    --InputBitDepth=8 --InputChromaFormat=420 --GPUBackend=cuda --GPUExperimentalSAD=1
    --FastSearch=0 --SearchRange=64 --QP=32
  )
elseif( TEST_KIND STREQUAL "QPA" )
  set( ENV{VTM_CUDA_QPA_TEST_FAILURE} kernel )
  set( TEST_CONFIG "${TEST_SOURCE_DIR}/cfg/encoder_intra_vtm.cfg" )
  set( TEST_EXPECTED "CUDA QPA execution failed after selection" )
  set( TEST_ARGUMENTS
    --SourceWidth=512 --SourceHeight=512 --FrameRate=1 --FramesToBeEncoded=1
    --InputBitDepth=8 --InputChromaFormat=420 --GPUBackend=cuda --GPUExperimentalQPA=1
    --PerceptQPA=1 --SliceChromaQPOffsetPeriodicity=1 --QP=32
  )
else()
  message( FATAL_ERROR "Unknown CUDA EncoderApp fail-fast test kind: ${TEST_KIND}" )
endif()

execute_process(
  COMMAND "${TEST_EXECUTABLE}" -c "${TEST_CONFIG}" -i "${TEST_INPUT}" -b "${TEST_OUTPUT}"
          ${TEST_ARGUMENTS}
  RESULT_VARIABLE TEST_RESULT
  OUTPUT_VARIABLE TEST_STDOUT
  ERROR_VARIABLE TEST_STDERR
)
set( TEST_OUTPUT_TEXT "${TEST_STDOUT}\n${TEST_STDERR}" )

if( NOT TEST_RESULT EQUAL 1 )
  message( FATAL_ERROR
    "EncoderApp ${TEST_KIND} returned ${TEST_RESULT}; expected exactly 1 after injected CUDA failure:\n${TEST_OUTPUT_TEXT}" )
endif()
if( NOT TEST_OUTPUT_TEXT MATCHES "${TEST_EXPECTED}" )
  message( FATAL_ERROR "EncoderApp ${TEST_KIND} failed without the required CUDA diagnostic:\n${TEST_OUTPUT_TEXT}" )
endif()

message( STATUS "EncoderApp ${TEST_KIND} fail-fast diagnostic and exit code 1 verified" )
