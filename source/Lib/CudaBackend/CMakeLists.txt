set( LIB_NAME CudaBackend )

add_library( ${LIB_NAME} STATIC
  ComputeBackend.cpp
  ComputeBackend.h
  CudaDistortion.h
  CudaContext.cpp
  CudaContext.h
  CudaPictureMirror.h
)

target_compile_features( ${LIB_NAME} PUBLIC cxx_std_17 )
target_include_directories( ${LIB_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/.." )

if( ENABLE_CUDA )
  target_sources( ${LIB_NAME} PRIVATE CudaRuntime.cu CudaRuntime.h )
  target_compile_definitions( ${LIB_NAME} PRIVATE VTM_ENABLE_CUDA=1 )
  target_link_libraries( ${LIB_NAME} PRIVATE CUDA::cudart )
  set_target_properties( ${LIB_NAME} PROPERTIES
    CUDA_ARCHITECTURES "${_vtm_cuda_architectures}"
    CUDA_RUNTIME_LIBRARY None
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )
else()
  target_compile_definitions( ${LIB_NAME} PRIVATE VTM_ENABLE_CUDA=0 )
endif()

set_target_properties( ${LIB_NAME} PROPERTIES FOLDER lib )

if( VTM_BUILD_CUDA_BACKEND_TESTS )
  target_compile_definitions( ${LIB_NAME} PRIVATE VTM_CUDA_TESTING=1 )
  add_executable( CudaBackendTest tests/CudaBackendTest.cpp )
  target_compile_definitions( CudaBackendTest PRIVATE VTM_CUDA_TESTING=1 )
  target_link_libraries( CudaBackendTest PRIVATE ${LIB_NAME} CommonLib )
  set_target_properties( CudaBackendTest PROPERTIES FOLDER test )

  add_test( NAME CudaBackend.ParseConfig COMMAND CudaBackendTest )
  if( ENABLE_CUDA )
    add_test( NAME CudaBackend.Runtime COMMAND CudaBackendTest --cuda 0 )
    set_tests_properties( CudaBackend.Runtime PROPERTIES LABELS gpu )
  else()
    add_test( NAME CudaBackend.Disabled COMMAND CudaBackendTest --disabled )
  endif()
endif()
