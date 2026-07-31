set( LIB_NAME CudaBackend )

add_library( ${LIB_NAME} STATIC
  ComputeBackend.cpp
  ComputeBackend.h
  CudaContext.cpp
  CudaContext.h
)

target_compile_features( ${LIB_NAME} PUBLIC cxx_std_17 )
target_include_directories( ${LIB_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/.." )

if( ENABLE_CUDA )
  target_sources( ${LIB_NAME} PRIVATE CudaRuntime.cu CudaRuntime.h )
  target_compile_definitions( ${LIB_NAME} PRIVATE VTM_ENABLE_CUDA=1 )
  target_link_libraries( ${LIB_NAME} PRIVATE CUDA::cudart )
  set_target_properties( ${LIB_NAME} PROPERTIES
    CUDA_ARCHITECTURES "${_vtm_cuda_architectures}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )
endif()

set_target_properties( ${LIB_NAME} PROPERTIES FOLDER lib )
