set_source_path(THIS_DIR_HEADERS
  allgather.hpp
  allreduce.hpp
  alltoall.hpp
  bcast.hpp
  communicator.hpp
  gather.hpp
  reduce.hpp
  reduce_scatter.hpp
  scatter.hpp
  pt2pt.hpp
  util.hpp
  )

set_source_path(THIS_DIR_CXX_SOURCES
  communicator.cpp
  )

set_source_path(THIS_DIR_CUDA_RMA_HEADERS
  rma.hpp
  rma_ipc.hpp
  rma_null.hpp
  rma_self.hpp
  )

set_source_path(THIS_DIR_CUDA_RMA_CXX_SOURCES
  rma.cpp
  )

if (AL_HAS_MPI_CUDA_RMA)
  list(APPEND THIS_DIR_HEADERS "${THIS_DIR_CUDA_RMA_HEADERS}")
  list(APPEND THIS_DIR_CXX_SOURCES "${THIS_DIR_CUDA_RMA_CXX_SOURCES}")
endif ()

if (AL_HAS_ROCM)
  hipify_header_files(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})
else ()
  set(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})
endif ()

# Custom install
install(FILES ${THIS_DIR_HEADERS_TO_INSTALL}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum/mpi_cuda)

# Propagate the files up the tree
set(CXX_SOURCES ${CXX_SOURCES} ${THIS_DIR_CXX_SOURCES} PARENT_SCOPE)
set(CUDA_SOURCES ${CUDA_SOURCES} ${THIS_DIR_CUDA_SOURCES} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
