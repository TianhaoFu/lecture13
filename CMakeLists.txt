cmake_minimum_required(VERSION 3.2)
project(needle C CXX)


include_directories("include")

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(pybind11 PATHS ${__pybind_path})


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++14 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()


# Source file lists
file(GLOB_RECURSE SRCS
  src/*.cc
  )

file(GLOB_RECURSE CUDA_SRCS
  src/*.cu
  )

file(GLOB_RECURSE PYBIND_SRCS
  python/pybind/*.cc
  )


set(LINER_LIBS "")
find_package(CUDA)

if(CUDA_FOUND)
  message(STATUS "Find cuda, build with cuda support")
  set(BUILD_WITH_CUDA ON)
else()
  set(BUILD_WITH_CUDA OFF)
endif()


include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


if(BUILD_WITH_CUDA)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUBLAS_LIBRARY})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  # set arch flags properly
  CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  CUDA_ADD_LIBRARY(main MODULE ${SRCS} ${CUDA_SRCS} ${PYBIND_SRCS} OPTIONS ${ARCH_FLAGS})
else()
  add_library(main MODULE ${SRCS} ${PYBIND_SRCS})
  target_compile_definitions(main PRIVATE -DNEEDLE_USE_CUDA=0)
endif()

target_link_libraries(main ${LINKER_LIBS})
target_compile_definitions(main PRIVATE -DDLPACK_EXPORTS)

pybind11_extension(main)
pybind11_strip(main)

# directly output to ffi folder
set_target_properties(main
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/_ffi/
  CXX_VISIBILITY_PRESET "hidden"
  CUDA_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET main PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()
