cmake_minimum_required(VERSION 3.21.3 FATAL_ERROR)

project(Solve)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

add_executable(Solve
               src/main.cpp
               src/int_funcs.cpp
               src/int_funcs.h)

set(CMAKE_CXX_FLAGS "-O2")
set(CMAKE_C_FLAGS "-O2")
set(CMAKE_EXE_LINKER_FLAGS "-O2")

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
target_link_libraries(Solve MPI::MPI_CXX)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(Solve Eigen3::Eigen)
