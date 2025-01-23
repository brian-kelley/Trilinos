TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)
TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

TRIBITS_ADD_EXECUTABLE(
  ex1
  SOURCES ex1.cpp
)

IF (Tpetra_INST_DOUBLE)
TRIBITS_ADD_EXECUTABLE(
  RelaxationWithEquilibration
  SOURCES RelaxationWithEquilibration.cpp
)
ENDIF()


TRIBITS_ADD_EXECUTABLE(
  BlockTriDiagonalSolver
  SOURCES BlockTriDi.cpp
  CATEGORIES BASIC PERFORMANCE
)


TRIBITS_ADD_TEST(
  BlockTriDiagonalSolver
  NAME BlockTriDiExample
  ARGS ""
  COMM serial mpi
  NUM_MPI_PROCS 1-4
  STANDARD_PASS_OUTPUT
)

ASSERT_DEFINED (
  ${PACKAGE_NAME}_ENABLE_Xpetra
  ${PACKAGE_NAME}_ENABLE_Galeri
)

IF(${PACKAGE_NAME}_ENABLE_Xpetra AND ${PACKAGE_NAME}_ENABLE_Galeri)

  set(nx 128)
  set(ny 128)
  set(nz 64)

  set(TEST_NAME "BTDS_Jacobi")
  #set(TEST_ARGS ${BTDS_ARGS} ${BTDS_SCHUR_ARGS} ${BTDS_DEFAULT_SCHUR_ARGS} ${BLOCK_JACOBI_ARGS})
  #set(TEST_NAMES "BTDS" "BTDS_Schur" "BTDS_Schur_Default" "BTDS_Jacobi")

  ASSERT_DEFINED(
    Tpetra_INST_CUDA
    Tpetra_INST_HIP
  )

  set(BLOCK_SIZES "7" "11")

  foreach(blockSize ${BLOCK_SIZES})
    set(COMMON_ARGS "--withoutOverlapCommAndComp --matrixType=Laplace3D --blockSize=${blockSize} --nx=${nx} --ny=${ny} --nz=${nz} --tol=0 --numIters=5 --withStackedTimer --numRepeats=5")
    set(BTDS_ARGS "${COMMON_ARGS} --sublinesPerLine=1 --sublinesPerLineSchur=1")
    set(BTDS_SCHUR_ARGS "${COMMON_ARGS} --sublinesPerLine=1 --sublinesPerLineSchur=2")
    set(BTDS_DEFAULT_SCHUR_ARGS "${COMMON_ARGS} --sublinesPerLine=1 --sublinesPerLineSchur=-1")
    set(BLOCK_JACOBI_ARGS "${COMMON_ARGS} --sublinesPerLine=-1 --quiet")

    set(TEST_ARGS ${BLOCK_JACOBI_ARGS})
    set(ARCH "HIP")
    IF (Tpetra_INST_${ARCH})
      MESSAGE(STATUS "Ifpack2: BlockTriDiagonalSolver ${ARCH} test ENABLED")
      TRIBITS_ADD_TEST(
        BlockTriDiagonalSolver
        NAME "${TEST_NAME}_${ARCH}_BS_${blockSize}"
        COMM mpi
        ARGS "${TEST_ARGS} --problemName=\"${TEST_NAME} ${ARCH} ${blockSize}x${nx}x${ny}x${nz} 1 ranks\""
        NUM_MPI_PROCS 1
        RUN_SERIAL
        CATEGORIES PERFORMANCE
      )
    ENDIF()
  endforeach()
ENDIF()
