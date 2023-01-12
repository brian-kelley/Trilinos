#!/bin/bash
set -o errexit
set -o pipefail

COMPILER_DIR=${COMPILER_ROOT}
MPI_DIR=${MPI_ROOT}
HDF5_DIR=${HDF5_ROOT}
NETCDF_DIR=${NETCDF_ROOT}
PNETCDF_DIR=${PNETCDF_ROOT}
ZLIB_DIR=/usr/lib64
CGNS_DIR=${CGNS_ROOT}
BOOST_DIR=${BOOST_ROOT}
METIS_DIR=${METIS_ROOT}
PARMETIS_DIR=${PARMETIS_ROOT}
SUPERLUDIST_DIR=${SUPERLUDIST_ROOT}

TRILINOS_HOME=${TRILINOS_REPO_DIR:-$(cd ..; pwd)}
TRIL_INSTALL_PATH=${TRIL_INSTALL_PATH:-$(cd ..; pwd)}

DEFAULT_VARIANT=opt
DEFAULT_LINKTYPE=static
DEFAULT_EXECUTIONSPACE=openmp
DEFAULT_PACKAGE=full
DEFAULT_USE_MPI=mpi
source $(dirname $(readlink -f ${0}))/config_parser.sh
extra_cmake_args=$@
extra_link_flags="-L${MPI_DIR}/lib -lmpich"
exe_linker_flags=""

if [[ "${VARIANT:?}" == "opt" ]]
then
  BUILD_TYPE=RELEASE
  BUILD_SUFFIX=opt
  BUILD_C_FLAGS=" -qoverride-limits"
  BUILD_CXX_FLAGS=" -qoverride-limits"
  BUILD_F_FLAGS=" -qoverride-limits"
  BOUNDS_CHECKING=OFF
elif [[ "${VARIANT:?}" == "dbg" ]]
then
  BUILD_TYPE=DEBUG
  BUILD_SUFFIX=dbg
  BUILD_C_FLAGS=""
  BUILD_CXX_FLAGS=""
  BUILD_F_FLAGS=""
  BOUNDS_CHECKING=ON
else
  echo "ERROR: Invalid variant '${VARIANT:?}'!" >&2
  exit 1
fi

if [[ "${LINKTYPE:?}" == "static" ]]
then
  LINK_STATIC=ON
  LINK_SHARED=OFF
elif [[ "${LINKTYPE:?}" == "shared" ]]
then
  LINK_STATIC=OFF
  LINK_SHARED=ON
else
  echo "ERROR: Invalid link type '${LINKTYPE:?}'!" >&2
  exit 1
fi

if [[ "${EXECUTIONSPACE:?}" == "serial" ]]
then
  USING_SERIAL=ON
  USING_OPENMP=OFF
elif [[ "${EXECUTIONSPACE:?}" == "openmp" ]]
then
  USING_SERIAL=OFF
  USING_OPENMP=ON
else
  echo "ERROR: Invalid execution space '${EXECUTIONSPACE:?}'!" >&2
  exit 1
fi

if [[ "${PACKAGE:?}" == "full" ]]
then
  BUILD_ALL_PACKAGES=ON
  BUILD_SEACAS=ON
  BUILD_KOKKOS=ON
elif [[ "${PACKAGE:?}" == "mini" ]]
then
  BUILD_ALL_PACKAGES=OFF
  BUILD_SEACAS=ON
  BUILD_KOKKOS=ON
elif [[ "${PACKAGE:?}" == "seacas" ]]
then
  BUILD_ALL_PACKAGES=OFF
  BUILD_SEACAS=ON
  BUILD_KOKKOS=OFF
else
  echo "ERROR: Invalid package '${PACKAGE:?}'!" >&2
  exit 1
fi

if [[ "${USE_MPI:?}" == "mpi" ]]
then
    TRIL_INSTALL_DIR=${SPARC_ARCH}_${SPARC_COMPILER}_${SPARC_MPI}_${EXECUTIONSPACE:?}_${LINKTYPE:?}_${VARIANT:?}
else
  echo "ERROR: Invalid using MPI flag '${USE_MPI:?}'!" >&2
  exit 1
fi

maybescript=${TRILINOS_HOME}/../sparc-devops/linalg.py
if [[ -f ${maybescript} ]]
then
    includes=$(${maybescript} --include-dirs)
    libdirs=$(${maybescript} --lib-dirs)
    libnames=$(${maybescript} --library-names)
else
    echo "REMARK: 'sparc-devops' not cloned as a part of this checkout; using hard-coded MKL paths"
    CBLAS=${CBLAS_ROOT}
    includes="${CBLAS}/mkl/include"
    libdirs="${CBLAS}/mkl/lib/intel64;${CBLAS}/compiler/lib/intel64"
    libnames="mkl_intel_lp64;mkl_intel_thread;mkl_core;iomp5"
fi

echo " *** Installing in: ${TRIL_INSTALL_PATH}/${TRIL_INSTALL_DIR}"
sleep 3

rm -f CMakeCache.txt; rm -rf CMakeFiles

cmake \
   -D CMAKE_VERBOSE_MAKEFILE=OFF \
   -D CMAKE_INSTALL_PREFIX:PATH="${TRIL_INSTALL_PATH}/${TRIL_INSTALL_DIR}" \
   -D CMAKE_BUILD_TYPE:STRING="${BUILD_TYPE}" \
   -D BUILD_SHARED_LIBS=${LINK_SHARED} \
   \
   -D CMAKE_C_COMPILER="cc" \
   -D CMAKE_CXX_COMPILER="CC" \
   -D CMAKE_Fortran_COMPILER="ftn" \
   \
   -D CMAKE_CXX_STANDARD=17 \
   \
   -D CMAKE_C_FLAGS="${BUILD_C_FLAGS}" \
   -D CMAKE_CXX_FLAGS="${BUILD_CXX_FLAGS}" \
   -D CMAKE_Fortran_FLAGS="${BUILD_F_FLAGS}" \
   -D CMAKE_EXE_LINKER_FLAGS="${exe_linker_flags}" \
   -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
   \
   -D TPL_FIND_SHARED_LIBS=${LINK_SHARED:?} \
   \
   -D Trilinos_VERBOSE_CONFIGURE=FALSE \
   -D Trilinos_ENABLE_ALL_PACKAGES=OFF \
   -D Trilinos_ENABLE_SECONDARY_TESTED_CODE=OFF \
   \
   -D Trilinos_ENABLE_TESTS=OFF \
   -D DART_TESTING_TIMEOUT:STRING="600" \
   \
   -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
   -D Tpetra_INST_FLOAT=OFF \
   -D Tpetra_INST_DOUBLE=ON \
   -D Tpetra_INST_COMPLEX_FLOAT=OFF \
   -D Tpetra_INST_COMPLEX_DOUBLE=OFF \
   -D Tpetra_INST_INT_INT=OFF \
   -D Tpetra_INST_INT_LONG=OFF \
   -D Tpetra_INST_INT_UNSIGNED=OFF \
   -D Tpetra_INST_INT_LONG_LONG=ON \
   -D Teuchos_ENABLE_COMPLEX=OFF \
   -D Zoltan_ENABLE_ULLONG_IDS=ON \
   \
   -D Trilinos_ENABLE_OpenMP=${USING_OPENMP:?} \
   -D TPL_ENABLE_Pthread=OFF \
   \
   -D Trilinos_ENABLE_Teuchos=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_Epetra=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_EpetraExt=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_AztecOO=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_Amesos=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_Stratimikos=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_Anasazi=${BUILD_ALL_PACKAGES:?} \
   -D Anasazi_ENABLE_RBGen=OFF \
   -D Trilinos_ENABLE_Ifpack=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_ML=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_Teko=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_NOX=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_Thyra=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_Rythmos=OFF \
   -D Trilinos_ENABLE_Sacado=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_Stokhos=OFF \
   -D Trilinos_ENABLE_Panzer=OFF \
   -D Trilinos_ENABLE_Tpetra=${BUILD_ALL_PACKAGES:?} \
   -D Tpetra_INST_SERIAL=${USING_SERIAL:?} \
   -D Tpetra_INST_OPENMP=${USING_OPENMP:?} \
   -D Trilinos_ENABLE_Belos=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_Amesos2=${BUILD_ALL_PACKAGES:?} \
   -D Amesos2_ENABLE_Epetra=OFF \
   -D Amesos2_ENABLE_KLU2=ON \
   -D Trilinos_ENABLE_Ifpack2=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_MueLu=${BUILD_ALL_PACKAGES:?} \
   -D MueLu_ENABLE_Epetra=OFF \
   -D Xpetra_ENABLE_Epetra=OFF \
   -D Xpetra_ENABLE_EpetraExt=OFF \
   -D Trilinos_ENABLE_Zoltan2=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_STKMesh=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_STKIO=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_STKTransfer=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_STKSearch=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_STKUtil=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_STKTopology=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_STKSimd=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_STKCoupling=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_Pamgen=OFF \
   -D Trilinos_ENABLE_Intrepid2=OFF \
   -D Trilinos_ENABLE_ShyLU_NodeHTS=${BUILD_ALL_PACKAGES:?} \
   -D Trilinos_ENABLE_ShyLU_NodeTacho=OFF \
   -D Trilinos_ENABLE_Kokkos=${BUILD_KOKKOS:?} \
   -D Trilinos_ENABLE_KokkosCore=${BUILD_KOKKOS:?} \
   -D Kokkos_ENABLE_SERIAL=${USING_SERIAL:?} \
   -D Kokkos_ENABLE_OPENMP=${USING_OPENMP:?} \
   -D Kokkos_ENABLE_PTHREAD=OFF \
   -D TPL_ENABLE_CUDA=OFF \
   -D Kokkos_ENABLE_CUDA=OFF \
   -D Kokkos_ENABLE_CUDA_UVM=OFF \
   -D Kokkos_ARCH_HSW=ON \
   \
   -D Kokkos_ENABLE_DEPRECATED_CODE_3=OFF \
   -D Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=${BOUNDS_CHECKING} \
   -D Tpetra_ENABLE_DEPRECATED_CODE=OFF  \
   -D Belos_HIDE_DEPRECATED_CODE=ON  \
   -D Epetra_HIDE_DEPRECATED_CODE=ON  \
   -D Ifpack2_HIDE_DEPRECATED_CODE=ON \
   -D Ifpack2_ENABLE_DEPRECATED_CODE=OFF \
   -D MueLu_ENABLE_DEPRECATED_CODE=OFF \
   -D STK_HIDE_DEPRECATED_CODE=ON \
   -D Teuchos_HIDE_DEPRECATED_CODE=ON\
   -D Thyra_HIDE_DEPRECATED_CODE=ON \
   \
   -D Trilinos_ENABLE_SEACAS=${BUILD_SEACAS:?} \
   -D TPL_ENABLE_X11=OFF \
   -D TPL_ENABLE_Matio=OFF \
   \
   -D Trilinos_ENABLE_Gtest=OFF \
   \
   -D Trilinos_ENABLE_TriKota=OFF \
   -D DAKOTA_ENABLE_TESTS=OFF \
   -D Trilinos_ENABLE_ROL=${BUILD_ALL_PACKAGES:?} \
   \
   -D TPL_ENABLE_MPI=ON \
   -D MPI_USE_COMPILER_WRAPPERS=OFF \
   -D MPI_BASE_DIR:PATH="${MPI_DIR}" \
   -D MPI_EXEC:PATH="mpirun" \
   -D MPI_EXEC_MAX_NUMPROCS:STRING="8" \
   -D MPI_EXEC_NUMPROCS_FLAG:STRING="-np" \
   \
   -D TPL_ENABLE_BinUtils=ON \
   \
   -D TPL_ENABLE_BLAS=ON \
   -D BLAS_INCLUDE_DIRS:PATH="${includes}" \
   -D BLAS_LIBRARY_DIRS:PATH="${libdirs}" \
   -D BLAS_LIBRARY_NAMES:STRING="${libnames}" \
   \
   -D TPL_ENABLE_LAPACK=ON \
   -D LAPACK_INCLUDE_DIRS:PATH="${includes}" \
   -D LAPACK_LIBRARY_DIRS:PATH="${libdirs}" \
   -D LAPACK_LIBRARY_NAMES:STRING="${libnames}" \
   \
   -D TPL_ENABLE_Boost=${BUILD_ALL_PACKAGES:?} \
   -D Boost_INCLUDE_DIRS:PATH="${BOOST_DIR}/include" \
   \
   -D TPL_ENABLE_BoostLib=${BUILD_ALL_PACKAGES:?} \
   -D BoostLib_INCLUDE_DIRS:PATH="${BOOST_DIR}/include" \
   -D BoostLib_LIBRARY_DIRS:PATH="${BOOST_DIR}/lib" \
   \
   -D HDF5_ROOT:PATH="${HDF5_DIR}" \
   -D HDF5_NO_SYSTEM_PATHS=ON \
   \
   -D PNetCDF_ROOT:PATH="${PNETCDF_DIR}" \
   \
   -D TPL_ENABLE_Netcdf=${BUILD_SEACAS:?} \
   -D NetCDF_ROOT:PATH="${NETCDF_DIR}" \
   \
   -D TPL_ENABLE_CGNS=${BUILD_SEACAS:?} \
   -D CGNS_INCLUDE_DIRS:PATH="${CGNS_DIR}/include" \
   -D CGNS_LIBRARY_DIRS:PATH="${CGNS_DIR}/lib" \
   -D CGNS_LIBRARY_NAMES:STRING="cgns" \
   \
   -D TPL_ENABLE_METIS=${BUILD_ALL_PACKAGES:?} \
   -D METIS_INCLUDE_DIRS:PATH="${METIS_DIR}/include" \
   -D METIS_LIBRARY_DIRS:PATH="${METIS_DIR}/lib" \
   \
   -D TPL_ENABLE_ParMETIS=${BUILD_ALL_PACKAGES:?} \
   -D ParMETIS_INCLUDE_DIRS:PATH="${PARMETIS_DIR}/include" \
   -D ParMETIS_LIBRARY_DIRS:PATH="${PARMETIS_DIR}/lib" \
   \
   -D TPL_ENABLE_SuperLUDist=${BUILD_ALL_PACKAGES:?} \
   -D SuperLUDist_INCLUDE_DIRS:PATH="${SUPERLUDIST_DIR}/include" \
   -D SuperLUDist_LIBRARY_DIRS:PATH="${SUPERLUDIST_DIR}/lib" \
   -D SuperLUDist_LIBRARY_NAMES:STRING="superlu_dist" \
   \
   -D Trilinos_EXTRA_LINK_FLAGS:STRING="${extra_link_flags}" \
   \
   ${extra_cmake_args} \
   ${TRILINOS_HOME}
