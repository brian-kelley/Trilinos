/*
// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER
*/
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_TestingUtilities.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include "TpetraExt_MatrixMatrix.hpp"
#include "Tpetra_MatrixIO.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_CrsMatrixMultiplyOp.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Export.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "MatrixMarket_Tpetra.hpp"
#include "Tpetra_RowMatrixTransposer.hpp"
#include "impl/Kokkos_Timer.hpp"
#include "TpetraExt_MatrixMatrix.hpp"
#include "KokkosSparse_spadd.hpp"

#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <vector>
#include <fstream>
#include "mkl.h"

namespace Tpetra
{
namespace AddProfiling
{

using Tpetra::MatrixMarket::Reader;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::Comm;

#define NUM_ROWS 10000
#define NNZ_PER_ROW 30
#define TRIALS 20

//Produce a random matrix with given nnz per global row
template<typename SC, typename LO, typename GO, typename NT>
RCP<Tpetra::CrsMatrix<SC, LO, GO, NT>> getTestMatrix(RCP<Tpetra::Map<LO, GO, NT>>& rowMap,
    RCP<Tpetra::Map<LO, GO, NT>>& colMap, int seed, RCP<const Comm<int>>& comm)
{
  //create a non-overlapping distributed row map
  auto mat = rcp(new Tpetra::CrsMatrix<SC, LO, GO, NT>(rowMap, colMap, NNZ_PER_ROW));
  //get consistent results between trials
  srand(comm->getRank() * 7 + 42 + seed);
  auto myCols = colMap->getNodeElementList();
  for(GO i = 0; i < NUM_ROWS; i++)
  {
    Teuchos::Array<SC> vals(NNZ_PER_ROW);
    Teuchos::Array<GO> inds(NNZ_PER_ROW);
    for(int j = 0; j < NNZ_PER_ROW; j++)
    {
      vals[j] = ((double) (rand() % RAND_MAX));
      inds[j] = myCols[rand() % myCols.size()];
    }
    mat->insertGlobalValues(i, inds(), vals());
  }
  mat->fillComplete(rowMap, rowMap);
  return mat;
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Tpetra_AddProfiling, sorted, SC, LO, GO, NT)
{
  typedef Tpetra::CrsMatrix<SC, LO, GO, NT> crs_matrix_type;
  typedef Tpetra::Map<LO, GO, NT> map_type;
  RCP<const Comm<int> > comm = DefaultPlatform::getDefaultPlatform().getComm();
  if(comm->getRank() == 0)
    std::cout << "Running sorted add test on " << comm->getSize() << " MPI ranks.\n";
  RCP<map_type> rowMap = rcp(new map_type(NUM_ROWS, 0, comm));
  RCP<map_type> colMap = rcp(new map_type(NUM_ROWS, 0, comm));
  RCP<crs_matrix_type> A = getTestMatrix<SC, LO, GO, NT>(rowMap, colMap, 1, comm);
  RCP<crs_matrix_type> B = getTestMatrix<SC, LO, GO, NT>(rowMap, colMap, 2, comm);
  Kokkos::Impl::Timer addTimer;
  auto one = Teuchos::ScalarTraits<SC>::one();
  for(int i = 0; i < TRIALS; i++)
    RCP<crs_matrix_type> C = MatrixMatrix::add(one, false, *A, one, false, *B);
  double tkernel = addTimer.seconds();
  std::cout << "sorted (kernel): addition took on avg " << (tkernel / TRIALS) << "s.\n";
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Tpetra_AddProfiling, different_col_maps, SC, LO, GO, NT)
{
  typedef Tpetra::CrsMatrix<SC, LO, GO, NT> crs_matrix_type;
  typedef Tpetra::Map<LO, GO, NT> map_type;
  RCP<const Comm<int> > comm = DefaultPlatform::getDefaultPlatform().getComm();
  if(comm->getRank() == 0)
    std::cout << "Running sorted add test on " << comm->getSize() << " MPI ranks.\n";
  RCP<map_type> rowMap = rcp(new map_type(NUM_ROWS, 0, comm));
  RCP<map_type> colMap = rcp(new map_type(NUM_ROWS, 0, comm));
  RCP<crs_matrix_type> A = getTestMatrix<SC, LO, GO, NT>(rowMap, colMap, 1, comm);
  RCP<crs_matrix_type> B = getTestMatrix<SC, LO, GO, NT>(rowMap, colMap, 2, comm);
  Kokkos::Impl::Timer addTimer;
  auto one = Teuchos::ScalarTraits<SC>::one();
  for(int i = 0; i < TRIALS; i++)
    RCP<crs_matrix_type> C = MatrixMatrix::add(one, false, *A, one, false, *B);
  double tkernel = addTimer.seconds();
  std::cout << "sorted (kernel): addition took on avg " << (tkernel / TRIALS) << "s.\n";
}

template<typename SC, typename LO, typename GO, typename NT>
void testMatrix(std::string testDir, std::string csvFile, int trials)
{
  //within directory, all test mats have the same filename
  typedef CrsMatrix<SC, LO, GO, NT> crs_matrix_type;
  typedef typename crs_matrix_type::impl_scalar_type ISC;
  typedef typename crs_matrix_type::local_matrix_type KCRS;
  typedef typename NT::device_type device_type;
  typedef typename device_type::execution_space execution_space;
  typedef typename execution_space::memory_space memory_space;
  typedef typename KCRS::values_type::non_const_type values_array;
  typedef typename KCRS::row_map_type::non_const_type row_ptrs_array;
  typedef typename KCRS::index_type::non_const_type col_inds_array;
  typedef typename Kokkos::View<const GO*, device_type> local_map_type;
  typedef typename Kokkos::View<GO*, device_type> global_col_inds_array;
  typedef KokkosKernels::Experimental::KokkosKernelsHandle<typename col_inds_array::size_type, LO, ISC, execution_space, memory_space, memory_space> KKH;
  RCP<const Comm<int> > comm = DefaultPlatform::getDefaultPlatform().getComm();
  if(comm->getSize() != 1 && comm->getRank() == 0)
  {
    std::cout << "\n\n\n\n***** ONLY RUN THIS TEST WITH 1 MPI RANK!!!!! *****\n\n\n\n";
    exit(1);
  }
  std::string pFile = testDir + "/P_1.m";
  std::string ptentFile = testDir + "/Ptent_1.m";
  std::cout << "\nLoading matrix \"" << pFile << "\"...";
  auto tpetraP = Reader<crs_matrix_type>::readSparseFile(pFile, comm, true);
  std::cout << "done.\n";
  std::cout << "Loading matrix \"" << ptentFile << "\"...";
  auto tpetraPtent = Reader<crs_matrix_type>::readSparseFile(ptentFile, comm, true);
  std::cout << "done.\n";
  KCRS P = tpetraP->getLocalMatrix();
  KCRS Ptent = tpetraPtent->getLocalMatrix();
  MKL_INT nrows = P.numRows();
  std::cout << "Adding matrices with " << nrows << " rows and " << P.values.dimension_0() / nrows << " nnz/row.\n";
  std::vector<double> kkSymbolic;
  std::vector<double> kkNumeric;
  std::vector<double> mklTimes;
  Kokkos::Impl::Timer timer;
  std::cout << "Running KK addition " << trials << " times...";
  for(int i = 0; i < trials; i++)
  {
    timer.reset();
    KKH handle;
    handle.create_spadd_handle(false);
    row_ptrs_array Crowptrs("Crowptrs", nrows + 1);
    KokkosSparse::Experimental::spadd_symbolic <KKH,
      typename row_ptrs_array::const_type, typename col_inds_array::const_type,
      typename row_ptrs_array::const_type, typename col_inds_array::const_type,
      row_ptrs_array, col_inds_array>
        (&handle, P.graph.row_map, P.graph.entries, Ptent.graph.row_map, Ptent.graph.entries, Crowptrs);
    kkSymbolic.push_back(timer.seconds());
    timer.reset();
    values_array Cvalues("C values", handle.get_spadd_handle()->get_max_result_nnz());
    col_inds_array Ccolinds("C colinds", handle.get_spadd_handle()->get_max_result_nnz());
    KokkosSparse::Experimental::spadd_numeric<KKH,
      typename row_ptrs_array::const_type, typename col_inds_array::const_type, ISC, typename values_array::const_type,
      typename row_ptrs_array::const_type, typename col_inds_array::const_type, ISC, typename values_array::const_type,
      row_ptrs_array, col_inds_array, values_array>
    (&handle,
        P.graph.row_map, P.graph.entries, P.values, 1,
        Ptent.graph.row_map, Ptent.graph.entries, Ptent.values, 1,
        Crowptrs, Ccolinds, Cvalues);
    kkNumeric.push_back(timer.seconds());
  }
  std::cout << "done.\n";
  MKL_INT* rowptrsIntP = new MKL_INT[nrows + 1];
  MKL_INT* rowptrsIntPtent = new MKL_INT[nrows + 1];
  for(int i = 0; i <= nrows; i++)
  {
    rowptrsIntP[i] = P.graph.row_map(i);
    rowptrsIntPtent[i] = Ptent.graph.row_map(i);
  }
  std::cout << "Running MKL addition " << trials << " times...";
  for(int i = 0; i < trials; i++)
  {
    timer.reset();
    sparse_matrix_t PMKL;
    sparse_matrix_t PtentMKL;
    mkl_sparse_d_create_csr(&PMKL, SPARSE_INDEX_BASE_ZERO, nrows, nrows, rowptrsIntP, rowptrsIntP + 1, P.graph.entries.data(), P.values.data());
    mkl_sparse_d_create_csr(&PtentMKL, SPARSE_INDEX_BASE_ZERO, nrows, nrows, rowptrsIntPtent, rowptrsIntPtent+ 1, Ptent.graph.entries.data(), Ptent.values.data());
    sparse_matrix_t CMKL;
    if(SPARSE_STATUS_SUCCESS != mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, PMKL, 1, PtentMKL, &CMKL))
    {
      std::cout << "mkl_sparse_d_add() encountered error!\n";
      exit(1);
    }
    //extract CRS arrays of C
    auto zeroIndexing = SPARSE_INDEX_BASE_ZERO;
    MKL_INT* rowsStart = NULL;
    MKL_INT* rowsEnd = NULL;
    MKL_INT* colinds = NULL;
    double* vals = NULL;
    if(SPARSE_STATUS_SUCCESS != mkl_sparse_d_export_csr(CMKL, &zeroIndexing, &nrows, &nrows, &rowsStart, &rowsEnd, &colinds, &vals))
    {
      std::cout << "mkl_sparse_d_export_csr() encountered error!\n";
      exit(1);
    }
    //don't leak the memory allocated for C each trial
    //mkl_sparse_destroy(PMKL);
    //mkl_sparse_destroy(PtentMKL);
    mkl_sparse_destroy(CMKL);
    mklTimes.push_back(timer.seconds());
  }
  std::cout << "done.\n";
  delete[] rowptrsIntP;
  delete[] rowptrsIntPtent;
  std::cout << "Writing output data to \"" << csvFile << "\"...";
  std::ofstream csv(csvFile.c_str());
  //Write out two columns with the data points
  csv << "KK_Symbolic, KK_Numeric, MKL\n";
  for(int i = 0; i < trials; i++)
  {
    csv << kkSymbolic[i] << ", " << kkNumeric[i] << ", " << mklTimes[i] << '\n';
  }
  csv.close();
  std::cout << "done.\n";
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Tpetra_AddProfiling, SIAM, SC, LO, GO, NT)
{
  if(std::is_same<NT, Kokkos::Compat::KokkosSerialWrapperNode>::value)
    return;
  if(std::is_same<GO, long long>::value)
    return;
  int threads = atoi(getenv("OMP_NUM_THREADS"));
  testMatrix<SC, LO, GO, NT>("/home/bmkelle/TestMatrices/Brick3D_25", "Brick25_T" + std::to_string(threads) + ".csv", 1000);
  testMatrix<SC, LO, GO, NT>("/home/bmkelle/TestMatrices/Brick3D_50", "Brick50_T" + std::to_string(threads) + ".csv", 500);
  testMatrix<SC, LO, GO, NT>("/home/bmkelle/TestMatrices/Brick3D_100", "Brick100_T" + std::to_string(threads) + ".csv", 80);
  testMatrix<SC, LO, GO, NT>("/home/bmkelle/TestMatrices/Brick3D_200", "Brick200_T" + std::to_string(threads) + ".csv", 10);
}

#define UNIT_TEST_GROUP_SC_LO_GO_NO( SC, LO, GO, NT )			\
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Tpetra_AddProfiling, SIAM, SC, LO, GO, NT)
  //TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Tpetra_AddProfiling, sorted, SC, LO, GO, NT) \
  //TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Tpetra_AddProfiling, different_col_maps, SC, LO, GO, NT)

  TPETRA_ETI_MANGLING_TYPEDEFS()
  TPETRA_INSTANTIATE_SLGN_NO_ORDINAL_SCALAR( UNIT_TEST_GROUP_SC_LO_GO_NO )

} //namespace AddProfiling
} //namespace Tpetra

