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
#include "mkl.h"

namespace Tpetra
{
namespace AddProfiling
{

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::Comm;

#define NUM_ROWS 100000
#define NNZ_PER_ROW 30
#define TRIALS 5

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

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Tpetra_AddProfiling, SIAM, SC, LO, GO, NT)
{
  if(std::is_same<NT, Kokkos::Compat::KokkosSerialWrapperNode>::value)
    return;
  typedef Tpetra::CrsMatrix<SC, LO, GO, NT> crs_matrix_type;
  typedef Tpetra::Map<LO, GO, NT> map_type;
  {
    //not actually using comm/map for anything, except to force Tpetra
    //to initialize Kokkos properly
    RCP<const Comm<int> > comm = DefaultPlatform::getDefaultPlatform().getComm();
    if(comm->getRank() == 0)
      std::cout << "Running sorted add test on " << comm->getSize() << " MPI ranks.\n";
    rcp(new map_type(NUM_ROWS, 0, comm));
  }
  std::cout << "Node type: " << typeid(NT).name() << '\n';
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
  row_ptrs_array Arowptrs("A rowptrs", NUM_ROWS + 1);
  row_ptrs_array Browptrs("B rowptrs", NUM_ROWS + 1);
  row_ptrs_array Crowptrs("C rowptrs", NUM_ROWS + 1);
  for(int i = 0; i <= NUM_ROWS; i++)
  {
    Arowptrs(i) = i * NNZ_PER_ROW;
    Browptrs(i) = i * NNZ_PER_ROW;
  }
  values_array Avalues("A values", NUM_ROWS * NNZ_PER_ROW);
  values_array Bvalues("B values", NUM_ROWS * NNZ_PER_ROW);
  for(int i = 0; i < NUM_ROWS * NNZ_PER_ROW; i++)
  {
    Avalues(i) = 10 * (double(rand()) / RAND_MAX);
    Bvalues(i) = 10 * (double(rand()) / RAND_MAX);
  }
  col_inds_array Acolinds("A colinds", NUM_ROWS * NNZ_PER_ROW);
  col_inds_array Bcolinds("B colinds", NUM_ROWS * NNZ_PER_ROW);
  std::vector<LO> randColinds(NUM_ROWS);
  for(int i = 0; i < NUM_ROWS; i++)
  {
    //want randomized colinds with no duplicates
    std::set<LO> taken;
    for(int j = 0; j < NNZ_PER_ROW; j++)
    {
      LO next;
      do
      {
        next = rand() % NUM_ROWS;
      }
      while(taken.find(next) != taken.end());
      Acolinds(i * NNZ_PER_ROW + j) = next;
      taken.insert(next);
    }
    taken.clear();
    for(int j = 0; j < NNZ_PER_ROW; j++)
    {
      LO next;
      do
      {
        next = rand() % NUM_ROWS;
      }
      while(taken.find(next) != taken.end());
      Bcolinds(i * NNZ_PER_ROW + j) = next;
      taken.insert(next);
    }
  }
  Kokkos::Impl::Timer addTimer;
  for(int i = 0; i < TRIALS; i++)
  {
    KKH handle;
    handle.create_spadd_handle(false);
    KokkosSparse::Experimental::spadd_symbolic <KKH,
      typename row_ptrs_array::const_type, typename col_inds_array::const_type,
      typename row_ptrs_array::const_type, typename col_inds_array::const_type,
      row_ptrs_array, col_inds_array>
        (&handle, Arowptrs, Acolinds, Browptrs, Bcolinds, Crowptrs);
    values_array Cvalues("C values", handle.get_spadd_handle()->get_max_result_nnz());
    col_inds_array Ccolinds("C colinds", handle.get_spadd_handle()->get_max_result_nnz());
    KokkosSparse::Experimental::spadd_numeric<KKH,
      typename row_ptrs_array::const_type, typename col_inds_array::const_type, ISC, typename values_array::const_type,
      typename row_ptrs_array::const_type, typename col_inds_array::const_type, ISC, typename values_array::const_type,
      row_ptrs_array, col_inds_array, values_array>
    (&handle,
        Arowptrs, Acolinds, Avalues, 1,
        Browptrs, Bcolinds, Bvalues, 1,
        Crowptrs, Ccolinds, Cvalues);
  }
  double kkTime = addTimer.seconds();
  //copy to MKL-compatible format
  sparse_matrix_t AMKL;
  sparse_matrix_t BMKL;
  sparse_matrix_t CMKL;
  int* arowptrsInt = new int[NUM_ROWS + 1];
  int* browptrsInt = new int[NUM_ROWS + 1];
  for(int i = 0; i < NUM_ROWS + 1; i++)
  {
    arowptrsInt[i] = Arowptrs(i);
    browptrsInt[i] = Browptrs(i);
  }
  mkl_sparse_d_create_csr(&AMKL, SPARSE_INDEX_BASE_ZERO, NUM_ROWS, NUM_ROWS, arowptrsInt, arowptrsInt + 1, Acolinds.data(), Avalues.data());
  mkl_sparse_d_create_csr(&BMKL, SPARSE_INDEX_BASE_ZERO, NUM_ROWS, NUM_ROWS, browptrsInt, browptrsInt + 1, Bcolinds.data(), Bvalues.data());
  addTimer.reset();
  for(int i = 0; i < TRIALS; i++)
  {
  /*
    const int mklSortMode = 3;      //sort while adding
    const char mklTransMode = 'N';  //no transpose
    const int mklSymbolic = 1;
    const int mklNumeric = 2;
    mkl_dcsradd(&mklTransMode, &mklSymbolic, &mklSortMode, NUM_ROWS, NUM_ROWS, , MKL_INT *ja , MKL_INT *ia , const double *beta , double *b , MKL_INT *jb , MKL_INT *ib , double *c , MKL_INT *jc , MKL_INT *ic , const MKL_INT *nzmax , MKL_INT *info );
    */
    mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, AMKL, 1, BMKL, &CMKL);
    //don't leak the memory allocated for C each trial
    mkl_sparse_destroy(CMKL);
  }
  double mklTime = addTimer.seconds();
  delete[] arowptrsInt;
  delete[] browptrsInt;
  std::cout << "KK addition took on avg:  " << (kkTime/ TRIALS) << " s.\n";
  std::cout << "MKL addition took on avg: " << (mklTime / TRIALS) << " s.\n";
}

#define UNIT_TEST_GROUP_SC_LO_GO_NO( SC, LO, GO, NT )			\
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Tpetra_AddProfiling, SIAM, SC, LO, GO, NT)
  //TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Tpetra_AddProfiling, sorted, SC, LO, GO, NT) \
  //TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Tpetra_AddProfiling, different_col_maps, SC, LO, GO, NT)

  TPETRA_ETI_MANGLING_TYPEDEFS()
  TPETRA_INSTANTIATE_SLGN_NO_ORDINAL_SCALAR( UNIT_TEST_GROUP_SC_LO_GO_NO )

} //namespace AddProfiling
} //namespace Tpetra

