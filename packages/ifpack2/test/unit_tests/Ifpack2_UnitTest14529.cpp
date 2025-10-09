// @HEADER
// *****************************************************************************
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//
// Copyright 2009 NTESS and the Ifpack2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <Teuchos_ConfigDefs.hpp>
#include <Ifpack2_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Ifpack2_Version.hpp>
#include <iostream>

#include <Ifpack2_UnitTestHelpers.hpp>
#include <Ifpack2_BlockRelaxation.hpp>
#include <Ifpack2_BlockTriDiContainer.hpp>

namespace {
TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Ifpack2BlockTriDiContainer, Issue14529, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
  // Teuchos test passes in: Teuchos::FancyOStream& out, bool& success
  using RowMatrix = Tpetra::RowMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using BCRS = Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using Graph = typename BCRS::crs_graph_type;
  using Map = typename BCRS::map_type;
  using ValuesView = typename BCRS::local_matrix_device_type::values_type;
  using Teuchos::RCP;
  using Teuchos::rcp;

  /*
  RCP<const Graph> graph = tif_utest::create_tridiag_graph<LocalOrdinal, GlobalOrdinal, Node>(0);

  ValuesView vals("vals", 0);
  // Arbitrarily choosing blockSize = 5 here
  RCP<BCRS> A = rcp(new BCRS(*graph, vals, 5));
  //std::cout << "Matrix with 0 rows:\n";
  //A.describe(out, Teuchos::VERB_EXTREME);

  Teuchos::Array<Teuchos::Array<LocalOrdinal>> parts;
  //auto prec = Teuchos::rcp(new Ifpack2::BlockTriDiContainer<RowMatrix>(A, parts, 1, /* overlap_comm */ false, /* seq_method */ false, /* block_size */ 5));

  /*
  prec->initialize();
  prec->compute();
  */
  success = true;
}

#define UNIT_TEST_GROUP(Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Ifpack2BlockTriDiContainer, Issue14529, Scalar, LO, GO, Node)

#include "Ifpack2_ETIHelperMacros.h"

IFPACK2_ETI_MANGLING_TYPEDEFS()

IFPACK2_INSTANTIATE_SLGN(UNIT_TEST_GROUP)

}
