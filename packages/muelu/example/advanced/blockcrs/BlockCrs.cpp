// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
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
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#include <iostream>
#include <stdexcept>

// Teuchos
#include <Teuchos_StandardCatchMacros.hpp>

// Galeri
#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraProblemFactory.hpp>
#include <Galeri_XpetraUtils.hpp>
#include <Galeri_XpetraMaps.hpp>

#include "Xpetra_ConfigDefs.hpp"

#include <MueLu.hpp>
#include <MueLu_Level.hpp>
#include <Tpetra_Operator.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_CrsMatrixMultiplyOp.hpp>
#include <Tpetra_BlockCrsMatrix_Helpers.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_Utilities.hpp>
#include <Xpetra_TpetraVector.hpp>
#include <Xpetra_TpetraBlockCrsMatrix.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Ifpack2_Factory.hpp>

// Belos
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosSolverFactory.hpp"
#include <BelosTpetraAdapter.hpp>

using Teuchos::RCP;
using Teuchos::rcp;


//----------------------------------------------------------------------------------------------------------
//
// This example demonstrates how to use MueLu in a fashion that looks like ML's LevelWrap
//
// In this example, we suppose that the user provides a fine matrix A and P & R operators.  These are
// given to MueLu which then forms the next finest matrix and makes a hierarchy from there.
//
//----------------------------------------------------------------------------------------------------------

const std::string thickSeparator = "==========================================================================================================================";
const std::string thinSeparator  = "--------------------------------------------------------------------------------------------------------------------------";

const std::string prefSeparator = "=====================================";


namespace MueLuExamples {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  typename Teuchos::ScalarTraits<Scalar>::magnitudeType diff_vectors(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> & X,
      const Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> & Y) {
    RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > diff = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(X.getMap());
#else
      const Xpetra::Vector<Scalar,Node> & X,
      const Xpetra::Vector<Scalar,Node> & Y) {
    RCP<Xpetra::Vector<Scalar,Node> > diff = Xpetra::VectorFactory<Scalar,Node>::Build(X.getMap());
#endif
    diff->update(1.0,X,-1.0,Y,0.0);
    typedef typename Teuchos::ScalarTraits<Scalar>::magnitudeType mgn;
    Teuchos::Array<mgn> mt(1);
    diff->norm2(mt);
    return mt[0];
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  typename Teuchos::ScalarTraits<Scalar>::magnitudeType compute_resid_norm(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> & A,
      const Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> & X,
      const Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> & B) {
    RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > temp = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(X.getMap());
#else
      const Xpetra::Matrix<Scalar,Node> & A,
      const Xpetra::Vector<Scalar,Node> & X,
      const Xpetra::Vector<Scalar,Node> & B) {
    RCP<Xpetra::Vector<Scalar,Node> > temp = Xpetra::VectorFactory<Scalar,Node>::Build(X.getMap());
#endif
    A.apply(X,*temp);
    temp->update(1.0,B,-1.0);
    typedef typename Teuchos::ScalarTraits<Scalar>::magnitudeType mgn;
    Teuchos::Array<mgn> mt(1);
    temp->norm2(mt);
    return mt[0];
  }

// --------------------------------------------------------------------------------------
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  void solve_system_belos(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > & A,
      RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > & X,
      RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > & B,
#else
      RCP<Xpetra::Matrix<Scalar,Node> > & A,
      RCP<Xpetra::Vector<Scalar,Node> > & X,
      RCP<Xpetra::Vector<Scalar,Node> > & B,
#endif
      Teuchos::ParameterList & MueLuList,
      const std::string & belos_solver,
      RCP<Teuchos::ParameterList> & SList) {
    using Teuchos::RCP;
    using Teuchos::rcp;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Tpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> Tpetra_Operator;
    typedef Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> Tpetra_CrsMatrix;
    typedef Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> Tpetra_BlockCrsMatrix;
    typedef Xpetra::TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> Xpetra_TpetraBlockCrsMatrix;
    typedef Tpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> Tpetra_Vector;
    typedef Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> Tpetra_MultiVector;
#else
    typedef Tpetra::Operator<Scalar,Node> Tpetra_Operator;
    typedef Tpetra::CrsMatrix<Scalar,Node> Tpetra_CrsMatrix;
    typedef Tpetra::BlockCrsMatrix<Scalar,Node> Tpetra_BlockCrsMatrix;
    typedef Xpetra::TpetraBlockCrsMatrix<Scalar,Node> Xpetra_TpetraBlockCrsMatrix;
    typedef Tpetra::Vector<Scalar,Node> Tpetra_Vector;
    typedef Tpetra::MultiVector<Scalar,Node> Tpetra_MultiVector;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Tpetra_Operator>    At = MueLu::Utilities<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Op2NonConstTpetraRow(A);
#else
    RCP<Tpetra_Operator>    At = MueLu::Utilities<Scalar,Node>::Op2NonConstTpetraRow(A);
#endif
    RCP<Tpetra_Operator>    Mt = MueLu::CreateTpetraPreconditioner(At,MueLuList);
    RCP<Tpetra_MultiVector> Xt = Xpetra::toTpetra(*X);
    RCP<Tpetra_MultiVector> Bt = Xpetra::toTpetra(*B);

    if(Xt.is_null() || Bt.is_null() || At.is_null() || Mt.is_null()) throw std::runtime_error("ERROR: Xpetra to Tpetra conversion failed");

    typedef Tpetra_MultiVector MV;
    typedef Tpetra_Operator OP;
    RCP<Belos::LinearProblem<Scalar,MV,OP> > belosProblem = rcp(new Belos::LinearProblem<Scalar,MV,OP>(At, Xt, Bt));
    belosProblem->setLeftPrec(Mt);
    belosProblem->setProblem(Xt,Bt);

    Belos::SolverFactory<Scalar, MV, OP> BelosFactory;
    Teuchos::RCP<Belos::SolverManager<Scalar, MV, OP> > BelosSolver = BelosFactory.create(belos_solver, SList);
    BelosSolver->setProblem(belosProblem);
    BelosSolver->solve();
  }

 // --------------------------------------------------------------------------------------
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  void solve_system_ifpack2(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > & A,
      RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > & X,
      RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > & B,
#else
      RCP<Xpetra::Matrix<Scalar,Node> > & A,
      RCP<Xpetra::Vector<Scalar,Node> > & X,
      RCP<Xpetra::Vector<Scalar,Node> > & B,
#endif
      const std::string & ifpack2_solver,
      Teuchos::ParameterList & Ifpack2List) {
    using Teuchos::RCP;
    using Teuchos::rcp;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Tpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> Tpetra_Operator;
    typedef Ifpack2::Preconditioner<Scalar,LocalOrdinal,GlobalOrdinal,Node> Ifpack2_Preconditioner;
    typedef Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> Tpetra_CrsMatrix;
    typedef Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> Tpetra_RowMatrix;
    typedef Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> Tpetra_BlockCrsMatrix;
    typedef Xpetra::TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> Xpetra_TpetraBlockCrsMatrix;
    typedef Tpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> Tpetra_Vector;
    typedef Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> Tpetra_MultiVector;
#else
    typedef Tpetra::Operator<Scalar,Node> Tpetra_Operator;
    typedef Ifpack2::Preconditioner<Scalar,Node> Ifpack2_Preconditioner;
    typedef Tpetra::CrsMatrix<Scalar,Node> Tpetra_CrsMatrix;
    typedef Tpetra::RowMatrix<Scalar,Node> Tpetra_RowMatrix;
    typedef Tpetra::BlockCrsMatrix<Scalar,Node> Tpetra_BlockCrsMatrix;
    typedef Xpetra::TpetraBlockCrsMatrix<Scalar,Node> Xpetra_TpetraBlockCrsMatrix;
    typedef Tpetra::Vector<Scalar,Node> Tpetra_Vector;
    typedef Tpetra::MultiVector<Scalar,Node> Tpetra_MultiVector;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Tpetra_RowMatrix>    At = MueLu::Utilities<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Op2NonConstTpetraRow(A);
#else
    RCP<Tpetra_RowMatrix>    At = MueLu::Utilities<Scalar,Node>::Op2NonConstTpetraRow(A);
#endif
    RCP<Tpetra_MultiVector> Xt = Xpetra::toTpetra(*X);
    RCP<Tpetra_MultiVector> Bt = Xpetra::toTpetra(*B);

    RCP<Ifpack2_Preconditioner> Solver = Ifpack2::Factory::create<Tpetra_RowMatrix>(ifpack2_solver,At);
    Solver->setParameters(Ifpack2List);
    Solver->initialize();
    Solver->compute();

    Solver->apply(*Bt,*Xt);
  }

  // --------------------------------------------------------------------------------------
  // This routine generate's the user's original A matrix and nullspace
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void generate_user_matrix_and_nullspace(std::string &matrixType,  Xpetra::UnderlyingLib & lib,Teuchos::ParameterList &galeriList,  RCP<const Teuchos::Comm<int> > &comm, RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > & A, RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > & nullspace){
#else
  template<class Scalar, class Node>
  void generate_user_matrix_and_nullspace(std::string &matrixType,  Xpetra::UnderlyingLib & lib,Teuchos::ParameterList &galeriList,  RCP<const Teuchos::Comm<int> > &comm, RCP<Xpetra::Matrix<Scalar,Node> > & A, RCP<Xpetra::MultiVector<Scalar,Node> > & nullspace){
#endif
    using Teuchos::RCP;

    RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    Teuchos::FancyOStream& out = *fancy;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef typename Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> map_type;
    typedef typename Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> multivector_type;
    typedef typename Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,LocalOrdinal,GlobalOrdinal,Node> realvaluedmultivector_type;
    typedef typename Xpetra::CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node> matrixwrap_type;
#else
    typedef typename Xpetra::Map<Node> map_type;
    typedef typename Xpetra::MultiVector<Scalar,Node> multivector_type;
    typedef typename Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,Node> realvaluedmultivector_type;
    typedef typename Xpetra::CrsMatrixWrap<Scalar,Node> matrixwrap_type;
#endif
    RCP<const map_type >   map;
    RCP<multivector_type > coordinates;
    if (matrixType == "Laplace1D") {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      map = Galeri::Xpetra::CreateMap<LocalOrdinal, GlobalOrdinal, Node>(lib, "Cartesian1D", comm, galeriList);
#else
      map = Galeri::Xpetra::CreateMap<Node>(lib, "Cartesian1D", comm, galeriList);
#endif
      coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<Scalar,LocalOrdinal,GlobalOrdinal,map_type,multivector_type>("1D", map, galeriList);

    } else if (matrixType == "Laplace2D" || matrixType == "Star2D" || matrixType == "BigStar2D" || matrixType == "Elasticity2D") {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      map = Galeri::Xpetra::CreateMap<LocalOrdinal, GlobalOrdinal, Node>(lib, "Cartesian2D", comm, galeriList);
#else
      map = Galeri::Xpetra::CreateMap<Node>(lib, "Cartesian2D", comm, galeriList);
#endif
      coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<Scalar,LocalOrdinal,GlobalOrdinal,map_type,multivector_type>("2D", map, galeriList);

    } else if (matrixType == "Laplace3D" || matrixType == "Brick3D" || matrixType == "Elasticity3D") {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      map = Galeri::Xpetra::CreateMap<LocalOrdinal, GlobalOrdinal, Node>(lib, "Cartesian3D", comm, galeriList);
#else
      map = Galeri::Xpetra::CreateMap<Node>(lib, "Cartesian3D", comm, galeriList);
#endif
      coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<Scalar,LocalOrdinal,GlobalOrdinal,map_type,multivector_type>("3D", map, galeriList);
    }

    // Expand map to do multiple DOF per node for block problems
    if (matrixType == "Elasticity2D" || matrixType == "Elasticity3D")
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      map = Xpetra::MapFactory<LocalOrdinal,GlobalOrdinal,Node>::Build(map, (matrixType == "Elasticity2D" ? 2 : 3));
#else
      map = Xpetra::MapFactory<Node>::Build(map, (matrixType == "Elasticity2D" ? 2 : 3));
#endif

    out << "Processor subdomains in x direction: " << galeriList.get<GlobalOrdinal>("mx") << std::endl
        << "Processor subdomains in y direction: " << galeriList.get<GlobalOrdinal>("my") << std::endl
        << "Processor subdomains in z direction: " << galeriList.get<GlobalOrdinal>("mz") << std::endl
        << "========================================================" << std::endl;

    RCP<Galeri::Xpetra::Problem<map_type,matrixwrap_type,multivector_type> > Pr =
      Galeri::Xpetra::BuildProblem<Scalar,LocalOrdinal,GlobalOrdinal,map_type,matrixwrap_type,multivector_type>(matrixType, map, galeriList);

    A = Pr->BuildMatrix();

    if (matrixType == "Elasticity2D" || matrixType == "Elasticity3D") {
      nullspace = Pr->BuildNullspace();
      A->SetFixedBlockSize((matrixType == "Elasticity2D") ? 2 : 3);
    }

  }

}


// --------------------------------------------------------------------------------------
//int main(int argc, char *argv[]) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
int main_(Teuchos::CommandLineProcessor &clp, Xpetra::UnderlyingLib lib, int argc, char *argv[]) {
#include <MueLu_UseShortNames.hpp>
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::TimeMonitor;
  bool success = true;
  bool verbose = true;
  try {
    RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();

    RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    Teuchos::FancyOStream& out = *fancy;

    typedef Teuchos::ScalarTraits<SC> STS;

    // =========================================================================
    // Parameters initialization
    // =========================================================================
    //Teuchos::CommandLineProcessor clp(false);

    GO nx = 100, ny = 100, nz = 100;
    Galeri::Xpetra::Parameters<GO> galeriParameters(clp, nx, ny, nz, "Laplace2D"); // manage parameters of the test case
    Xpetra::Parameters             xpetraParameters(clp);                          // manage parameters of Xpetra
    std::string matFileName       = "";  clp.setOption("matrix",&matFileName,"read matrix from a file");
    LO blocksize = 1;                    clp.setOption("blocksize",&blocksize,"block size");

    switch (clp.parse(argc, argv)) {
      case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:        return EXIT_SUCCESS; break;
      case Teuchos::CommandLineProcessor::PARSE_ERROR:
      case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: return EXIT_FAILURE; break;
      case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:                               break;
    }

    //Xpetra::UnderlyingLib lib = xpetraParameters.GetLib();
    Teuchos::ParameterList galeriList = galeriParameters.GetParameterList();

    if(lib!=Xpetra::UseTpetra)
      throw std::runtime_error("This test only works with Tpetra linear algebra");

    // =========================================================================
    // Problem construction
    // =========================================================================
    RCP<const Map>   map;
    RCP<Matrix> A;
    RCP<MultiVector> nullspace;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Tpetra::CrsMatrix<SC,LO,GO,NO> Tpetra_CrsMatrix;
    typedef Tpetra::Operator<SC,LO,GO,NO> Tpetra_Operator;
    typedef Tpetra::BlockCrsMatrix<SC,LO,GO,NO> Tpetra_BlockCrsMatrix;
    typedef Xpetra::TpetraBlockCrsMatrix<SC,LO,GO,NO> Xpetra_TpetraBlockCrsMatrix;
    typedef Xpetra::CrsMatrix<SC,LO,GO,NO> Xpetra_CrsMatrix;
    typedef Xpetra::CrsMatrixWrap<SC,LO,GO,NO> Xpetra_CrsMatrixWrap;
#else
    typedef Tpetra::CrsMatrix<SC,NO> Tpetra_CrsMatrix;
    typedef Tpetra::Operator<SC,NO> Tpetra_Operator;
    typedef Tpetra::BlockCrsMatrix<SC,NO> Tpetra_BlockCrsMatrix;
    typedef Xpetra::TpetraBlockCrsMatrix<SC,NO> Xpetra_TpetraBlockCrsMatrix;
    typedef Xpetra::CrsMatrix<SC,NO> Xpetra_CrsMatrix;
    typedef Xpetra::CrsMatrixWrap<SC,NO> Xpetra_CrsMatrixWrap;
#endif
    typedef typename Teuchos::ScalarTraits<SC>::magnitudeType SCN;

    SC one = Teuchos::ScalarTraits<SC>::one();

    RCP<Tpetra_CrsMatrix> Acrs;
    RCP<Tpetra_BlockCrsMatrix> Ablock;

    if(matFileName.length() > 0) {
      // Read matrix from disk
      out << thickSeparator << std::endl << "Reading matrix from disk" <<std::endl;
      // TODO
      typedef Tpetra::MatrixMarket::Reader<Tpetra_CrsMatrix> reader_type;
      Acrs = reader_type::readSparseFile(matFileName,comm);
    }
    else{
      // Use Galeri
      out << thickSeparator << std::endl << xpetraParameters << galeriParameters;
      std::string matrixType = galeriParameters.GetMatrixType();
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Axp;
      MueLuExamples::generate_user_matrix_and_nullspace<Scalar,LocalOrdinal,GlobalOrdinal,Node>(matrixType,lib,galeriList,comm,Axp,nullspace);
      Acrs = Xpetra::Helpers<SC,LO,GO,NO>::Op2NonConstTpetraCrs(Axp);
#else
      RCP<Xpetra::Matrix<Scalar,Node> > Axp;
      MueLuExamples::generate_user_matrix_and_nullspace<Scalar,Node>(matrixType,lib,galeriList,comm,Axp,nullspace);
      Acrs = Xpetra::Helpers<SC,NO>::Op2NonConstTpetraCrs(Axp);
#endif
    }
    // Block this bad boy
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Ablock = Tpetra::convertToBlockCrsMatrix<SC,LO,GO,NO>(*Acrs,blocksize);
#else
    Ablock = Tpetra::convertToBlockCrsMatrix<SC,NO>(*Acrs,blocksize);
#endif

    // Now wrap BlockCrs to Xpetra::Matrix
    RCP<Xpetra_CrsMatrix> Axt = rcp(new Xpetra_TpetraBlockCrsMatrix(Ablock));
    A = rcp(new Xpetra_CrsMatrixWrap(Axt));

    // =========================================================================
    // Setups and solves
    // =========================================================================
    map=Xpetra::toXpetra(Acrs->getRowMap());

    RCP<Vector> X1 = VectorFactory::Build(map);
    RCP<Vector> X2 = VectorFactory::Build(map);
    RCP<Vector> B = VectorFactory::Build(map);
    B->setSeed(846930886);
    B->randomize();
    RCP<TimeMonitor> tm;

    // Belos Options
    RCP<Teuchos::ParameterList> SList = rcp(new Teuchos::ParameterList );
    SList->set("Verbosity",Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
    SList->set("Output Frequency",10);
    SList->set("Output Style",Belos::Brief);
    SList->set("Maximum Iterations",10);
    SList->set("Convergence Tolerance",5e-2);

    // =========================================================================
    // Solve #1 (fixed point + Jacobi)
    // =========================================================================
    out << thickSeparator << std::endl;
    out << prefSeparator << " Solve 1: Fixed Point + Jacobi"<< prefSeparator <<std::endl;
    {
      Teuchos::ParameterList MueList;
      MueList.set("max levels",1);
      MueList.set("coarse: type", "RELAXATION");

      std::string belos_solver("Fixed Point");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      MueLuExamples::solve_system_belos<Scalar,LocalOrdinal,GlobalOrdinal,Node>(A,X1,B,MueList,belos_solver,SList);
#else
      MueLuExamples::solve_system_belos<Scalar,Node>(A,X1,B,MueList,belos_solver,SList);
#endif
      std::cout << "I" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      SCN result = MueLuExamples::compute_resid_norm<Scalar,LocalOrdinal,GlobalOrdinal,Node>(*A,*X1,*B);
#else
      SCN result = MueLuExamples::compute_resid_norm<Scalar,Node>(*A,*X1,*B);
#endif
      out<<"Solve #1: Residual Norm = "<<result<<std::endl;
    }

    // =========================================================================
    // Solve #2 (striaght up Jacobi)
    // =========================================================================
    out << thickSeparator << std::endl;
    out << prefSeparator << " Solve 2: Fixed Jacobi"<< prefSeparator <<std::endl;
    {
      Teuchos::ParameterList IList;
      IList.set("relaxation: type","Jacobi");
      IList.set("relaxation: damping factor",one);
      IList.set("relaxation: sweeps",10);
      std::string ifpack2_precond("RELAXATION");

      MueLuExamples::solve_system_ifpack2(A,X2,B,ifpack2_precond,IList);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      SCN result = MueLuExamples::compute_resid_norm<Scalar,LocalOrdinal,GlobalOrdinal,Node>(*A,*X2,*B);
#else
      SCN result = MueLuExamples::compute_resid_norm<Scalar,Node>(*A,*X2,*B);
#endif
      out<<"Solve #2: Residual Norm = "<<result<<std::endl;
    }

    // Compare 1 & 2
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    SCN norm = MueLuExamples::diff_vectors<Scalar,LocalOrdinal,GlobalOrdinal,Node>(*X1,*X2);
#else
    SCN norm = MueLuExamples::diff_vectors<Scalar,Node>(*X1,*X2);
#endif
    if(norm > 1e-10) {
      out<<"ERROR: Norm of Solve #1 and Solve #2 differs by "<<norm<<std::endl;
      success=false;
    }

  }//end try
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}

//- -- --------------------------------------------------------
#define MUELU_AUTOMATIC_TEST_ETI_NAME main_
#include "MueLu_Test_ETI.hpp"

int main(int argc, char *argv[]) {
  return Automatic_Test_ETI(argc,argv);
}


