// @HEADER
//
// ***********************************************************************
//
//             Xpetra: A linear algebra interface package
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
// Get rid of template parameters

// New definition of types using the types Scalar, LocalOrdinal, GlobalOrdinal, Node of the current context.

// Note: There is no #ifndef/#define/#end in this header file because it can be included more than once (it can be included in methods templated by Scalar, LocalOrdinal, GlobalOrdinal, Node).

#ifdef XPETRA_CRSMATRIX_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> CrsMatrix;
#else
typedef Xpetra::CrsMatrix<Scalar, Node> CrsMatrix;
#endif
#endif

#ifdef XPETRA_IO_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node> IO;
#else
typedef Xpetra::IO<Scalar, Node> IO;
#endif
#endif

#ifdef XPETRA_ITERATOROPS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::IteratorOps<Scalar, LocalOrdinal, GlobalOrdinal, Node> IteratorOps;
#else
typedef Xpetra::IteratorOps<Scalar, Node> IteratorOps;
#endif
#endif

#ifdef XPETRA_VECTOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> Vector;
#else
typedef Xpetra::Vector<Scalar, Node> Vector;
#endif
#endif

#ifdef XPETRA_BLOCKEDVECTOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> BlockedVector;
#else
typedef Xpetra::BlockedVector<Scalar, Node> BlockedVector;
#endif
#endif

#ifdef XPETRA_MULTIVECTOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> MultiVector;
#else
typedef Xpetra::MultiVector<Scalar, Node> MultiVector;
#endif
#endif

#ifdef XPETRA_MATRIX_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> Matrix;
#else
typedef Xpetra::Matrix<Scalar, Node> Matrix;
#endif
#endif

#ifdef XPETRA_MATRIXMATRIX_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> MatrixMatrix;
#else
typedef Xpetra::MatrixMatrix<Scalar, Node> MatrixMatrix;
#endif
#endif

#ifdef XPETRA_TRIPLEMATRIXMULTIPLY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::TripleMatrixMultiply<Scalar, LocalOrdinal, GlobalOrdinal, Node> TripleMatrixMultiply;
#else
typedef Xpetra::TripleMatrixMultiply<Scalar, Node> TripleMatrixMultiply;
#endif
#endif

#ifdef XPETRA_MATRIXUTILS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MatrixUtils<Scalar, LocalOrdinal, GlobalOrdinal, Node> MatrixUtils;
#else
typedef Xpetra::MatrixUtils<Scalar, Node> MatrixUtils;
#endif
#endif

#ifdef XPETRA_OPERATOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> Operator;
#else
typedef Xpetra::Operator<Scalar, Node> Operator;
#endif
#endif

#ifdef XPETRA_TPETRAOPERATOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::TpetraOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node> TpetraOperator;
#else
typedef Xpetra::TpetraOperator<Scalar, Node> TpetraOperator;
#endif
#endif

#ifdef XPETRA_BLOCKEDCRSMATRIX_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> BlockedCrsMatrix;
#else
typedef Xpetra::BlockedCrsMatrix<Scalar, Node> BlockedCrsMatrix;
#endif
#endif

#ifdef XPETRA_BLOCKEDMULTIVECTOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::BlockedMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> BlockedMultiVector;
#else
typedef Xpetra::BlockedMultiVector<Scalar, Node> BlockedMultiVector;
#endif
#endif

#ifdef XPETRA_REORDEREDBLOCKEDMULTIVECTOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::ReorderedBlockedMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> ReorderedBlockedMultiVector;
#else
typedef Xpetra::ReorderedBlockedMultiVector<Scalar, Node> ReorderedBlockedMultiVector;
#endif
#endif

#ifdef XPETRA_REORDEREDBLOCKEDCRSMATRIX_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::ReorderedBlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> ReorderedBlockedCrsMatrix;
#else
typedef Xpetra::ReorderedBlockedCrsMatrix<Scalar, Node> ReorderedBlockedCrsMatrix;
#endif
#endif

#ifdef HAVE_XPETRA_THYRA
#ifdef XPETRA_THYRAUTILS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::ThyraUtils<Scalar, LocalOrdinal, GlobalOrdinal, Node> ThyraUtils;
#else
typedef Xpetra::ThyraUtils<Scalar, Node> ThyraUtils;
#endif
#endif
#endif

#ifdef XPETRA_CRSMATRIXWRAP_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node> CrsMatrixWrap;
#else
typedef Xpetra::CrsMatrixWrap<Scalar, Node> CrsMatrixWrap;
#endif
#endif

#ifdef XPETRA_VECTORFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::VectorFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node> VectorFactory;
#else
typedef Xpetra::VectorFactory<Scalar, Node> VectorFactory;
#endif
#endif

#ifdef XPETRA_CRSMATRIXFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::CrsMatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node> CrsMatrixFactory;
#else
typedef Xpetra::CrsMatrixFactory<Scalar, Node> CrsMatrixFactory;
#endif
#endif

#ifdef XPETRA_MULTIVECTORFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MultiVectorFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node> MultiVectorFactory;
#else
typedef Xpetra::MultiVectorFactory<Scalar, Node> MultiVectorFactory;
#endif
#endif

#ifdef XPETRA_MATRIXFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node> MatrixFactory;
#else
typedef Xpetra::MatrixFactory<Scalar, Node> MatrixFactory;
#endif
#endif

#ifdef XPETRA_MATRIXFACTORY2_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MatrixFactory2<Scalar, LocalOrdinal, GlobalOrdinal, Node> MatrixFactory2;
#else
typedef Xpetra::MatrixFactory2<Scalar, Node> MatrixFactory2;
#endif
#endif

#ifdef XPETRA_TPETRACRSMATRIX_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> TpetraCrsMatrix;
#else
typedef Xpetra::TpetraCrsMatrix<Scalar, Node> TpetraCrsMatrix;
#endif
#endif

// TODO remove this
#ifdef XPETRA_EPETRACRSMATRIX_SHORT
#ifndef XPETRA_EPETRA_NO_32BIT_GLOBAL_INDICES
typedef Xpetra::EpetraCrsMatrixT<long long, Xpetra::EpetraNode> EpetraCrsMatrix64;
#endif
typedef Xpetra::EpetraCrsMatrixT<int, Xpetra::EpetraNode> EpetraCrsMatrix; // do we need this???
#endif
// TODO remove above entries

#ifdef XPETRA_TPETRAMULTIVECTOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> TpetraMultiVector;
#else
typedef Xpetra::TpetraMultiVector<Scalar, Node> TpetraMultiVector;
#endif
#endif

#ifdef XPETRA_TPETRAVECTOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::TpetraVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> TpetraVector;
#else
typedef Xpetra::TpetraVector<Scalar, Node> TpetraVector;
#endif
#endif

#ifdef XPETRA_MAPEXTRACTOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node> MapExtractor;
#else
typedef Xpetra::MapExtractor<Scalar, Node> MapExtractor;
#endif
#endif

#ifdef XPETRA_MAPEXTRACTORFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MapExtractorFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node> MapExtractorFactory;
#else
typedef Xpetra::MapExtractorFactory<Scalar, Node> MapExtractorFactory;
#endif
#endif

// TODO: add namespace {} for shortcut types

// Define convenient shortcut for data types
typedef Scalar    SC;
// TODO: do the same for Epetra object (problem of namespace)
