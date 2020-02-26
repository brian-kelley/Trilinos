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

// New definition of types using the types LocalOrdinal, GlobalOrdinal, Node of the current context.
#ifdef XPETRA_MAP_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> Map;
#else
typedef Xpetra::Map<Node> Map;
#endif
#endif

#ifdef XPETRA_MAPUTILS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MapUtils<LocalOrdinal, GlobalOrdinal, Node> MapUtils;
#else
typedef Xpetra::MapUtils<Node> MapUtils;
#endif
#endif

#ifdef XPETRA_MAPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node> MapFactory;
#else
typedef Xpetra::MapFactory<Node> MapFactory;
#endif
#endif

#ifdef XPETRA_BLOCKEDMAP_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::BlockedMap<LocalOrdinal, GlobalOrdinal, Node> BlockedMap;
#else
typedef Xpetra::BlockedMap<Node> BlockedMap;
#endif
#endif

#ifdef XPETRA_CRSGRAPH_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> CrsGraph;
#else
typedef Xpetra::CrsGraph<Node> CrsGraph;
#endif
#endif

#ifdef XPETRA_CRSGRAPHFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::CrsGraphFactory<LocalOrdinal, GlobalOrdinal, Node> CrsGraphFactory;
#else
typedef Xpetra::CrsGraphFactory<Node> CrsGraphFactory;
#endif
#endif

#ifdef XPETRA_VECTOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::Vector<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node> LocalOrdinalVector;
typedef Xpetra::Vector<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> GlobalOrdinalVector;
#else
typedef Xpetra::Vector<LocalOrdinal, Node> LocalOrdinalVector;
typedef Xpetra::Vector<GlobalOrdinal, Node> GlobalOrdinalVector;
#endif
typedef LocalOrdinalVector  LOVector;
typedef GlobalOrdinalVector GOVector;
#endif

#ifdef XPETRA_MULTIVECTOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MultiVector<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node> LocalOrdinalMultiVector;
typedef Xpetra::MultiVector<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> GlobalOrdinalMultiVector;
#else
typedef Xpetra::MultiVector<LocalOrdinal, Node> LocalOrdinalMultiVector;
typedef Xpetra::MultiVector<GlobalOrdinal, Node> GlobalOrdinalMultiVector;
#endif
typedef LocalOrdinalMultiVector  LOMultiVector;
typedef GlobalOrdinalMultiVector GOMultiVector;
#endif

#ifdef XPETRA_VECTORFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::VectorFactory<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node> LocalOrdinalVectorFactory;
typedef Xpetra::VectorFactory<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> GlobalOrdinalVectorFactory;
#else
typedef Xpetra::VectorFactory<LocalOrdinal, Node> LocalOrdinalVectorFactory;
typedef Xpetra::VectorFactory<GlobalOrdinal, Node> GlobalOrdinalVectorFactory;
#endif
typedef LocalOrdinalVectorFactory  LOVectorFactory;
typedef GlobalOrdinalVectorFactory GOVectorFactory;
#endif

#ifdef XPETRA_MULTIVECTORFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::MultiVectorFactory<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node> LocalOrdinalMultiVectorFactory;
typedef Xpetra::MultiVectorFactory<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> GlobalOrdinalMultiVectorFactory;
#else
typedef Xpetra::MultiVectorFactory<LocalOrdinal, Node> LocalOrdinalMultiVectorFactory;
typedef Xpetra::MultiVectorFactory<GlobalOrdinal, Node> GlobalOrdinalMultiVectorFactory;
#endif
typedef LocalOrdinalMultiVectorFactory  LOMultiVectorFactory;
typedef GlobalOrdinalMultiVectorFactory GOMultiVectorFactory;
#endif

#ifdef XPETRA_IMPORT_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::Import<LocalOrdinal, GlobalOrdinal, Node> Import;
#else
typedef Xpetra::Import<Node> Import;
#endif
#endif

#ifdef XPETRA_EXPORT_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::Export<LocalOrdinal, GlobalOrdinal, Node> Export;
#else
typedef Xpetra::Export<Node> Export;
#endif
#endif

#ifdef XPETRA_IMPORTFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::ImportFactory<LocalOrdinal, GlobalOrdinal, Node> ImportFactory;
#else
typedef Xpetra::ImportFactory<Node> ImportFactory;
#endif
#endif

#ifdef XPETRA_EXPORTFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::ExportFactory<LocalOrdinal, GlobalOrdinal, Node> ExportFactory;
#else
typedef Xpetra::ExportFactory<Node> ExportFactory;
#endif
#endif

#ifdef XPETRA_TPETRAMAP_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::TpetraMap<LocalOrdinal, GlobalOrdinal, Node> TpetraMap;
#else
typedef Xpetra::TpetraMap<Node> TpetraMap;
#endif
#endif

#ifdef XPETRA_TPETRACRSGRAPH_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::TpetraCrsGraph<LocalOrdinal, GlobalOrdinal, Node> TpetraCrsGraph;
#else
typedef Xpetra::TpetraCrsGraph<Node> TpetraCrsGraph;
#endif
#endif

#ifdef XPETRA_STRIDEDMAP_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::StridedMap<LocalOrdinal, GlobalOrdinal, Node> StridedMap;
#else
typedef Xpetra::StridedMap<Node> StridedMap;
#endif
#endif

#ifdef XPETRA_STRIDEDMAPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef Xpetra::StridedMapFactory<LocalOrdinal, GlobalOrdinal, Node> StridedMapFactory;
#else
typedef Xpetra::StridedMapFactory<Node> StridedMapFactory;
#endif
#endif

// Note: There is no #ifndef/#define/#end in this header file because it can be included more than once (it can be included in methods templated by Scalar, LocalOrdinal, GlobalOrdinal, Node).

// TODO: add namespace {} for shortcut types

// Define convenient shortcut for data types
typedef LocalOrdinal  LO;
typedef GlobalOrdinal GO;
typedef Node          NO;

// TODO: do the same for Epetra object (problem of namespace)
