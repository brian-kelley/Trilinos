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

#ifndef MUELU_COARSENINGVISUALIZATIONFACTORY_DEF_HPP_
#define MUELU_COARSENINGVISUALIZATIONFACTORY_DEF_HPP_

#include <Xpetra_Matrix.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include "MueLu_CoarseningVisualizationFactory_decl.hpp"
#include "MueLu_Level.hpp"


namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> CoarseningVisualizationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {

    RCP<ParameterList> validParamList = VizHelpers::GetVizParameterList();

    validParamList->set< int >                   ("visualization: start level",             0,                     "visualize only levels with level ids greater or equal than start level");// Remove me?

    validParamList->set< RCP<const FactoryBase> >("P",           Teuchos::null, "Prolongator factory. The user has to declare either P or Ptent but not both at the same time.");
    validParamList->set< RCP<const FactoryBase> >("Ptent",       Teuchos::null, "Tentative prolongator factory. The user has to declare either P or Ptent as input but not both at the same time");
    validParamList->set< RCP<const FactoryBase> >("Coordinates", Teuchos::null, "Factory for Coordinates.");
    validParamList->set< RCP<const FactoryBase> >("Graph",       Teuchos::null, "Factory for Graph.");

    return validParamList;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CoarseningVisualizationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level &fineLevel, Level &coarseLevel) const {
    this->Input(fineLevel, "Coordinates");

    const ParameterList & pL = this->GetParameterList();

    if (GetFactory("P") != Teuchos::null)
      this->Input(coarseLevel, "P");
    if (GetFactory("Ptent") != Teuchos::null)
      this->Input(coarseLevel, "Ptent");

    if(pL.get<bool>("visualization: fine graph edges"))
      Input(fineLevel, "Graph");
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CoarseningVisualizationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level &fineLevel, Level &coarseLevel) const {

    typedef VizHelpers::AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeometry;
    typedef VizHelpers::EdgeGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> EdgeGeometry;
    typedef VizHelpers::VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node> VTKEmitter;

    RCP<GraphBase> fineGraph = Teuchos::null;
    RCP<Matrix>    P         = Teuchos::null;
    RCP<Matrix>    Ptent     = Teuchos::null;
    ParameterList pL = this->GetParameterList();
    if (this->GetFactory("P") != Teuchos::null)
      P = Get< RCP<Matrix> >(coarseLevel, "P");
    if (this->GetFactory("Ptent") != Teuchos::null)
      Ptent = Get< RCP<Matrix> >(coarseLevel, "Ptent");

    LocalOrdinal dofsPerNode = 0;
    LocalOrdinal colsPerNode = 0;
    if(!P.is_null())
    {
      dofsPerNode = getDofsPerNode(P);
      colsPerNode = getColsPerNode(P);
    }
    else
    {
      dofsPerNode = getDofsPerNode(Ptent);
      colsPerNode = getColsPerNode(Ptent);
    }

    bool doGraphEdges = pL.get<bool>("visualization: fine graph edges", false);

    RCP<const Teuchos::Comm<int> > comm = P->getRowMap()->getComm();

    RCP<const StridedMap> strDomainMap = Teuchos::null;
    if (P->IsView("stridedMaps") && Teuchos::rcp_dynamic_cast<const StridedMap>(P->getRowMap("stridedMaps")) != Teuchos::null)
    {
      strDomainMap = Teuchos::rcp_dynamic_cast<const StridedMap>(P->getColMap("stridedMaps"));
    }

    TEUCHOS_TEST_FOR_EXCEPTION(strDomainMap.is_null(), Exceptions::RuntimeError,
        "CoarseningVisualizationFactory requires P/Ptent to have strided domain map but it did not.");

    // TODO add support for overlapping aggregates
    //TEUCHOS_TEST_FOR_EXCEPTION(strDomainMap->getNodeNumElements() != P->getColMap()->getNodeNumElements(), Exceptions::RuntimeError,
    //                                           "CoarseningVisualization only supports non-overlapping transfers");


    // get fine level coordinate information
    Teuchos::RCP<Xpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> > coords = Get<RCP<Xpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> > >(fineLevel, "Coordinates");

    TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::as<LO>(P->getRowMap()->getNodeNumElements()) / dofsPerNode != Teuchos::as<LocalOrdinal>(coords->getLocalLength()), Exceptions::RuntimeError,
                                           "Number of fine level nodes in coordinates is inconsistent with dof based information");

    if (doGraphEdges)
    {
      fineGraph = Get<RCP<GraphBase> >(fineLevel, "Graph");
      // communicate fine level coordinates
      RCP<Import> coordImporter = Xpetra::ImportFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(coords->getMap(), fineGraph->GetImportMap());
      RCP<Xpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> > ghostedCoords = Xpetra::MultiVectorFactory<double, LocalOrdinal, GlobalOrdinal, Node>::Build(fineGraph->GetImportMap(), coords->getNumVectors());
      ghostedCoords->doImport(*coords, *coordImporter, Xpetra::INSERT);
      coords = ghostedCoords;
    }

    auto nodeMap = coords->getMap();

    int levelID = fineLevel.GetLevelID();

    VTKEmitter vtk(pL, comm->getSize(), levelID, comm->getRank(), nodeMap, Teuchos::null);

    int vizLevel = pL.get<int>("visualization: start level");
    if(vizLevel <= levelID)
    {
      auto aggStyle = pL.get<std::string>("visualization: style");
      if(!P.is_null())
      {
        AggGeometry aggGeom(P, nodeMap, comm, coords, dofsPerNode, colsPerNode, false);
        if(!aggGeom.build(aggStyle))
        {
#ifdef HAVE_MUELU_CGAL
          GetOStream(Warnings0) << "   Warning: Unrecognized agg style.\nPossible values are Point Cloud, Jacks, Convex Hulls, Alpha Hulls.\nDefaulted to Point Cloud." << std::endl;
#else
          GetOStream(Warnings0) << "   Warning: Unrecognized agg style.\nPossible values are Point Cloud, Jacks, Convex Hulls.\nDefaulted to Point Cloud." << std::endl;
#endif
        }
        vtk.writeAggGeom(aggGeom);
      }
      if(!Ptent.is_null())
      {
        AggGeometry aggGeom(Ptent, nodeMap, comm, coords, dofsPerNode, colsPerNode, true);
        if(!aggGeom.build(aggStyle))
        {
#ifdef HAVE_MUELU_CGAL
          GetOStream(Warnings0) << "   Warning: Unrecognized agg style.\nPossible values are Point Cloud, Jacks, Convex Hulls, Alpha Hulls.\nDefaulted to Point Cloud." << std::endl;
#else
          GetOStream(Warnings0) << "   Warning: Unrecognized agg style.\nPossible values are Point Cloud, Jacks, Convex Hulls.\nDefaulted to Point Cloud." << std::endl;
#endif
        }
        vtk.writeAggGeom(aggGeom);
      }
      if(doGraphEdges)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(fineGraph == Teuchos::null, Exceptions::RuntimeError,
            "Could not get information about fine graph.");
        EdgeGeometry edgeGeom(coords, fineGraph, dofsPerNode);
        edgeGeom.build();
        vtk.writeEdgeGeom(edgeGeom, true);
      }
    }

    vtk.writePVTU();

    if(comm->getRank() == 0 && pL.get<bool>("visualization: build colormap")) {
      vtk.buildColormap();
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  LocalOrdinal CoarseningVisualizationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  getDofsPerNode(const Teuchos::RCP<Matrix>& P) const
  {
    LocalOrdinal dofsPerNode = 1;
    //LocalOrdinal stridedRowOffset = 0;
    RCP<const StridedMap> strRowMap    = Teuchos::null;
    if (P->IsView("stridedMaps") && Teuchos::rcp_dynamic_cast<const StridedMap>(P->getRowMap("stridedMaps")) != Teuchos::null) {
      strRowMap = Teuchos::rcp_dynamic_cast<const StridedMap>(P->getRowMap("stridedMaps"));
      LocalOrdinal blockid       = strRowMap->getStridedBlockId();
      if (blockid > -1) {
        std::vector<size_t> stridingInfo = strRowMap->getStridingData();
        //for (size_t j = 0; j < Teuchos::as<size_t>(blockid); j++)
        //  stridedRowOffset += stridingInfo[j];
        dofsPerNode = Teuchos::as<LocalOrdinal>(stridingInfo[blockid]);
      } else {
        dofsPerNode = strRowMap->getFixedBlockSize();
      }
      GetOStream(Runtime1) << "CoarseningVisualizationFactory::Build():" << " #dofs per node = " << dofsPerNode << std::endl;
    }
    return dofsPerNode;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  LocalOrdinal CoarseningVisualizationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  getColsPerNode(const Teuchos::RCP<Matrix>& P) const
  {
    LocalOrdinal stridedColumnOffset = 0;
    //note: strDomainMap is a non-overlapping map of nodes
    RCP<const StridedMap> strDomainMap = Teuchos::null;
    if (P->IsView("stridedMaps") && Teuchos::rcp_dynamic_cast<const StridedMap>(P->getRowMap("stridedMaps")) != Teuchos::null) {
      LocalOrdinal columnsPerNode;
      strDomainMap = Teuchos::rcp_dynamic_cast<const StridedMap>(P->getColMap("stridedMaps"));
      LocalOrdinal blockid = strDomainMap->getStridedBlockId();

      if (blockid > -1) {
        std::vector<size_t> stridingInfo = strDomainMap->getStridingData();
        for (size_t j = 0; j < Teuchos::as<size_t>(blockid); j++)
          stridedColumnOffset += stridingInfo[j];
        columnsPerNode = Teuchos::as<LocalOrdinal>(stridingInfo[blockid]);
      } else {
        columnsPerNode = strDomainMap->getFixedBlockSize();
      }
      GetOStream(Runtime1) << "CoarseningVisualizationFactory::Build():" << " #columns per node = " << columnsPerNode << std::endl;
      return columnsPerNode;
    }
    return getDofsPerNode(P);
  }

} // namespace MueLu

#endif /* MUELU_AGGREGATIONEXPORTFACTORY_DEF_HPP_ */

