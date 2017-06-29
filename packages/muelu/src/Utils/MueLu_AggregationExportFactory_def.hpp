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
/*
 * MueLu_AggregationExportFactory_def.hpp
 *
 *  Created on: Feb 10, 2012
 *      Author: wiesner
 */

#ifndef MUELU_AGGREGATIONEXPORTFACTORY_DEF_HPP_
#define MUELU_AGGREGATIONEXPORTFACTORY_DEF_HPP_

#include <Xpetra_Matrix.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include "MueLu_Visualization_def.hpp"
#include "MueLu_AggregationExportFactory_decl.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_Aggregates.hpp"
#include "MueLu_Graph.hpp"
#include "MueLu_AmalgamationFactory.hpp"
#include "MueLu_AmalgamationInfo.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_Utilities.hpp"
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cmath>
//For alpha hulls (is optional feature requiring a third-party library)
#ifdef HAVE_MUELU_CGAL //Include all headers needed for both 2D and 3D fixed-alpha alpha shapes
#include "CGAL/Exact_predicates_inexact_constructions_kernel.h"
#include "CGAL/Delaunay_triangulation_2.h"
#include "CGAL/Delaunay_triangulation_3.h"
#include "CGAL/Alpha_shape_2.h"
#include "CGAL/Fixed_alpha_shape_3.h"
#include "CGAL/algorithm.h"
#endif

namespace MueLu {
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> AggregationExportFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

    std::string output_msg = "Output filename template (%TIMESTEP is replaced by \'Output file: time step\' variable,"
        "%ITER is replaced by \'Output file: iter\' variable, %LEVELID is replaced level id, %PROCID is replaced by processor id)";
    std::string output_def = "aggs_level%LEVELID_proc%PROCID.out";

    validParamList->set< RCP<const FactoryBase> >("A", Teuchos::null, "Factory for A.");
    validParamList->set< RCP<const FactoryBase> >("Coordinates", Teuchos::null, "Factory for Coordinates.");
    validParamList->set< RCP<const FactoryBase> >("Graph", Teuchos::null, "Factory for Graph.");
    validParamList->set< RCP<const FactoryBase> >("Aggregates", Teuchos::null, "Factory for Aggregates.");
    validParamList->set< RCP<const FactoryBase> >("UnAmalgamationInfo", Teuchos::null, "Factory for UnAmalgamationInfo.");
    validParamList->set< RCP<const FactoryBase> >("DofsPerNode", Teuchos::null, "Factory for DofsPerNode.");
    // CMS/BMK: Old style factory-only options.  Deprecate me.
    validParamList->set< std::string >           ("Output filename",           output_def, output_msg);
    validParamList->set< int >                   ("Output file: time step",             0, "time step variable for output file name");
    validParamList->set< int >                   ("Output file: iter",                  0, "nonlinear iteration variable for output file name");

    // New-style master list options (here are same defaults as in masterList.xml)
    validParamList->set< std::string >           ("aggregation: output filename",                    "",                    "filename for VTK-style visualization output");
    validParamList->set< int >                   ("aggregation: output file: time step",             0,                     "time step variable for output file name");// Remove me?
    validParamList->set< int >                   ("aggregation: output file: iter",                  0,                     "nonlinear iteration variable for output file name");//Remove me?
    validParamList->set<std::string>             ("aggregation: output file: agg style",             "Point Cloud",         "style of aggregate visualization for VTK output");
    validParamList->set<bool>                    ("aggregation: output file: fine graph edges",      false,                 "Whether to draw all fine node connections along with the aggregates.");
    validParamList->set<bool>                    ("aggregation: output file: coarse graph edges",    false,                 "Whether to draw all coarse node connections along with the aggregates.");
    validParamList->set<bool>                    ("aggregation: output file: build colormap",        false,                 "Whether to output a random colormap for ParaView in a separate XML file.");

    /* New + improved style parameter names (not in master list yet) */
    validParamList->set< std::string >           ("visualization: output filename",                    "viz%LEVELID",                    "filename for VTK-style visualization output");
    validParamList->set< int >                   ("visualization: output file: time step",             0,                     "time step variable for output file name");// Remove me?
    validParamList->set< int >                   ("visualization: output file: iter",                  0,                     "nonlinear iteration variable for output file name");//Remove me?
    validParamList->set<std::string>             ("visualization: style", "Point Cloud", "style of aggregate visualization for VTK output. Can be 'Point Cloud', 'Jacks', 'Convex Hulls'");
    validParamList->set<bool>                    ("visualization: build colormap",        false,       "Whether to build a random color map in a separate xml file.");
    validParamList->set<bool>                    ("visualization: fine graph edges",      false,                 "Whether to draw all fine node connections along with the aggregates.");
    return validParamList;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggregationExportFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level &fineLevel, Level &coarseLevel) const {
    Input(fineLevel, "Aggregates");         //< factory which created aggregates
    Input(fineLevel, "DofsPerNode");        //< CoalesceAndDropFactory (needed for DofsPerNode variable)
    Input(fineLevel, "UnAmalgamationInfo"); //< AmalgamationFactory (needed for UnAmalgamationInfo variable)

    const ParameterList & pL = GetParameterList();
    //Only pull in coordinates if the user explicitly requests direct VTK output, so as not to break uses of old code
    if(pL.isParameter("aggregation: output filename") && pL.get<std::string>("aggregation: output filename").length())
    {
      Input(fineLevel, "Coordinates");
      Input(fineLevel, "A");
      Input(fineLevel, "Graph");
      if(pL.get<bool>("aggregation: output file: coarse graph edges"))
      {
        Input(coarseLevel, "Coordinates");
        Input(coarseLevel, "A");
        Input(coarseLevel, "Graph");
        Input(coarseLevel, "DofsPerNode");
      }
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggregationExportFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level &fineLevel, Level &coarseLevel) const {
    using std::string;
    using std::vector;
    using Teuchos::RCP;
    using Teuchos::ArrayRCP;
    typedef VizHelpers::EdgeGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> EdgeGeometry;
    typedef VizHelpers::VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node> VTKEmitter;

    //Decide which build function to follow, based on input params
    const ParameterList& pL = GetParameterList();
    FactoryMonitor m(*this, "AggregationExportFactory", coarseLevel);
    RCP<Aggregates> aggregates      = Get< RCP<Aggregates> >(fineLevel,"Aggregates");
    RCP<const Teuchos::Comm<int> > comm = aggregates->GetMap()->getComm();
    int numProcs = comm->getSize();
    int myRank   = comm->getRank();
    string masterFilename = pL.get<std::string>("aggregation: output filename"); //filename parameter from master list
    string localFilename = pL.get<std::string>("Output filename");
    string filenameToWrite;
    bool useVTK = false;
    bool doCoarseGraphEdges = pL.get<bool>("aggregation: output file: coarse graph edges");
    bool doFineGraphEdges = pL.get<bool>("aggregation: output file: fine graph edges");
    if(masterFilename.length())
    {
      useVTK = true;
      filenameToWrite = masterFilename;
      if(filenameToWrite.rfind(".vtu") == string::npos) //Must have the file extension in the name
        filenameToWrite.append(".vtu");
      if(numProcs > 1 && filenameToWrite.rfind("%PROCID") == string::npos) //filename can't be identical between processsors in parallel problem
        filenameToWrite.insert(filenameToWrite.rfind(".vtu"), "-proc%PROCID");
    }
    else
      filenameToWrite = localFilename;
    LocalOrdinal          DofsPerNode = Get< LocalOrdinal >          (fineLevel, "DofsPerNode");
    RCP<AmalgamationInfo> amalgInfo   = Get< RCP<AmalgamationInfo> > (fineLevel, "UnAmalgamationInfo");
    RCP<Matrix> Amat = Get<RCP<Matrix> >(fineLevel, "A");
    RCP<Matrix> Ac;
    if(doCoarseGraphEdges)
    {
      Ac = Get<RCP<Matrix> >(coarseLevel, "A");
    }
    RCP<Xpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> > coords = Teuchos::null;
    RCP<Xpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> > coordsCoarse = Teuchos::null;
    RCP<GraphBase> fineGraph = Teuchos::null;
    RCP<GraphBase> coarseGraph = Teuchos::null;
    if(doFineGraphEdges)
      fineGraph = Get<RCP<GraphBase> >(fineLevel, "Graph");
    if(doCoarseGraphEdges)
      coarseGraph = Get<RCP<GraphBase> >(coarseLevel, "Graph");
    int dims = 3;
    if(useVTK) //otherwise leave null, will not be accessed by non-vtk code
    {
      coords = Get<RCP<Xpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> > >(fineLevel, "Coordinates");
      if(doCoarseGraphEdges)
        coordsCoarse = Get<RCP<Xpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> > >(coarseLevel, "Coordinates");
      dims = coords->getNumVectors();  //2D or 3D?
      if(numProcs > 1)
      {
        {
          RCP<Import> coordImporter = Xpetra::ImportFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(coords->getMap(), Amat->getColMap());
          RCP<Xpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> > ghostedCoords = Xpetra::MultiVectorFactory<double, LocalOrdinal, GlobalOrdinal, Node>::Build(Amat->getColMap(), dims);
          ghostedCoords->doImport(*coords, *coordImporter, Xpetra::INSERT);
          coords = ghostedCoords;
        }
        if(doCoarseGraphEdges)
        {
          RCP<Import> coordImporter = Xpetra::ImportFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(coordsCoarse->getMap(), Ac->getColMap());
          RCP<Xpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> > ghostedCoords = Xpetra::MultiVectorFactory<double, LocalOrdinal, GlobalOrdinal, Node>::Build(Ac->getColMap(), dims);
          ghostedCoords->doImport(*coordsCoarse, *coordImporter, Xpetra::INSERT);
          coordsCoarse = ghostedCoords;
        }
      }
    }
    GetOStream(Runtime0) << "AggregationExportFactory: DofsPerNode: " << DofsPerNode << std::endl;
    RCP<LocalOrdinalVector> vertex2AggId_vector = aggregates->GetVertex2AggId();
    RCP<LocalOrdinalVector> procWinner_vector   = aggregates->GetProcWinner();
    ArrayRCP<LocalOrdinal>  vertex2AggId        = aggregates->GetVertex2AggId()->getDataNonConst(0);
    ArrayRCP<LocalOrdinal>  procWinner          = aggregates->GetProcWinner()->getDataNonConst(0);

    // prepare for calculating global aggregate ids
    std::vector<GlobalOrdinal> numAggsGlobal (numProcs, 0);
    std::vector<GlobalOrdinal> numAggsLocal  (numProcs, 0);
    std::vector<GlobalOrdinal> minGlobalAggId(numProcs, 0);

    numAggsLocal[myRank] = aggregates->GetNumAggregates();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, numProcs, &numAggsLocal[0], &numAggsGlobal[0]);
    for (int i = 1; i < Teuchos::as<int>(numAggsGlobal.size()); ++i)
    {
      numAggsGlobal [i] += numAggsGlobal[i-1];
      minGlobalAggId[i]  = numAggsGlobal[i-1];
    }
    ArrayRCP<LO>            aggStart;
    ArrayRCP<GlobalOrdinal> aggToRowMap;
    amalgInfo->UnamalgamateAggregates(*aggregates, aggStart, aggToRowMap);
    int timeStep = pL.get< int > ("Output file: time step");
    int iter = pL.get< int > ("Output file: iter");
    filenameToWrite = VizHelpers::replaceAll(filenameToWrite, "%LEVELID",  toString(fineLevel.GetLevelID()));
    filenameToWrite = VizHelpers::replaceAll(filenameToWrite, "%TIMESTEP", toString(timeStep));
    filenameToWrite = VizHelpers::replaceAll(filenameToWrite, "%ITER",     toString(iter));
    //Proc id MUST be included in vtu filenames to distinguish them (if multiple procs)
    //In all other cases (else), including processor # in filename is optional
    string masterStem = "";
    if(useVTK)
    {
      masterStem = filenameToWrite.substr(0, filenameToWrite.rfind(".vtu"));
      masterStem = VizHelpers::replaceAll(masterStem, "%PROCID", "");
    }
    string baseFname = filenameToWrite;  //get a version of the filename string with the %PROCID token, but without substituting myRank (needed for pvtu output)
    filenameToWrite = VizHelpers::replaceAll(filenameToWrite, "%PROCID", toString(myRank));
    GetOStream(Runtime0) << "AggregationExportFactory: outputfile \"" << filenameToWrite << "\"" << std::endl;
    std::ofstream fout(filenameToWrite.c_str());
    GO numAggs = aggregates->GetNumAggregates();
    if(!useVTK)
    {
      GO indexBase = aggregates->GetMap()->getIndexBase(); // extract indexBase from overlapping map within aggregates structure. The indexBase is constant throughout the whole simulation (either 0 = C++ or 1 = Fortran)
      GO offset    = amalgInfo->GlobalOffset();            // extract offset for global dof ids
      vector<GlobalOrdinal> nodeIds;
      for (int i = 0; i < numAggs; ++i) {
        fout << "Agg " << minGlobalAggId[myRank] + i << " Proc " << myRank << ":";

        // TODO: Use k+=DofsPerNode instead of ++k and get rid of std::unique call afterwards
        for (int k = aggStart[i]; k < aggStart[i+1]; ++k) {
          nodeIds.push_back((aggToRowMap[k] - offset - indexBase) / DofsPerNode + indexBase);
        }

        // remove duplicate entries from nodeids
        std::sort(nodeIds.begin(), nodeIds.end());
        typename std::vector<GlobalOrdinal>::iterator endLocation = std::unique(nodeIds.begin(), nodeIds.end());
        nodeIds.erase(endLocation, nodeIds.end());

        // print out nodeids
        for(typename std::vector<GlobalOrdinal>::iterator printIt = nodeIds.begin(); printIt != nodeIds.end(); printIt++)
          fout << " " << *printIt;
        nodeIds.clear();
        fout << std::endl;
      }
      fout.close();
    }
    else
    {
      using std::string;
      //Make sure we have coordinates
      TEUCHOS_TEST_FOR_EXCEPTION(coords.is_null(), Exceptions::RuntimeError,"AggExportFactory could not get coordinates, but they are required for VTK output.");
      //get access to fine coord data
      typedef ArrayRCP<const double> Coords;
      Coords fx = Teuchos::arcp_reinterpret_cast<const double>(coords->getData(0));
      Coords fy = Teuchos::arcp_reinterpret_cast<const double>(coords->getData(1));
      Coords fz;
      Coords cx, cy, cz;
      if(doCoarseGraphEdges)
      {
        cx = Teuchos::arcp_reinterpret_cast<const double>(coordsCoarse->getData(0));
        cy = Teuchos::arcp_reinterpret_cast<const double>(coordsCoarse->getData(1));
      }
      if(dims == 3)
      {
        fz = Teuchos::arcp_reinterpret_cast<const double>(coords->getData(2));
        if(doCoarseGraphEdges)
          cz = Teuchos::arcp_reinterpret_cast<const double>(coordsCoarse->getData(2));
      }
      //Get the sizes of the aggregates to speed up grabbing node IDs
      string aggStyle = "Point Cloud";
      try
      {
        aggStyle = pL.get<string>("aggregation: output file: agg style"); //Let "Point Cloud" be the default style
      }
      catch(std::exception& e) {}
      auto fineMap = Amat->getMap();
      decltype(fineMap) coarseMap;
      if(!Ac.is_null())
      {
        coarseMap = Ac->getMap();
      }
      VTKEmitter vtk(pL, numProcs, fineLevel.GetLevelID(), myRank, fineMap, coarseMap);
      AggGeometry aggGeom(aggregates, comm, coords);
      if(!aggGeom.build(aggStyle))
      {
#ifdef HAVE_MUELU_CGAL
        GetOStream(Warnings0) << "   Warning: Unrecognized agg style.\nPossible values are Point Cloud, Jacks, Convex Hulls, Alpha Hulls.\nDefaulted to Point Cloud." << std::endl;
#else
        GetOStream(Warnings0) << "   Warning: Unrecognized agg style.\nPossible values are Point Cloud, Jacks, Convex Hulls.\nDefaulted to Point Cloud." << std::endl;
#endif
      }
      //write agg geom
      vtk.writeAggGeom(aggGeom);
      //do fine and coarse edges, if requested
      if(doFineGraphEdges)
      {
      //EdgeGeometry(Teuchos::RCP<CoordArray>& coords, Teuchos::RCP<GraphBase>& G, int dofs, Teuchos::RCP<Matrix> A = Teuchos::null);
        EdgeGeometry fineEdge(coords, fineGraph, DofsPerNode, Amat);
        fineEdge.build();
        vtk.writeEdgeGeom(fineEdge, true);
      }
      if(doCoarseGraphEdges)
      {
        LocalOrdinal dofsCoarse = Get<LocalOrdinal> (coarseLevel, "DofsPerNode");
        EdgeGeometry coarseEdge(coordsCoarse, coarseGraph, dofsCoarse, Ac);
        coarseEdge.build();
        vtk.writeEdgeGeom(coarseEdge, false);
      }
      //note: this will only output pvtu file if it is needed, and only process #0 will write it
      vtk.writePVTU();
      if(myRank == 0 && pL.get<bool>("aggregation: output file: build colormap"))
      {
        vtk.buildColormap();
      }
    }
  }
}

#endif /* MUELU_AGGREGATIONEXPORTFACTORY_DEF_HPP_ */

