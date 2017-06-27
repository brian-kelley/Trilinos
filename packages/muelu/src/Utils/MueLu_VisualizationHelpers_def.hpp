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

#ifndef MUELU_VISUALIZATIONHELPERS_DEF_HPP_
#define MUELU_VISUALIZATIONHELPERS_DEF_HPP_

#include <Xpetra_Matrix.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include "MueLu_VisualizationHelpers_decl.hpp"
#include "MueLu_Graph.hpp"
#include "MueLu_Monitor.hpp"
#include <vector>
#include <list>
#include <algorithm>
#include <string>

#ifdef HAVE_MUELU_CGAL
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_3.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/convex_hull_3.h>
#endif

namespace MueLu {
namespace VizHelpers {

  /* Non-member utility functions */

  inline std::string replaceAll(std::string original, std::string replaceWhat, std::string replaceWithWhat)
  {
    while(1) {
      const size_t pos = original.find(replaceWhat);
      if (pos == std::string::npos)
        break;
      original.replace(pos, replaceWhat.size(), replaceWithWhat);
    }
    return original;
  }

  inline Vec3 crossProduct(Vec3 v1, Vec3 v2)
  {
    return Vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
  }

  inline double dotProduct(Vec2 v1, Vec2 v2)
  {
    return v1.x * v2.x + v1.y * v2.y;
  }

  inline double dotProduct(Vec3 v1, Vec3 v2)
  {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  }

  inline bool isInFront(Vec3 point, Vec3 inPlane, Vec3 n)
  {
    Vec3 rel(point.x - inPlane.x, point.y - inPlane.y, point.z - inPlane.z); //position of the point relative to the plane
    return dotProduct(rel, n) > 1e-12 ? true : false;
  }

  inline bool collinear(Vec2 v1, Vec2 v2, Vec2 v3)
  {
    Vec3 diff1(v2.x - v1.x, v2.y - v1.y, 0);
    Vec3 diff2(v3.x - v2.x, v3.y - v2.y, 0);
    //diff1 and diff2 share v2, so if they are parallel then all 3 points collinear
    //cross product of parallel vectors is 0
    return mag(crossProduct(diff1, diff2)) <= 1e-8;
  }

  inline double mag(Vec2 vec)
  {
    return sqrt(dotProduct(vec, vec));
  }

  inline double mag(Vec3 vec)
  {
    return sqrt(dotProduct(vec, vec));
  }

  inline double distance2(Vec2 p1, Vec2 p2)
  {
    return mag(p1 - p2);
  }
  
  inline double distance(Vec3 p1, Vec3 p2)
  {
    return mag(p1 - p2);
  }

  inline Vec2 segmentNormal(Vec2 v) //"normal" to a 2D vector - just rotate 90 degrees to left
  {
    return Vec2(v.y, -v.x);
  }

  inline Vec3 triNormal(Vec3 v1, Vec3 v2, Vec3 v3) //normal to face of triangle (will be outward rel. to polyhedron) (v1, v2, v3 are in CCW order when normal is toward viewpoint)
  {
    return crossProduct(v2 - v1, v3 - v1).normalize();
  }

  //get minimum distance from 'point' to plane containing v1, v2, v3 (or the triangle with v1, v2, v3 as vertices)
  inline double pointDistFromTri(Vec3 point, Vec3 v1, Vec3 v2, Vec3 v3)
  {
    using namespace std;
    //get (oriented) unit normal for triangle (v1,v2,v3)
    Vec3 norm = triNormal(v1, v2, v3);
    //must normalize the normal vector
    double normScl = mag(norm);
    double rv = 0.0;
    if (normScl > 1e-8) {
      norm.x /= normScl;
      norm.y /= normScl;
      norm.z /= normScl;
      rv = fabs(dotProduct(norm, point - v1));
    } else {
      // degenerate triangle (collinear vertices)
      Vec3 test1 = v3 - v1;
      Vec3 test2 = v2 - v1;
      bool useTest1 = mag(test1) > 0.0 ? true : false;
      bool useTest2 = mag(test2) > 0.0 ? true : false;
      if(useTest1 == true) {
        double normScl1 = mag(test1);
        test1.x /= normScl1;
        test1.y /= normScl1;
        test1.z /= normScl1;
        rv = fabs(dotProduct(test1, point - v1));
      } else if (useTest2 == true) {
        double normScl2 = mag(test2);
        test2.x /= normScl2;
        test2.y /= normScl2;
        test2.z /= normScl2;
        rv = fabs(dotProduct(test2, point - v1));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, Exceptions::RuntimeError,
                 "VisualizationHelpers::pointDistFromTri: Could not determine the distance of a point to a triangle.");
      }
    }
    return rv;
  }

  Teuchos::RCP<Teuchos::ParameterList> GetVizParameterList()
  {
    auto pl = Teuchos::rcp(new Teuchos::ParameterList);
    pl->set<std::string>("visualization: output filename", "viz%LEVELID", "Output filename for VTK-formatted aggregate visualization");
    pl->set<std::string>("visualization: style", "Convex Hulls", "Style of aggregate visualization. Can be 'Point Cloud', 'Jacks', 'Convex Hulls', or 'Alpha Hulls'");
    pl->set<bool>("visualization: build colormap", false, "Whether to output a randomized colormap for use in ParaView.");
    pl->set<bool>("visualization: fine graph edges", false, "Whether to draw all fine node connections along with the aggregates.");
    return pl;
  }

  /*----------------------------*/
  /* AggGeometry implementation */
  /*----------------------------*/

  //Constructor for AggregationExportFactory
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  AggGeometry(const Teuchos::RCP<Aggregates>& aggs, const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
      const Teuchos::RCP<CoordArray>& coords)
  {
    bubbles_ = false;
    numNodes_ = coords->getLocalLength();
    numLocalAggs_ = aggs->GetNumAggregates();
    dims_ = coords->getNumVectors();
    aggs_ = aggs;
    //Set vertex positions
    {
      auto localLength = coords->getLocalLength();
      auto coordsMap = coords->getMap();
      auto x = coords->getData(0);
      auto y = coords->getData(1);
      if(dims_ == 3)
      {
        auto z = coords->getData(2);
        for(LocalOrdinal lv = 0; lv < localLength; lv++)
        {
          GlobalOrdinal gv = coordsMap->getGlobalElement(lv);
          verts_[gv] = Vec3(x[lv], y[lv], z[lv]);
        }
      }
      else
      {
        for(LocalOrdinal lv = 0; lv < localLength; lv++)
        {
          GlobalOrdinal gv = coordsMap->getGlobalElement(lv);
          verts_[gv] = Vec3(x[lv], y[lv], 0);
        }
      }
    }
    this->rank_ = comm->getRank();
    this->nprocs_ = comm->getSize();
    auto vertex2Agg_ = aggs->GetVertex2AggId()->getData(0);
    if(nprocs_ != 1)
    {
      //serial, so local agg ID == global agg ID
      firstAgg_ = 0;
    }
    else
    {
      //prepare for calculating global aggregate ids
      std::vector<GlobalOrdinal> numAggsGlobal(nprocs_);
      std::vector<GlobalOrdinal> numAggsLocal(nprocs_, 0);
      std::vector<GlobalOrdinal> minGlobalAggId(nprocs_);
      numAggsLocal[rank_] = aggs->GetNumAggregates();
      Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, nprocs_, &numAggsLocal[0], &numAggsGlobal[0]);
      for(size_t i = 1; i < numAggsGlobal.size(); ++i)
      {
        numAggsGlobal[i] += numAggsGlobal[i-1];
        minGlobalAggId[i]  = numAggsGlobal[i-1];
      }
      firstAgg_ = minGlobalAggId[rank_];
    }
    //can already compute aggOffsets_, since sizes of all aggregates known
    aggOffsets_.reserve(numLocalAggs_ + 1);
    GlobalOrdinal accum = 0;
    auto aggSizes = aggs->ComputeAggregateSizes();
    for(size_t i = 0; i < numLocalAggs_; i++)
    {
      aggOffsets_[i] = accum;
      accum += aggSizes[i];
    }
    aggOffsets_[numLocalAggs_] = accum;
    aggVerts_.resize(accum);
    //temporary array for counting vertices inserted into aggVerts_
    std::vector<int> vertCount(aggSizes.size(), 0);
    for(size_t i = 0; i < vertex2Agg_.size(); i++)
    {
      LocalOrdinal agg = vertex2Agg_[i];
      aggVerts_[aggOffsets_[agg] + vertCount[agg]] = i;
      vertCount[agg]++;
    }
  }

  //Constructor for CoarseningVisualizationFactory
  //"aggregates" described by aggVerts/aggOffsets can be overlapping
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  AggGeometry(const Teuchos::RCP<Matrix>& P, const Teuchos::RCP<const Map>& map, const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
      const Teuchos::RCP<CoordArray>& coords, LocalOrdinal dofsPerNode, LocalOrdinal colsPerNode, bool ptent)
  {
    using std::vector;
    bubbles_ = !ptent;
    //use P (possibly including non-local entries)
    //to populate aggVerts_, aggOffsets_, vertex2Agg_, firstAgg_
    dims_ = coords->getNumVectors();
    aggs_ = Teuchos::null;
    auto xVals = coords->getData(0);
    auto yVals = coords->getData(1);
    auto zVals = coords->getData(2);
    //lambdas for getting x,y,z for a local row (just for use within this function)
    auto xCoord = [&] (LocalOrdinal row) -> double {return xVals[row];};
    auto yCoord = [&] (LocalOrdinal row) -> double {return yVals[row];};
    auto zCoord =
      [&] (LocalOrdinal row) -> double
    {
      if(zVals.is_null())
        return 0;
      else
        return zVals[row];
    };
    this->rank_ = comm->getRank();
    this->nprocs_ = comm->getSize();

    RCP<const StridedMap> strDomainMap = Teuchos::null;
    if (P->IsView("stridedMaps") && Teuchos::rcp_dynamic_cast<const StridedMap>(P->getRowMap("stridedMaps")) != Teuchos::null)
    {
      strDomainMap = Teuchos::rcp_dynamic_cast<const StridedMap>(P->getColMap("stridedMaps"));
    }
    TEUCHOS_TEST_FOR_EXCEPTION(strDomainMap.is_null(), std::runtime_error, "Aggregate geometry requires the strided domain map from P/Ptent, but it was null.");
    numLocalAggs_ = strDomainMap->getNodeNumElements() / colsPerNode;
    numNodes_ = coords->getLocalLength();
    auto rowMap = P->getRowMap();
    auto colMap = P->getColMap();
    if(ptent)
    {
      //know that aggs are not overlapping, so save time by using only local row views
      vector< std::set<LocalOrdinal> > localAggs(numLocalAggs_);
      // do loop over all local rows of prolongator and extract column information
      for (LocalOrdinal row = 0; row < Teuchos::as<LocalOrdinal>(P->getRowMap()->getNodeNumElements()); ++row)
      {
        ArrayView<const LocalOrdinal> indices;
        ArrayView<const Scalar> vals;
        P->getLocalRowView(row, indices, vals);
        for(auto c = indices.begin(); c != indices.end(); ++c)
        {
          localAggs[(*c)/colsPerNode].insert(row/dofsPerNode);
        }
      }
      //fill aggOffsets
      aggOffsets_.resize(numLocalAggs_ + 1);
      LocalOrdinal accum = 0;
      for(LocalOrdinal i = 0; i < numLocalAggs_; i++)
      {
        aggOffsets_[i] = accum;
        accum += localAggs[i].size();
      }
      aggOffsets_[numLocalAggs_] = accum;
      //fill aggVerts
      aggVerts_.resize(accum);
      //number of vertices inserted into each aggregate so far
      vector<int> numInserted(accum, 0);
      for(auto& agg : localAggs)
      {
        size_t i = 0;
        for(auto v : agg)
        {
          aggVerts_[aggOffsets_[i] + numInserted[i]++] = rowMap->getGlobalElement(v);
          i++;
        }
      }
      //populate vertex positions
      for(LocalOrdinal i = 0; i < numNodes_; i++)
      {
        verts_[map->getGlobalElement(i)] = Vec3(xCoord(i), yCoord(i), zCoord(i));
      }
    }
    else  //smoothed P
    {
      //aggs can overlap and locally owned vertices can be included in off-process aggregates
      //such vert-agg pairs must be communicated to the process owning the aggregate
      //use the VertexData struct to communicate each vertex <-> agg pair
      struct VertexData
      {
        VertexData() {}
        VertexData(GlobalOrdinal aggIn, GlobalOrdinal vertexIn, double xIn, double yIn, double zIn = 0)
        {
          agg = aggIn;
          vertex = vertexIn;
          x = xIn;
          y = yIn;
          z = zIn;
        }
        GlobalOrdinal agg;
        GlobalOrdinal vertex;
        double x;
        double y;
        double z;
      };
      int rank = comm->getRank();
      int nprocs = comm->getSize();
      //Temporary but easy to work with representation of all local aggregates and the global rows they contain
      vector<std::set<GlobalOrdinal>> aggMembers;
      //First, get all local vert-agg pairs as VertexData
      vector<VertexData> localVerts;
      {
        //Do this by getting local row views of every local row and adding LocalOrdinals to a std::set representing agg
        vector< std::set<LocalOrdinal> > localAggs(numLocalAggs_);
        // do loop over all local rows of prolongator and extract column information
        for (LocalOrdinal row = 0; row < Teuchos::as<LocalOrdinal>(P->getRowMap()->getNodeNumElements()); ++row)
        {
          ArrayView<const LocalOrdinal> indices;
          ArrayView<const Scalar> valsUnused;
          P->getLocalRowView(row, indices, valsUnused);
          for(auto c = indices.begin(); c != indices.end(); ++c)
          {
            aggMembers[(*c) / colsPerNode].insert(row / dofsPerNode);
          }
        }
      }
      //next, get VertexData for owned rows in non-owned aggregates
      vector<VertexData> ghost;
      for(int i = 0; i < nprocs; i++)
      {
        //find all local nonzeros that are in aggregates owned by proc i
        vector<VertexData> toSend;
        if(i != rank)
        {
          //all VertexData to send from proc rank to proc i
          for(size_t row = 0; row < rowMap->getGlobalNumElements(); row++)
          { 
            Teuchos::ArrayView<const GlobalOrdinal> indices;
            Teuchos::ArrayView<const Scalar> values;
            P->getGlobalRowView(row, indices, values);
            //for each entry, ask the col map whether the column is owned
            for(size_t entry = 0; entry < indices.size(); entry++)
            {
              if(colMap->isNodeGlobalElement(indices[entry]))
              {
                //record this vertex-aggregate pair
                GlobalOrdinal thisAgg = entry / colsPerNode;
                GlobalOrdinal thisVert = rowMap->getGlobalElement(row / dofsPerNode);
                toSend.emplace_back(thisAgg, thisVert, xCoord(thisVert), yCoord(thisVert), zCoord(thisVert));
              }
            }
          }
        }
        //now, gather all toSend arrays to proc i from all other procs
        {
          vector<int> sendCounts(nprocs);
          vector<int> localSendCounts(nprocs, 0);
          localSendCounts[rank] = toSend.size();
          Teuchos::reduceAll<int, int>(*comm, Teuchos::REDUCE_MAX, nprocs, &localSendCounts[0], &sendCounts[0]);
          //locally get recvDispls as prefix sum from sendCounts
          vector<int> recvDispls(nprocs, 0);
          int runningTotal = 0;
          for(int j = 0; j < nprocs; j++)
          {
            recvDispls[j] = runningTotal;
            runningTotal += sendCounts[j];
          }
          //note: in the next few lines, ghost should only be modified on process i
          //running total is now the total number of VertexData to receive on proc i
          if(i == rank)
            ghost.resize(runningTotal);
          Teuchos::gatherv<int, VertexData>(&toSend[0], toSend.size(), &ghost[0], &sendCounts[0], &recvDispls[0], i, *comm);
          if(i == rank)
          {
            //add ghosts' positions and agg info
            for(VertexData& it : ghost)
            {
              verts_[it.vertex] = Vec3(it.x, it.y, it.z);
              aggMembers[it.agg].insert(it.vertex);
            }
          }
        }
      }
      //now that aggMembers is fully populated with both local and nonlocal entries, populate aggVerts_ and aggOffsets_
      GlobalOrdinal totalAggVerts = 0;
      aggOffsets_.resize(numLocalAggs_ + 1);
      for(LocalOrdinal i = 0; i < numLocalAggs_; i++)
      {
        aggOffsets_[i] = totalAggVerts;
        totalAggVerts += aggMembers[i].size();
      }
      aggOffsets_[numLocalAggs_] = totalAggVerts;
      aggVerts_.resize(totalAggVerts);
      for(LocalOrdinal i = 0; i < numLocalAggs_; i++)
      {
        //sort the global rows belonging to this agg
        vector<GlobalOrdinal> thisAggVerts;
        thisAggVerts.reserve(aggMembers[i].size());
        std::copy(aggMembers[i].begin(), aggMembers[i].end(), std::back_inserter(thisAggVerts));
        std::sort(thisAggVerts.begin(), thisAggVerts.end());
        //copy this agg's verts into correct slice of aggVerts_
        for(size_t j = 0; j < thisAggVerts.size(); j++)
        {
          aggVerts_[aggOffsets_[i] + j] = thisAggVerts[j];
        }
      }
    }
    // determine number of aggs per proc and calculate local agg offset (aka minimum global agg index)
    {
      vector<GlobalOrdinal> myLocalAggsPerProc(comm->getSize(), 0);
      vector<GlobalOrdinal> numLocalAggsPerProc(comm->getSize(), 0);
      myLocalAggsPerProc[comm->getRank()] = numLocalAggs_;
      Teuchos::reduceAll<int, GlobalOrdinal>(*comm, Teuchos::REDUCE_MAX, comm->getSize(), &myLocalAggsPerProc[0], &numLocalAggsPerProc[0]);
      firstAgg_ = 0;
      for(int i = 0; i < comm->getRank(); ++i) {
        firstAgg_ += numLocalAggsPerProc[i];
      }
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  AggGeometry(std::vector<Vec3>& coords, int dims)
  {
    numNodes_ = coords.size();
    numLocalAggs_ = 1;
    firstAgg_ = 0;
    for(size_t i = 0; i < coords.size(); i++)
    {
      verts_[i] = coords[i];
    }
    //Don't need map
    map_ = Teuchos::null;
    aggVerts_.resize(coords.size());
    for(size_t i = 0; i < coords.size(); i++)
    {
      aggVerts_[i] = i;
    }
    aggOffsets_.resize(2);
    aggOffsets_[0] = 0;
    aggOffsets_[1] = coords.size();
    aggs_ = Teuchos::null;
    dims_ = dims;
    rank_ = 0;
    nprocs_ = 1;
    bubbles_ = false;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  bool AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  build(std::string& style)
  {
    if(style == "Point Cloud")
    {
      pointCloud();
    }
    else if(style == "Jacks")
    {
      jacks();
    }
    else if(style == "Convex Hulls")
    {
#ifdef HAVE_MUELU_CGAL
      if(dims_ == 2)
        cgalConvexHulls2D();
      else
        cgalConvexHulls3D();
#else
      if(dims_ == 2)
        convexHulls2D();
      else
        convexHulls3D();
#endif
    }
#ifdef HAVE_MUELU_CGAL
    else if(style == "Alpha Hulls")
    {
      if(dims_ == 2)
        cgalAlphaHulls2D();
      else
        cgalAlphaHulls3D();
    }
#endif
    else
    {
      pointCloud();
      return false;
    }
    return true;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  pointCloud() {
    geomVerts_.reserve(aggVerts_.size());
    geomSizes_.reserve(aggVerts_.size());
    for(LocalOrdinal agg = 0; agg < numLocalAggs_; agg++)
    {
      GlobalOrdinal numVerts = aggOffsets_[agg + 1] - aggOffsets_[agg];
      for(GlobalOrdinal i = 0; i < numVerts; i++)
      {
        GlobalOrdinal v = aggVerts_[aggOffsets_[agg] + i];
        geomVerts_.emplace_back(v, agg);
        geomSizes_.push_back(1);
      }
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  computeIsRoot()
  {
    isRoot_.clear();
    if(!aggs_.is_null())
    {
      //can ask Aggregates whether each node is root
      for(LocalOrdinal i = 0; i < numNodes_; i++)
      {
        isRoot_[map_->getGlobalElement(i)] = aggs_->IsRoot(i);
      }
    }
    else
    {
      //must fake the roots, assume root of each agg is probably
      //the vertex closest to average position of vertices in agg
      for(LocalOrdinal agg = 0; agg < numLocalAggs_; agg++)
      {
        Vec3 avgPos(0, 0, 0);
        GlobalOrdinal aggSize = aggOffsets_[agg + 1] - aggOffsets_[agg];
        for(GlobalOrdinal i = aggOffsets_[agg]; i < aggOffsets_[agg + 1]; i++)
        {
          avgPos += verts_[aggVerts_[i]];
        }
        avgPos *= (1.0 / aggSize);
        //now find the aggregate that is closest to avgPos
        GlobalOrdinal centerVert;
        //distance squared between centerVert and avg
        double distSquared = 1e30;
        for(GlobalOrdinal i = aggOffsets_[agg]; i < aggOffsets_[agg + 1]; i++)
        {
          GlobalOrdinal vert = aggVerts_[i];
          //check distance between v and avg
          double ds = mag(verts_[vert] - avgPos);
          if(ds < distSquared)
          {
            distSquared = ds;
            centerVert = vert;
          }
        }
        isRoot_[centerVert] = true;
      }
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  jacks() {
    if(isRoot_.size() == 0)
    {
      computeIsRoot();
    }
    for(LocalOrdinal agg = 0; agg < numLocalAggs_; agg++)
    {
      //iterate through aggregate's vertices to find root
      GlobalOrdinal root = -1;
      GlobalOrdinal aggStart = aggOffsets_[agg];
      GlobalOrdinal aggEnd = aggOffsets_[agg + 1];
      for(GlobalOrdinal i = aggStart; i < aggEnd; i++)
      {
        GlobalOrdinal vert = aggVerts_[i];
        if(isRoot_[vert])
        {
          root = vert;
          break;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(root == -1, Exceptions::RuntimeError,
          std::string("MueLu: \"Jacks\" aggregate visualization requires a root vertex "
            "per aggregate, but aggregate ") + std::to_string(agg) + " on proc " + std::to_string(rank_) + " does not have a root.");
      //for each vert in agg (except root), make a line segment between the vert and root
      for(GlobalOrdinal i = aggStart; i < aggEnd; i++)
      {
        int vert = aggVerts_[i];
        if(vert == root)
          continue;
        geomVerts_.emplace_back(vert, agg);
        geomVerts_.emplace_back(root, agg);
        geomSizes_.push_back(2);
      }
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  convexHulls2D() {
    //giftWrap() can handle any 2D collection of points (including point/line/collinear)
    for(LocalOrdinal agg = 0; agg < numLocalAggs_; agg++)
    {
      std::vector<GlobalOrdinal> aggPoints;
      aggPoints.reserve(aggOffsets_[agg + 1] - aggOffsets_[agg]);
      for(GlobalOrdinal i = aggOffsets_[agg]; i < aggOffsets_[agg + 1]; i++)
      {
        aggPoints.push_back(aggVerts_[i]);
      }
      auto geom = giftWrap(aggPoints, verts_);
      for(size_t i = 0; i < geom.size(); i++)
      {
        geomVerts_.emplace_back(geom[i], agg);
      }
      geomSizes_.push_back(geom.size());
    }
  }

#ifdef HAVE_MUELU_CGAL
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  cgalConvexHulls2D() {
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::Point_2 Point_2;

    for(LocalOrdinal agg = 0; agg < numLocalAggs_; agg++) {
      std::vector<GlobalOrdinal> aggNodes;
      std::vector<Point_2> aggPoints;
      for(GlobalOrdinal i = aggOffsets_[agg]; i < aggOffsets_[agg + 1]; i++)
      {
        auto point = verts_[i];
        aggPoints.push_back(Point_2(point.x, point.y));
        aggNodes.push_back(i);
      }
      if(handleDegenerate(aggNodes, false))
      {
        continue;
      }
      //have a list of nodes in the aggregate
      TEUCHOS_TEST_FOR_EXCEPTION(aggNodes.size() == 0, Exceptions::RuntimeError,
               "CoarseningVisualization::doCGALConvexHulls2D: aggregate contains zero nodes!");
      // aggregate has > 2 points and isn't collinear; must run the CGAL convex hull algo
      {
        std::vector<Point_2> result;
        CGAL::convex_hull_2(aggPoints.begin(), aggPoints.end(), std::back_inserter(result));
        const double eps = 1e-8;
        // loop over all result points and find the corresponding node id
        for (size_t r = 0; r < result.size(); r++) {
          // loop over all aggregate nodes and find corresponding node id
          for(size_t l = 0; l < aggPoints.size(); l++)
          {
            if(fabs(result[r].x() - aggPoints[l].x()) < eps &&
               fabs(result[r].y() - aggPoints[l].y()) < eps)
            {
              geomVerts_.emplace_back(aggNodes[l], agg);
              break;
            }
          }
        }
        geomSizes_.push_back(result.size());
      }
    }
  }

#endif

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  convexHulls3D() {
    using std::vector;
    using std::stack;
    using std::set;
    //Use 3D quickhull algo.
    //Vector of node indices representing triangle vertices
    //Note: Calculate the hulls first since will only include point data for points in the hulls
    //Effectively the size() of vertIndices after each hull is added to it
    for(LocalOrdinal agg = 0; agg < numLocalAggs_; agg++) {
      vector<GlobalOrdinal> aggNodes; //At first, list of all nodes in the aggregate. As nodes are enclosed or included by/in hull, remove them
      aggNodes.reserve(aggOffsets_[agg + 1] - aggOffsets_[agg]);
      for(GlobalOrdinal i = aggOffsets_[agg]; i < aggOffsets_[agg + 1]; i++)
      {
        aggNodes.push_back(aggVerts_[i]);
      }
      std::cout << "Hello from convexHulls 3D, agg " << agg << " has " << aggNodes.size() << " verts.\n";
      //First, check anomalous cases
      TEUCHOS_TEST_FOR_EXCEPTION(aggNodes.size() == 0, Exceptions::RuntimeError,
               "CoarseningVisualization::doConvexHulls3D: aggregate contains zero nodes!");
      if(handleDegenerate(aggNodes, agg))
      {
        continue;
      }
      GlobalOrdinal extremeSix[6];  //verts with minx, maxx, miny, maxy, minz, maxz
      Vec3 extremeVectors[6];
      for(int i = 0; i < 6; i++)
      {
        extremeSix[i] = aggNodes[0];
        extremeVectors[i] = verts_[extremeSix[i]];
      }
      auto lessX = [&] (Vec3 v1, Vec3 v2) -> bool {
        return v1.x < v2.x || (v1.x == v2.x && v1.y < v2.y) || (v1.x == v2.x && v1.y == v2.y && v1.z < v2.z);
      };
      auto greaterX = [&] (Vec3 v1, Vec3 v2) -> bool {
        return v1.x > v2.x || (v1.x == v2.x && v1.y > v2.y) || (v1.x == v2.x && v1.y == v2.y && v1.z > v2.z);
      };
      auto lessY = [&] (Vec3 v1, Vec3 v2) -> bool {
        return v1.y < v2.y || (v1.y == v2.y && v1.z < v2.z) || (v1.y == v2.y && v1.z == v2.z && v1.x < v2.x);
      };
      auto greaterY = [&] (Vec3 v1, Vec3 v2) -> bool {
        return v1.y > v2.y || (v1.y == v2.y && v1.z > v2.z) || (v1.y == v2.y && v1.z == v2.z && v1.x > v2.x);
      };
      auto lessZ = [&] (Vec3 v1, Vec3 v2) -> bool {
        return v1.z < v2.z || (v1.z == v2.z && v1.x < v2.x) || (v1.z == v2.z && v1.x == v2.x && v1.y < v2.y);
      };
      auto greaterZ = [&] (Vec3 v1, Vec3 v2) -> bool {
        return v1.x < v2.x || (v1.z == v2.z && v1.x > v2.x) || (v1.z == v2.z && v1.x == v2.x && v1.y > v2.y);
      };
      for(size_t i = 1; i < aggNodes.size(); i++) {
        GlobalOrdinal test = aggNodes[i];
        Vec3 testVert = verts_[test];
        if(lessX(testVert, extremeVectors[0])) {
          extremeSix[0] = test;
          extremeVectors[0] = testVert;
        }
        if(greaterX(testVert, extremeVectors[1])) {
          extremeSix[1] = test;
          extremeVectors[1] = testVert;
        }
        if(lessY(testVert, extremeVectors[2])) {
          extremeSix[2] = test;
          extremeVectors[2] = testVert;
        }
        if(greaterY(testVert, extremeVectors[3])) {
          extremeSix[3] = test;
          extremeVectors[3] = testVert;
        }
        if(lessZ(testVert, extremeVectors[4])) {
          extremeSix[4] = test;
          extremeVectors[4] = testVert;
        }
        if(greaterZ(testVert, extremeVectors[5])) {
          extremeSix[5] = test;
          extremeVectors[5] = testVert;
        }
      }
      for(int i = 0; i < 6; i++) {
        extremeVectors[i] = verts_[extremeSix[i]];
      }
      double maxDist = 0;
      //ints from 0-5: which pair out of the 6 extreme points are the most distant? (indices in extremeSix and extremeVectors)
      int base1 = 0; 
      int base2 = 0;
      for(int i = 0; i < 5; i++) {
        for(int j = i + 1; j < 6; j++) {
          double thisDist = distance(extremeVectors[i], extremeVectors[j]);
          if(thisDist > maxDist)
          {
            maxDist = thisDist;
            base1 = i;
            base2 = j;
          }
        }
      }
      vector<Triangle> hull;    //each Triangle is a triplet of nodes (int IDs) that form a triangle
      //remove base1 and base2 iters from aggNodes, they are known to be in the hull
      aggNodes.erase(find(aggNodes.begin(), aggNodes.end(), extremeSix[base1]));
      aggNodes.erase(find(aggNodes.begin(), aggNodes.end(), extremeSix[base2]));
      //extremeSix[base1] and [base2] still have the Vec3 representation
      Triangle tri;
      tri.v1 = extremeSix[base1];
      tri.v2 = extremeSix[base2];
      //Now find the point that is furthest away from the line between base1 and base2
      maxDist = 0;
      //need the vectors to do "quadruple product" formula
      Vec3 b1 = extremeVectors[base1];
      std::cout << "Starting tri base 1: " << extremeSix[base1] << '\n';
      Vec3 b2 = extremeVectors[base2];
      std::cout << "Starting tri base 2: " << extremeSix[base2] << '\n';
      GlobalOrdinal thirdNode;
      for(size_t i = 0; i < aggNodes.size(); i++)
      {
        Vec3 nodePos = verts_[aggNodes[i]];
        double dist = mag(crossProduct(nodePos - b1, nodePos - b2)) / mag(b2 - b1);
        if(dist > maxDist) {
          maxDist = dist;
          thirdNode = aggNodes[i];
        }
      }
      //Now know the last node in the first triangle
      std::cout << "Starting tri third point: " << thirdNode << '\n';
      tri.v3 = thirdNode;
      hull.push_back(tri);
      Vec3 b3 = verts_[thirdNode];
      aggNodes.erase(find(aggNodes.begin(), aggNodes.end(), thirdNode));
      //Find the fourth node (most distant from triangle) to form tetrahedron
      maxDist = 0;
      GlobalOrdinal fourthVertex = -1;
      for(size_t i = 0; i < aggNodes.size(); i++)
      {
        Vec3 thisNode = verts_[aggNodes[i]];
        double nodeDist = pointDistFromTri(thisNode, b1, b2, b3);
        if(nodeDist > maxDist)
        {
          maxDist = nodeDist;
          fourthVertex = aggNodes[i];
        }
      }
      aggNodes.erase(find(aggNodes.begin(), aggNodes.end(), fourthVertex));
      std::cout << "Starting tetrahedron final point: " << fourthVertex << '\n';
      Vec3 b4 = verts_[fourthVertex];
      //Add three new triangles to hull to form the first tetrahedron
      //use tri to hold the triangle info temporarily before being added to list
      tri = hull.front();
      tri.v1 = fourthVertex;
      hull.push_back(tri);
      tri = hull.front();
      tri.v2 = fourthVertex;
      hull.push_back(tri);
      tri = hull.front();
      tri.v3 = fourthVertex;
      hull.push_back(tri);
      //now orient all four triangles so that the vertices are oriented clockwise (so getNorm_ points outward for each)
      Vec3 barycenter((b1.x + b2.x + b3.x + b4.x) / 4.0, (b1.y + b2.y + b3.y + b4.y) / 4.0, (b1.z + b2.z + b3.z + b4.z) / 4.0);
      std::cout << "  Have the starting tetrahedron, with barycenter " << barycenter.x << " " << barycenter.y << " " << barycenter.z << '\n';
      for(size_t i = 0; i < 4; i++)
      {
        Triangle& tetTri = hull[i];
        Vec3 triVert1 = verts_[tetTri.v1];
        Vec3 triVert2 = verts_[tetTri.v2];
        Vec3 triVert3 = verts_[tetTri.v3];
        Vec3 trinorm = triNormal(triVert1, triVert2, triVert3);
        Vec3 ptInPlane = (i == 0) ? b1 : b4; //first triangle definitely has b1 in it, other three definitely have b4
        if(isInFront(barycenter, ptInPlane, trinorm)) {
          //don't want the faces of the tetrahedron to face inwards (towards barycenter) so reverse orientation
          //by swapping two vertices
          auto temp = tetTri.v1;
          tetTri.v1 = tetTri.v2;
          tetTri.v2 = temp;
        }
      }
      //now, have starting polyhedron in hull (all faces are facing outwards according to getNorm_) and remaining nodes to process are in aggNodes
      //recursively, for each triangle, make a list of the points that are 'in front' of the triangle. Find the point with the maximum distance from the triangle.
      //Add three new triangles, each sharing one edge with the original triangle but now with the most distant point as a vertex. Remove the most distant point from
      //the list of all points that need to be processed. Also from that list remove all points that are in front of the original triangle but not in front of all three
      //new triangles, since they are now enclosed in the hull.
      //Construct point lists for each face of the tetrahedron individually.
      Vec3 trinorms[4]; //normals to the four tetrahedron faces, now oriented outwards
      int index = 0;
      for(auto& tetTri : hull)
      {
        Vec3 triVert1 = verts_[tetTri.v1];
        Vec3 triVert2 = verts_[tetTri.v2];
        Vec3 triVert3 = verts_[tetTri.v3];
        trinorms[index] = triNormal(triVert1, triVert2, triVert3);
        index++;
      }
      vector<GlobalOrdinal> startPoints[4];
      //scope this so that 'point' is not in function scope
      for(auto& pt : aggNodes)
      {
        Vec3 ptPos = verts_[pt];
        if(isInFront(ptPos, b1, trinorms[0]))
          startPoints[0].push_back(pt);
        else if(isInFront(ptPos, b4, trinorms[1]))
          startPoints[1].push_back(pt);
        else if(isInFront(ptPos, b4, trinorms[2]))
          startPoints[2].push_back(pt);
        else if(isInFront(ptPos, b4, trinorms[3]))
          startPoints[3].push_back(pt);
        //else: point already inside starting tetrahedron, so ignore
      }
      for(int i = 0; i < 4; i++)
      {
        std::cout << "Tetrahedron face " << i << " has " << startPoints[i].size() << " points in front.\n";
      }
      aggNodes.clear();
      aggNodes.shrink_to_fit();
      stack<int> trisToProcess;   //list of triangles still to process - done when empty
      stack<int> freelist;        //list of free indices in hull (triangles that have been deleted)
      //set up the neighbors of the first four triangles --
      //  in a tetrahedron, every triangle is a neighbor of every other triangle
      for(int i = 0; i < 4; i++)
      {
        hull[i].valid = true;
        int numNeighbors = 0;
        for(int j = 0; j < 4; j++)
        {
          if(i == j)
            continue;
          hull[i].neighbor[numNeighbors++] = j;
        }
        if(startPoints[i].size())
        {
          trisToProcess.push(i);
          hull[i].setPointList(startPoints[i]);
          startPoints[i].clear();
          startPoints[i].shrink_to_fit();
        }
        else
        {
          hull[i].frontPoints = NULL;
          hull[i].numPoints = 0;
        }
      }
      std::cout << "Done setting up for DFS\n";
      int asdf = 0;
      while(!trisToProcess.empty())
      {
        std::cout << "In DFS iter " << asdf++ << '\n';
        std::cout << "  Have " << trisToProcess.size() << " tris left to process and " << freelist.size() << " free triangles in hull\n";
        int triIndex = trisToProcess.top();
        std::cout << "Processing tri " << triIndex << '\n';
        trisToProcess.pop();
        //note: since faces was in queue, it is guaranteed to have front points 
        //therefore, it is also guaranteed to be replaced
        Triangle t = hull[triIndex];
        //mark space as free
        freelist.push(triIndex);
        hull[triIndex].valid = false;
        //note: t is a shallow copy that keeps the front point list
        //out of the point list, get the most distant point
        std::cout << "  Finding the most distant point in front...\n";
        double furthest = 0;
        int bestInd = -1;
        for(int i = 0; i < t.numPoints; i++)
        {
          double thisDist = pointDistFromTri(verts_[t.frontPoints[i]], verts_[t.v1], verts_[t.v2], verts_[t.v3]);
          if(thisDist > furthest)
          {
            bestInd = i;
            furthest = thisDist;
          }
        }
        std::cout << "  Point " << t.frontPoints[bestInd] << " is the most distant point.\n";
        //get the set of triangles adjacent to t which are visible from the furthest point
        auto farPoint = t.frontPoints[bestInd];
        vector<Triangle> visible;
        visible.push_back(t);
        for(int i = 0; i < 3; i++)
        {
          auto& nei = hull[t.neighbor[i]];
          if(pointInFront(nei, farPoint))
          {
            std::cout << "  have visible neighbor " << t.neighbor[i] << '\n';
            visible.push_back(nei);
            freelist.push(t.neighbor[i]);
            hull[t.neighbor[i]].valid = false;
          }
        }
        struct Edge
        {
          Edge() {}
          Edge(GlobalOrdinal v1i, GlobalOrdinal v2i)
          {
            if(v1i < v2i)
            {
              v1 = v1i;
              v2 = v2i;
            }
            else
            {
              v1 = v2i;
              v2 = v1i;
            }
          }
          bool operator==(const Edge& rhs)
          {
            return v1 == rhs.v1 && v2 == rhs.v2;
          }
          GlobalOrdinal v1;
          GlobalOrdinal v2;
        };
        //get a list of all edges contained in all the deleted triangles (keeping duplicates)
        vector<Edge> allEdges;
        for(auto& nei : visible)
        {
          allEdges.emplace_back(nei.v1, nei.v2);
          allEdges.emplace_back(nei.v2, nei.v3);
          allEdges.emplace_back(nei.v1, nei.v3);
        }
        std::cout << "  Have a list of " << allEdges.size() << " edges.\n";
        //sort it - already have the two vertices in each edge sorted (v1 < v2)
        std::sort(allEdges.begin(), allEdges.end(),
            [] (const Edge& e1, const Edge& e2) -> bool
            {return e1.v1 < e2.v1 || (e1.v1 == e2.v1 && e1.v2 < e2.v2);});
        //make a new list of edges that only appear in this list once (the boundary of the hole in the mesh)
        //note: edges can appear at most twice
        vector<Edge> boundary;
        for(size_t i = 0; i < allEdges.size(); i++)
        {
          if(i < allEdges.size() - 1 && allEdges[i] == allEdges[i + 1])
          {
            i++;
          }
          boundary.push_back(allEdges[i]);
        }
        std::cout << "  Have a list of " << boundary.size() << " hole boundary edges.\n";
        //for each boundary edge, form a new triangle with the farPoint - can't assign neighbors yet
        vector<int> newTris;
        std::cout << "  Creating " << boundary.size() << " new tris...\n";
        for(auto& e : boundary)
        {
          int ind;
          if(!freelist.empty())
          {
            ind = freelist.top();
            freelist.pop();
          }
          else
          {
            hull.emplace_back();
            ind = hull.size() - 1;
          }
          Triangle& newTri = hull[ind];
          newTri.valid = true;
          newTri.v1 = e.v1;
          newTri.v2 = e.v2;
          newTri.v3 = farPoint;
          newTri.clearNeighbors();
          newTris.push_back(ind);
          //make sure triangle is oriented correctly - barycenter must be behind
          if(pointDistFromTri(barycenter, verts_[newTri.v1], verts_[newTri.v2], verts_[newTri.v3]) < 0)
          {
            auto tempv = newTri.v1;
            newTri.v1 = newTri.v2;
            newTri.v2 = tempv;
          }
        }
        std::cout << "  Done creating new tris\n";
        //now set up neighbors for new triangles
        //first, build set of all triangels to search: all still-valid neighbors of visible and also all new triangles
        std::cout << "  Searching for all neighbors of new tris.\n";
        set<int> edgeSearch;
        for(auto& newTri : newTris)
        {
          edgeSearch.insert(newTri);
        }
        for(auto& visibleTri : visible)
        {
          if(hull[visibleTri.v1].valid)
            edgeSearch.insert(visibleTri.v1);
          if(hull[visibleTri.v2].valid)
            edgeSearch.insert(visibleTri.v2);
          if(hull[visibleTri.v3].valid)
            edgeSearch.insert(visibleTri.v3);
        }
        std::cout << "  Have " << edgeSearch.size() << " candidate edges making up the boundary of all new tris\n";
        for(auto newTri : newTris)
        {
          Triangle& nt = hull[newTri];
          std::cout << "    Finding neighbors for tri " << newTri << '\n';
          std::cout << "    Note: existing neighbors (should be -1): " << nt.neighbor[0] << " " << nt.neighbor[1] << " " << nt.neighbor[2] << '\n';
          for(auto& searchTri : edgeSearch)
          {
            if(nt.adjacent(hull[searchTri]) && !nt.hasNeighbor(searchTri))
            {
              std::cout << "      Adding unique neighbor " << searchTri << '\n';
              nt.addNeighbor(searchTri);
            }
          }
        }
        //now, collect all the front points from visible (deleted) triangles into one vector, and free the originals
        int totalFrontPoints = 0;
        for(auto& v : visible)
        {
          totalFrontPoints += v.numPoints;
        }
        vector<GlobalOrdinal> frontPoints;
        frontPoints.reserve(totalFrontPoints);
        for(auto& v : visible)
        {
          for(int i = 0; i < v.numPoints; i++)
          {
            frontPoints.push_back(v.frontPoints[i]);
          }
          v.freePointList();
        }
        //now, redistribute the points among all new triangles
        vector<vector<GlobalOrdinal>> triFrontPoints(newTris.size());
        for(auto frontPoint : frontPoints)
        {
          //find a new triangle such that frontPoint is in front of it
          for(size_t i = 0; i < newTris.size(); i++)
          {
            int nt = newTris[i];
            if(pointInFront(hull[nt], frontPoint))
            {
              triFrontPoints[i].push_back(frontPoint);
              break;
            }
          }
        }
        //for each triangle that has some front points, allocate its points array and add it to stack
        for(size_t i = 0; i < newTris.size(); i++)
        {
          int newTri = newTris[i];
          Triangle& nt = hull[newTri];
          if(triFrontPoints[i].size())
          {
            std::cout << "  New tri " << newTri << " has " << triFrontPoints[i].size() << " points in front.\n";
            nt.setPointList(triFrontPoints[i]);
            trisToProcess.push(newTri);
          }
        }
      }
      //hull now has all triangles that make up this hull.
      //Dump hull info into the list of all triangles for the scene.
      geomVerts_.reserve(geomVerts_.size() + 3 * hull.size());
      for(auto& hullTri : hull)
      {
        if(hullTri.valid)
        {
          geomVerts_.emplace_back(hullTri.v1, agg);
          geomVerts_.emplace_back(hullTri.v2, agg);
          geomVerts_.emplace_back(hullTri.v3, agg);
          geomSizes_.push_back(3);
        }
      }
    }
  }

#ifdef HAVE_MUELU_CGAL
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  cgalConvexHulls3D()
{
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::Point_3 Point_3;
    typedef CGAL::Polyhedron_3<K> Polyhedron_3;
    typedef std::vector<int>::iterator Iter;
    for(int agg = 0; agg < numLocalAggs_; agg++) {
      std::vector<GlobalOrdinal> aggNodes;
      std::vector<Point_3> aggPoints;
      for(GlobalOrdinal i = aggOffsets_[agg]; i < aggOffsets_[agg + 1]; i++)
      {
        auto vert = aggVerts_[i];
        aggNodes.push_back(vert);
        auto vertPos = verts_[vert];
        aggPoints.emplace_back(vertPos.x, vertPos.y, vertPos.z);
      }
      //First, check anomalous cases
      TEUCHOS_TEST_FOR_EXCEPTION(aggNodes.size() == 0, Exceptions::RuntimeError,
               "CoarseningVisualization::doCGALConvexHulls3D: aggregate contains zero nodes!");
      if(handleDegenerate(aggNodes, agg))
      {
        continue;
      }
      Polyhedron_3 result;
      CGAL::convex_hull_3( aggPoints.begin(), aggPoints.end(), result);
      // loop over all facets
      Polyhedron_3::Facet_iterator fi;
      for (fi = result.facets_begin(); fi != result.facets_end(); fi++) {
        int cntVertInAgg = 0;
        Polyhedron_3::Halfedge_around_facet_const_circulator hit = fi->facet_begin();
        do {
          const Point_3 & pp = hit->vertex()->point();
          // loop over all aggregate nodes and find corresponding node id
          for(size_t l = 0; l < aggNodes.size(); l++)
          {
            auto aggVert = verts_[aggNodes[l]];
            if(fabs(pp.x() - aggVert.x) < 1e-12 &&
               fabs(pp.y() - aggVert.y) < 1e-12 &&
               fabs(pp.z() - aggVert.z) < 1e-12)
            {
              geomVerts_.emplace_back(aggNodes[l], agg);
              cntVertInAgg++;
              break;
            }
          }
        } while (++hit != fi->facet_begin());
        geomSizes_.push_back(cntVertInAgg);
      }
    }
  }

  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  cgalAlphaHulls2D()
  {
    //CGAL setup
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::FT FT;
    typedef K::Point_2 Point;
    typedef K::Segment_2 Segment;
    typedef CGAL::Alpha_shape_vertex_base_2<K> Vb;
    typedef CGAL::Alpha_shape_face_base_2<K> Fb;
    typedef CGAL::Triangulation_data_structure_2<Vb,Fb> Tds;
    typedef CGAL::Delaunay_triangulation_2<K,Tds> Triangulation_2;
    typedef CGAL::Alpha_shape_2<Triangulation_2> Alpha_shape_2;
    typedef Alpha_shape_2::Alpha_iterator Alpha_iterator;
    typedef Alpha_shape_2::Alpha_shape_edges_iterator Alpha_shape_edges_iterator;
    using std::vector;
    using std::pair;
    for(LocalOrdinal agg = 0; agg < numLocalAggs_; agg++)
    {
      //Populate a list of Point_2 for this aggregate
      GlobalOrdinal aggStart = aggOffsets_[agg];
      GlobalOrdinal aggEnd = aggOffsets_[agg + 1];
      //Handle point and line segment case
      //because make assumption later that alpha shape is a polygon
      vector<Point> aggPoints;
      vector<GlobalOrdinal> aggNodes;
      for(GlobalOrdinal vi = aggStart; vi < aggEnd; vi++)
      {
        GlobalOrdinal v = aggVerts_[vi];
        aggNodes.push_back(v);
        auto vpos = verts_[v];
        aggPoints.push_back(Point(vpos.x, vpos.y));
      }
      if(handleDegenerate(aggNodes, agg, false))
      {
        continue;
      }
      Alpha_shape_2 hull(aggPoints.begin(), aggPoints.end());
      //Find smallest alpha value where alpha shape is one contiguous polygon
      Alpha_iterator it = hull.find_optimal_alpha(1);
      hull.set_alpha(*it);
      vector<Segment> cgalSegments;
      for(auto seg = hull.alpha_shape_edges_begin(); seg != hull.alpha_shape_edges_end(); seg++)
      {
        cgalSegments.push_back(hull.segment(*seg));
      }
      //map points back to vertices
      vector<pair<GlobalOrdinal, GlobalOrdinal>> segments;
      //get edges from hull
      for(size_t j = 0; j < cgalSegments.size(); j++)
      {
        bool foundFirst = false;
        pair<GlobalOrdinal, GlobalOrdinal> seg(-1, -1);
        for(GlobalOrdinal k = aggStart; k < aggEnd; k++)
        {
          GlobalOrdinal v = aggVerts_[k];
          auto vpos = verts_[v];
          if(cgalSegments[j][0].x() == vpos.x && cgalSegments[j][0].y() == vpos.y)
          {
            if(!foundFirst)
            {
              seg.first = v;
              foundFirst = true;
            }
            else
            {
              seg.second = v;
              segments.push_back(seg);
              break;
            }
          }
        }
        TEUCHOS_TEST_FOR_EXCEPTION(seg.first < 0 || seg.second < 0, Exceptions::RuntimeError,
            "CGAL 2D alpha shape edge has vertex that wasn't found in aggregate.");
      }
      //convert list of segments to list of vertices
      //note: segments form a closed loop, and there are >= 3 segments
      vector<LocalOrdinal> polyVerts;
      //start polyVerts with vertex from first segment that isn't in second segment
      if(segments[0].first == segments[1].first || segments[0].first == segments[1].second)
      {
        polyVerts.push_back(segments[0].second);
      }
      else
      {
        polyVerts.push_back(segments[0].first);
      }
      for(size_t s = 1; s < segments.size(); s++)
      {
        //add the vertex shared between segment s and segment s-1
        if(segments[s - 1].first == polyVerts.back())
        {
          polyVerts.push_back(segments[s - 1].second);
        }
        else
        {
          polyVerts.push_back(segments[s - 1].first);
        }
      }
      //add polygon to geometry
      for(size_t i = 0; i < polyVerts.size(); i++)
      {
        geomVerts_.emplace_back(polyVerts[i], agg);
      }
      geomSizes_.push_back(polyVerts.size());
    }
  }

  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  cgalAlphaHulls3D()
  {
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Gt;
    typedef CGAL::Tag_true Alpha_cmp_tag;
    typedef CGAL::Alpha_shape_vertex_base_3<Gt, CGAL::Default, Alpha_cmp_tag> Vb;
    typedef CGAL::Alpha_shape_cell_base_3<Gt, CGAL::Default, Alpha_cmp_tag> Fb;
    typedef CGAL::Triangulation_data_structure_3<Vb, Fb> Tds;
    typedef CGAL::Delaunay_triangulation_3<Gt, Tds> Triangulation_3;
    typedef Gt::Point_3 Point;
    typedef CGAL::Alpha_shape_3<Triangulation_3, CGAL::Tag_true> Alpha_shape_3;

    typedef typename Alpha_shape_3::Alpha_iterator Alpha_iterator;
    //typedef typename Alpha_shape_3::Cell_handle Cell_handle;
    typedef typename Alpha_shape_3::Vertex_handle Vertex_handle;
    typedef typename Alpha_shape_3::Facet Facet;
    //typedef typename Alpha_shape_3::Edge Edge;
    using std::vector;
    for(LocalOrdinal agg = 0; agg < numLocalAggs_; agg++)
    {
      vector<Point> aggPoints;
      vector<GlobalOrdinal> aggNodes;
      GlobalOrdinal aggStart = aggOffsets_[agg];
      GlobalOrdinal aggEnd = aggOffsets_[agg + 1];
      for(GlobalOrdinal i = aggStart; i < aggEnd; i++)
      {
        GlobalOrdinal v = aggVerts_[i];
        aggNodes.push_back(v);
        auto vpos = verts_[v];
        aggPoints.emplace_back(vpos.x, vpos.y, vpos.z);
      }
      Alpha_shape_3 hull(aggPoints.begin(), aggPoints.end());
      //vector<Cell_handle> cells;
      vector<Facet> facets;
      //vector<Edge> edges;
      hull.get_alpha_shape_facets(back_inserter(facets), Alpha_shape_3::REGULAR);
      for(size_t i = 0; i < facets.size(); i++)
      {
        int indices[3];
        indices[0] = (facets[i].second + 1) % 4;
        indices[1] = (facets[i].second + 2) % 4;
        indices[2] = (facets[i].second + 3) % 4;
        if(facets[i].second % 2 == 0)
        {
          std::swap(indices[0], indices[1]);
        }
        Point facetPts[3];
        facetPts[0] = facets[i].first->vertex(indices[0])->point();
        facetPts[1] = facets[i].first->vertex(indices[1])->point();
        facetPts[2] = facets[i].first->vertex(indices[2])->point();
        //add triangles in terms of node indices
        //this assumes that no two vertices in an aggregate have same position
        for(size_t ap = 0; ap < aggPoints.size(); ap++)
        {
          for(int j = 0; j < 3; j++)
          {
            if(facetPts[j] == aggPoints[ap])
            {
              geomVerts_.emplace_back(aggVerts_[aggStart + ap], agg);
            }
          }
        }
        geomSizes_.push_back(3);
      }
    }
  }

#endif
  
  template <class GlobalOrdinal>
  std::vector<GlobalOrdinal> giftWrap(std::vector<GlobalOrdinal>& points, std::map<GlobalOrdinal, Vec3>& verts)
  {
    if(points.size() < 3) {
      //aggregate is a point or line segment, so just use all its points as the geometry
      return points;
    }
    //first, get the normal to the plane containing the points
    Vec3 normal;
    {
      Vec3 v1 = verts[points[0]];
      Vec3 v2 = verts[points[1]];
      Vec3 v3 = verts[points[2]];
      size_t i = 3;
      //make sure triangle is not degenerate when getting its normal
      while(i < points.size() && mag(crossProduct(v1 - v2, v1 - v3)) < 1e-8)
      {
        v3 = verts[points[i]];
        i++;
      }
      normal = crossProduct(v1 - v2, v1 - v3).normalize();
    }
    //Will project all 3D points down to a 2D coordinate plane (either xy, yz or xz)
    //ignoreDim is which dimension (0 = x, etc.) to ignore
    //ignoreDim is given by whichever component of the normal is largest
    int ignoreDim;
    if(fabs(normal.x) >= fabs(normal.y) && fabs(normal.x) >= fabs(normal.z))
      ignoreDim = 0;
    else if(fabs(normal.y) >= fabs(normal.x) && fabs(normal.y) >= fabs(normal.z))
      ignoreDim = 1;
    else
      ignoreDim = 2;
    auto proj = [&] (Vec3 v) -> Vec2
    {
      switch(ignoreDim)
      {
        case 0:
          return Vec2(v.y, v.z);
        case 1:
          return Vec2(v.x, v.z);
        case 2:
          return Vec2(v.x, v.y);
      }
      return Vec2();
    };
    auto pos = [&] (GlobalOrdinal g) -> Vec2
    {
      return proj(verts[g]);
    };
    //check if all points are collinear, need to explicitly draw a line in that case.
    {
      //find first pair (v1, v2) of vertices that are not the same, in case multiple nodes share positions
      GlobalOrdinal v1 = points[0];
      GlobalOrdinal v2 = points[1];
      GlobalOrdinal iter;
      for(iter = 2; iter < points.size(); iter++)
      {
        if(pos(v1) != pos(v2))
          break;
        v2 = points[iter];
      }
      if(pos(v1) == pos(v2))
      {
        //all vertices in agg have same position (would be very strange corner case)
        return std::vector<GlobalOrdinal>(1, points[0]);
      }
      //check if the rest of the vertices are collinear with v1, v2
      Vec2 vec1 = pos(v1);
      Vec2 vec2 = pos(v2);
      bool allCollinear = true;
      for(; iter < points.size(); iter++)
      {
        Vec2 vec3 = pos(points[iter]);
        if(!collinear(vec1, vec2, vec3))
        {
          allCollinear = false;
          break;
        }
      }
      if(allCollinear)
      {
        //all vertices collinear
        //find min and max coordinates in agg and make line segment between them
        GlobalOrdinal minVert = points[0];
        GlobalOrdinal maxVert = points[0];
        Vec2 minPos = pos(minVert);
        Vec2 maxPos = pos(maxVert);
        for(size_t i = 1; i < points.size(); i++)
        {
          GlobalOrdinal v = points[i];
          //compare X first and then use Y as a tiebreak (will always find 2 most distant points on the line)
          Vec2 vPos = pos(v);
          if(vPos.x < minPos.x || (vPos.x == minPos.x && vPos.y < minPos.y)) 
          {
            minVert = v;
            minPos = vPos;
          }
          if(vPos.x > minPos.x || (vPos.x == minPos.x && vPos.y > minPos.y)) 
          {
            maxVert = v;
            maxPos = vPos;
          }
        }
        std::vector<GlobalOrdinal> lineSeg(2);
        lineSeg[0] = minVert;
        lineSeg[1] = maxVert;
        return lineSeg;
      }
    }
    if(points.size() == 3)
    {
      //all triangles are convex, so don't need to run algorithm
      return points;
    }
    //use "gift wrap" algorithm to find the convex hull
    //first, find vert with minimum x (and use min y as a tiebreak)
    GlobalOrdinal minVert = points[0];
    Vec2 minPos = pos(minVert);
    for(GlobalOrdinal i = 1; i < points.size(); i++)
    {
      GlobalOrdinal test = points[i];
      Vec2 testPos = pos(test);
      if(testPos.x < minPos.x || (testPos.x == minPos.x && testPos.y < minPos.y))
      {
        minVert = test;
        minPos = testPos;
      }
    }
    std::vector<GlobalOrdinal> loop;
    //start loop with minVert
    loop.push_back(minVert);
    //loop until a closed loop (with > 1 vertices) has been formed
    while(loop.size() == 1 || loop.front() != loop.back())
    {
      //find another vertex "ray" in the aggregate (one that is not the same as loopIter)
      //the geometric ray that sweeps to the left goes from loopIter to ray
      GlobalOrdinal loopIter = loop.back();
      GlobalOrdinal ray = points[0];
      if(ray == loopIter)
      {
        ray = points[1];
      }
      //sweep through all vertices, and if one is on the left side of loopIter->ray, change ray to it
      //"is point left of ray?" is answered by segmentNormal and dot prod
      Vec2 norm = segmentNormal(pos(ray) - pos(loopIter));
      double rayLen = mag(pos(ray) - pos(loopIter));
      for(size_t i = 0; i < points.size(); i++)
      {
        if(points[i] == loopIter || points[i] == ray)
          continue;
        double dotProd = dotProduct(norm, pos(points[i]) - pos(loopIter));
        double thisLen = mag(pos(points[i]) - pos(loopIter));
        //use point i as loopIter if it is left of the ray OR if it is exactly on the ray but further away from loopIter
        if(dotProd > 0 || (dotProd == 0 && thisLen > rayLen))
        {
          //update loopIter to point[i]
          rayLen = thisLen;
          ray = points[i];
          norm = segmentNormal(pos(ray) - pos(loopIter));
        }
      }
      loop.push_back(ray);
    }
    //first point same as last point: remove last because don't want repeats
    loop.pop_back();
    //loop now contains the vertex loop representing convex hull (with none repeated)
    return loop;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  bool AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  handleDegenerate(std::vector<GlobalOrdinal>& aggNodes, int agg, bool is3D)
  {
    if(aggNodes.size() == 1)
    {
      geomVerts_.emplace_back(aggNodes[0], agg);
      geomSizes_.push_back(1);
      return true;
    }
    else if(aggNodes.size() == 2)
    {
      geomVerts_.emplace_back(aggNodes[0], agg);
      geomVerts_.emplace_back(aggNodes[1], agg);
      geomSizes_.push_back(2);
      return true;
    }
    //check for collinearity
    bool areCollinear = true;
    {
      Vec3 firstVec = verts_[aggNodes[0]];
      Vec3 comp;
      {
        Vec3 secondVec = verts_[aggNodes[1]]; //cross this with other vectors to compare
        comp = secondVec - firstVec;
      }
      for(size_t i = 2; i < aggNodes.size(); i++)
      {
        Vec3 cross = crossProduct(verts_[aggNodes[i]] - firstVec, comp);
        if(mag(cross) > 1e-8) {
          areCollinear = false;
          break;
        }
      }
    }
    if(areCollinear)
    {
      //find the endpoints of segment describing all the points
      //compare x, if tie compare y, if still tie compare z
      GlobalOrdinal minVert = aggNodes[0];
      Vec3 minPos = verts_[minVert];
      GlobalOrdinal maxVert = aggNodes[0];
      Vec3 maxPos = verts_[maxVert];
      for(size_t i = 1; i < aggNodes.size(); i++)
      {
        Vec3 thisVert = verts_[aggNodes[i]];
        if(thisVert.x < minPos.x ||
            (thisVert.x == minPos.x && thisVert.y < minPos.y) ||
            (thisVert.x == minPos.x && thisVert.y == minPos.y && thisVert.z < minPos.z))
        {
          minVert = aggNodes[i];
          minPos = thisVert;
        }
        if(thisVert.x > minPos.x ||
            (thisVert.x == minPos.x && thisVert.y > minPos.y) ||
            (thisVert.x == minPos.x && thisVert.y == minPos.y && thisVert.z > minPos.z))
        {
          maxVert = aggNodes[i];
          maxPos = thisVert;
        }
      }
      geomVerts_.emplace_back(minVert, agg);
      geomVerts_.emplace_back(maxVert, agg);
      geomSizes_.push_back(2);
      return true;
    }
    if(!is3D)
    {
      //all 2D aggregates are coplanar
      return false;
    }
    bool areCoplanar = true;
    {
      //number of points is known to be >= 3 (not a point or line segment)
      Vec3 v1 = verts_[aggNodes[0]];
      Vec3 v2 = verts_[aggNodes[1]];
      Vec3 v3 = verts_[aggNodes[2]];
      //Make sure the first three points aren't also collinear (need a non-degenerate triangle to get a normal)
      for(size_t i = 3; i < aggNodes.size(); i++)
      {
        if(mag(crossProduct(v1 - v2, v1 - v3)) >= 1e-8)
          break;
        v3 = verts_[aggNodes[i]];
      }
      for(size_t i = 3; i < aggNodes.size(); i++)
      {
        Vec3 pt = verts_[aggNodes[i]];
        if(fabs(pointDistFromTri(pt, v1, v2, v3)) > 1e-8) {
          areCoplanar = false;
          break;
        }
      }
      if(areCoplanar) {
        //do 2D convex hull
        auto convhull2d = giftWrap(aggNodes, verts_);
        for(size_t i = 0; i < convhull2d.size(); i++)
        {
          geomVerts_.emplace_back(convhull2d[i], agg);
        }
        geomSizes_.push_back(convhull2d.size());
        return true;
      }
    }
    //not a point, line segment, collinear or coplanar
    return false;
  }

  /*-----------------------------*/
  /* EdgeGeometry implementation */
  /*-----------------------------*/

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  EdgeGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  EdgeGeometry(Teuchos::RCP<CoordArray>& coords, Teuchos::RCP<GraphBase>& G, int dofs, Teuchos::RCP<Matrix> A)
  {
    G_ = G;
    A_ = A;
    dofs_ = dofs;
    //in verts_, set position of each global row represented in coords (note: some may be ghosted)
    auto coordMap = coords->getMap();
    auto xCol = coords->getData(0);
    auto yCol = coords->getData(0);
    if(coords->getNumVectors() == 3)
    {
      auto zCol = coords->getData(0);
      for(LocalOrdinal i = 0; i < coords->getLocalLength(); i++)
      {
        verts_[coordMap->getGlobalElement(i)] = Vec3(xCol[i], yCol[i], zCol[i]);
      }
    }
    else
    {
      for(LocalOrdinal i = 0; i < coords->getLocalLength(); i++)
      {
        verts_[coordMap->getGlobalElement(i)] = Vec3(xCol[i], yCol[i], 0);
      }
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void EdgeGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::build()
  {
    using namespace std;
    ArrayView<const Scalar> values;
    ArrayView<const LocalOrdinal> neighbors;
    if(!A_->getRowMap()->isDistributed())
    {
      //Matrix is locally replicated - global row views available locally
      ArrayView<const GlobalOrdinal> indices;
      for(GlobalOrdinal globRow = 0; globRow < A_->getGlobalNumRows(); globRow++)
      {
        if(dofs_ == 1)
          A_->getGlobalRowView(globRow, indices, values);
        neighbors = G_->getNeighborVertices(globRow);
        size_t gEdge = 0;
        size_t aEdge = 0;
        while(gEdge != neighbors.size())
        {
          if(dofs_ == 1)
          {
            if(neighbors[gEdge] == indices[aEdge])
            {
              //graph and matrix both have this edge, wasn't filtered
              vertsNonFilt_.push_back(globRow);
              vertsNonFilt_.push_back(neighbors[gEdge]);
              gEdge++;
              aEdge++;
            }
            else
            {
              //graph contains an edge at gEdge which was filtered from A
              vertsFilt_.push_back(globRow);
              vertsFilt_.push_back(neighbors[gEdge]);
              gEdge++;
            }
          }
          else 
          {
            //for multiple DOF problems, don't try to detect filtered edges and ignore A
            //TODO bmk: do detect them
            vertsNonFilt_.push_back(globRow);
            vertsNonFilt_.push_back(neighbors[gEdge]);
            gEdge++;
          }
        }
      }
    }
    else
    {
      ArrayView<const LocalOrdinal> indices;
      for(LocalOrdinal locRow = 0; locRow < A_->getNodeNumRows(); locRow++)
      {
        if(dofs_ == 1)
          A_->getLocalRowView(locRow, indices, values);
        neighbors = G_->getNeighborVertices(locRow);
        //Add those local indices (columns) to the list of connections (which are pairs of the form (localM, localN))
        size_t gEdge = 0;
        size_t aEdge = 0;
        while(gEdge != neighbors.size())
        {
          if(dofs_ == 1)
          {
            if(neighbors[gEdge] == indices[aEdge])
            {
              vertsNonFilt_.push_back(locRow);
              vertsNonFilt_.push_back(neighbors[gEdge]);
              gEdge++;
              aEdge++;
            }
            else
            {
              vertsFilt_.push_back(locRow);
              vertsFilt_.push_back(neighbors[gEdge]);
              gEdge++;
            }
          }
          else
          {
            vertsNonFilt_.push_back(locRow);
            vertsNonFilt_.push_back(neighbors[gEdge]);
            gEdge++;
          }
        }
      }
    }
  }

  /*---------------------------*/
  /* VTKEmitter implementation */
  /*---------------------------*/

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  VTKEmitter(const Teuchos::ParameterList& pL, int numProcs, int level, int rank,
      Teuchos::RCP<const Map> fineMap, Teuchos::RCP<const Map> coarseMap)
  {
    baseName_    = pL.get<std::string>("visualization: output filename");
    int timeStep = pL.get<int>("visualization: output file: time step");
    int iter     = pL.get<int>("visualization: output file: iter");
    //strip vtu file extension
    if(baseName_.rfind(".vtu") == baseName_.length() - 4)
    {
      baseName_ = baseName_.substr(0, baseName_.length() - 4);
    }
    //if user didn't add the %PROCID but there are multiple processes,
    //add it to disambiguate files from different processes
    if(numProcs > 1 && baseName_.rfind("%PROCID") == std::string::npos)
    {
      baseName_ += "-proc%PROCID";
    }
    baseName_ = replaceAll(baseName_, "%LEVELID", toString(level));
    baseName_ = replaceAll(baseName_, "%TIMESTEP", toString(timeStep));
    baseName_ = replaceAll(baseName_, "%ITER", toString(iter));
    rank_ = rank;
    fineMap_ = fineMap;
    coarseMap_ = coarseMap;
    didFineEdges_ = false;
    didCoarseEdges_ = false;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  getAggFilename(int proc)
  {
    return replaceAll(baseName_, "%PROCID", std::to_string(proc)) + ".vtu";
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  getBubbleFilename(int proc)
  {
    return replaceAll(baseName_, "%PROCID", std::to_string(proc)) + "-bubbles.vtu";
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  getFineEdgeFilename(int proc)
  {
    return replaceAll(baseName_, "%PROCID", std::to_string(proc)) + "-finegraph.vtu";
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  getCoarseEdgeFilename(int proc)
  {
    return replaceAll(baseName_, "%PROCID", std::to_string(proc)) + "-coarsegraph.vtu";
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  getPVTUFilename()
  {
    return replaceAll(baseName_, "%PROCID", "") + "-master.pvtu";
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeAggGeom(AggGeom& ag)
  {
    //note: makeUnique modifies ag.geomVerts_ in place, but OK because it won't be used again
    auto uniqueVerts = getUniqueAggGeom(ag.geomVerts_);
    std::ofstream fout(getAggFilename(rank_));
    writeOpening(fout, uniqueVerts.size(), ag.geomSizes_.size());
    writeAggNodes(fout, uniqueVerts);
    writeAggData(fout, uniqueVerts, ag.firstAgg_);
    writeCoordinates(fout, uniqueVerts, ag.verts_);
    writeAggCells(fout, ag.geomVerts_, ag.geomSizes_);
    writeClosing(fout);
    fout.close();
    if(ag.bubbles_)
    {
      didBubbles_ = true;
    }
    else
    {
      didAggs_ = true;
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeEdgeGeom(EdgeGeom& eg, bool fine)
  {
    auto uniqueVertsNonFilt = getUniqueEdgeGeom(eg.vertsNonFilt_);
    auto uniqueVertsFilt = getUniqueEdgeGeom(eg.vertsFilt_);
    auto positions = eg.verts_;
    //get filename
    std::string filename = fine ? getFineEdgeFilename(rank_) : getCoarseEdgeFilename(rank_);
    std::ofstream fout(filename);
    writeOpening(fout, uniqueVertsNonFilt.size() + uniqueVertsFilt.size(),
        (eg.vertsNonFilt_.size() + eg.vertsFilt_.size()) / 2);
    writeEdgeNodes(fout, uniqueVertsNonFilt, uniqueVertsFilt);
    writeEdgeData(fout, uniqueVertsNonFilt.size(), uniqueVertsNonFilt.size());
    writeCoordinates(fout, uniqueVertsNonFilt, uniqueVertsFilt, positions);
    writeEdgeCells(fout, eg.vertsNonFilt_, eg.vertsFilt_, uniqueVertsNonFilt.size());
    writeClosing(fout);
    fout.close();
    if(fine)
    {
      didFineEdges_ = true;
    }
    else
    {
      didCoarseEdges_ = true;
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeOpening(std::ofstream& fout, size_t numVerts, size_t numCells) {
    std::string indent = "      ";
    fout << "<!--MueLu Aggregates/Coarsening Visualization-->" << std::endl;
    fout << "<VTKFile type=\"UnstructuredGrid\" byte_order=\"LittleEndian\">" << std::endl;
    fout << "  <UnstructuredGrid>" << std::endl;
    fout << "    <Piece NumberOfPoints=\"" << numVerts << "\" NumberOfCells=\"" << numCells << "\">" << std::endl;
    fout << "      <PointData Scalars=\"Node Aggregate Processor\">" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeAggNodes(std::ofstream& fout, std::vector<GeomPoint>& uniqueVerts)
  {
    std::string indent = "      ";
    fout << "        <DataArray type=\"Int32\" Name=\"Node\" format=\"ascii\">" << std::endl;
    indent = "          ";
    for(size_t i = 0; i < uniqueVerts.size(); i++)
    {
      fout << uniqueVerts[i].vert << " "; //if all nodes are on this processor, do not map from local to global
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeEdgeNodes(std::ofstream& fout, std::vector<GlobalOrdinal>& uniqueVertsNonFilt, std::vector<GlobalOrdinal>& uniqueVertsFilt)
  {
    std::string indent = "      ";
    fout << "        <DataArray type=\"Int32\" Name=\"Node\" format=\"ascii\">" << std::endl;
    indent = "          ";
    size_t i = 0;
    for(auto& v : uniqueVertsNonFilt)
    {
      fout << v << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
      i++;
    }
    for(auto& v : uniqueVertsFilt)
    {
      fout << v << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
      i++;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeAggData(std::ofstream& fout, std::vector<GeomPoint>& uniqueVerts, GlobalOrdinal firstAgg)
  {
    std::string indent = "          ";
    fout << "        <DataArray type=\"Int32\" Name=\"Aggregate\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < uniqueVerts.size(); i++)
    {
      fout << firstAgg + uniqueVerts[i].agg << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"Processor\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < uniqueVerts.size(); i++)
    {
      fout << rank_ << " ";
      if(i % 20 == 19)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "      </PointData>" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeEdgeData(std::ofstream& fout, size_t vertsNonFilt, size_t vertsFilt)
  {
    int contrast1 = -1;
    int contrast2 = -2;
    std::string indent = "          ";
    fout << "        <DataArray type=\"Int32\" Name=\"Aggregate\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < vertsNonFilt; i++) 
    {
      fout << contrast1 << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    for(size_t i = 0; i < vertsFilt; i++) 
    {
      fout << contrast2 << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"Processor\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < vertsNonFilt; i++) 
    {
      fout << contrast1 << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    for(size_t i = 0; i < vertsFilt; i++) 
    {
      fout << contrast2 << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "      </PointData>" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeCoordinates(std::ofstream& fout, std::vector<GeomPoint>& uniqueVerts, std::map<GlobalOrdinal, Vec3>& positions)
  {
    std::string indent = "      ";
    fout << "      <Points>" << std::endl;
    fout << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    fout << indent;
    size_t i = 0;
    for(auto& v : uniqueVerts)
    {
      auto vpos = positions[v.vert];
      fout << vpos.x << ' ' << vpos.y << ' ' << vpos.z << ' ';
      //write 3 coordinates per line
      if(i % 3 == 2)
        fout << std::endl << indent;
      i++;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "      </Points>" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeCoordinates(std::ofstream& fout,
      std::vector<GlobalOrdinal>& uniqueVertsNonFilt, std::vector<GlobalOrdinal>& uniqueVertsFilt,
      std::map<GlobalOrdinal, Vec3>& positions)
  {
    std::string indent = "      ";
    fout << "      <Points>" << std::endl;
    fout << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    fout << indent;
    size_t i = 0;
    for(auto& v : uniqueVertsNonFilt)
    {
      auto vpos = positions[v];
      fout << vpos.x << ' ' << vpos.y << ' ' << vpos.z << ' ';
      //write 3 coordinates per line
      if(i % 3 == 2)
        fout << std::endl << indent;
      i++;
    }
    for(auto& v : uniqueVertsFilt)
    {
      auto vpos = positions[v];
      fout << vpos.x << ' ' << vpos.y << ' ' << vpos.z << ' ';
      //write 3 coordinates per line
      if(i % 3 == 2)
        fout << std::endl << indent;
      i++;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "      </Points>" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeAggCells(std::ofstream& fout, std::vector<GeomPoint>& geomVerts, std::vector<int>& geomSizes)
  {
    std::string indent = "      ";
    fout << "      <Cells>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < geomVerts.size(); i++)
    {
      fout << geomVerts[i].vert << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
    fout << indent;
    int accum = 0;
    for(size_t i = 0; i < geomSizes.size(); i++)
    {
      accum += geomSizes[i];
      fout << accum << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"types\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < geomSizes.size(); i++)
    {
      switch(geomSizes[i])
      {
        case 1:
          fout << "1 "; //Point
          break;
        case 2:
          fout << "3 "; //Line
          break;
        case 3:
          fout << "5 "; //Triangle
          break;
        default:
          fout << "7 "; //Polygon
      }
      if(i % 30 == 29)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "      </Cells>" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeEdgeCells(std::ofstream& fout, std::vector<GlobalOrdinal>& vertsNonFilt, std::vector<GlobalOrdinal>& vertsFilt, size_t numUniqueNonFilt)
  {
    std::string indent = "      ";
    fout << "      <Cells>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
    fout << indent;
    size_t totalEdges = (vertsNonFilt.size() + vertsFilt.size()) / 2;
    for(size_t i = 0; i < vertsNonFilt.size(); i++)
    {
      fout << vertsNonFilt[i] << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    for(size_t i = 0; i < vertsFilt.size(); i++)
    {
      fout << numUniqueNonFilt + vertsFilt[i] << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
    fout << indent;
    int accum = 0;
    for(size_t i = 0; i < totalEdges; i++)
    {
      accum += 2;
      fout << accum << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
      i++;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"types\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < totalEdges; i++)
    {
      fout << "3 ";
      if(i % 30 == 29)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "      </Cells>" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeClosing(std::ofstream& fout)
  {
    fout << "    </Piece>" << std::endl;
    fout << "  </UnstructuredGrid>" << std::endl;
    fout << "</VTKFile>" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writePVTU() {
    //don't let multiple processes try to write same file
    if(rank_ == 0)
    {
      //decide whether PVTU is required
      if(nprocs_ == 0)
      {
        int numGeoms = 0;
        if(didAggs_)
          numGeoms++;
        if(didBubbles_)
          numGeoms++;
        if(didFineEdges_)
          numGeoms++;
        if(didCoarseEdges_)
          numGeoms++;
        if(numGeoms <= 1)
        {
          //no reason to emit .pvtu if only one .vtu, because user can just open it directly
          return;
        }
      }
      std::ofstream pvtu(getPVTUFilename());
      //If using vtk, filenameToWrite now contains final, correct ***.vtu filename (for the current proc)
      //So the root proc will need to use its own filenameToWrite to make a list of the filenames of all other procs to put in
      //pvtu file.
      pvtu << "<VTKFile type=\"PUnstructuredGrid\" byte_order=\"LittleEndian\">" << std::endl;
      pvtu << "  <PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
      pvtu << "    <PPointData Scalars=\"Node Aggregate Processor\">" << std::endl;
      pvtu << "      <PDataArray type=\"Int32\" Name=\"Node\"/>" << std::endl;
      pvtu << "      <PDataArray type=\"Int32\" Name=\"Aggregate\"/>" << std::endl;
      pvtu << "      <PDataArray type=\"Int32\" Name=\"Processor\"/>" << std::endl;
      pvtu << "    </PPointData>" << std::endl;
      pvtu << "    <PPoints>" << std::endl;
      pvtu << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>" << std::endl;
      pvtu << "    </PPoints>" << std::endl;
      //there are 4 sets of geometry, each with one output file per MPI process
      //can have any combination of these
      if(didAggs_)
      {
        for(int i = 0; i < nprocs_; i++)
          pvtu << "    <Piece Source=\"" << getAggFilename(i) << "\"/>" << std::endl;
      }
      if(didBubbles_)
      {
        for(int i = 0; i < nprocs_; i++)
          pvtu << "    <Piece Source=\"" << getBubbleFilename(i) << "\"/>" << std::endl;
      }
      if(didFineEdges_)
      {
        for(int i = 0; i < nprocs_; i++)
          pvtu << "    <Piece Source=\"" << getFineEdgeFilename(i) << "\"/>" << std::endl;
      }
      if(didCoarseEdges_)
      {
        for(int i = 0; i < nprocs_; i++)
          pvtu << "    <Piece Source=\"" << getCoarseEdgeFilename(i) << "\"/>" << std::endl;
      }
      pvtu << "  </PUnstructuredGrid>" << std::endl;
      pvtu << "</VTKFile>" << std::endl;
      pvtu.close();
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::buildColormap() {
    try {
      std::ofstream color("random-colormap.xml");
      color << "<ColorMap name=\"MueLu-Random\" space=\"RGB\">" << std::endl;
      //Give -1, and -2 distinctive colors (so that the style functions can have constrasted geometry types)
      //Do red + orange to constrast with cool color spectrum for aggregates
      color << "  <Point x=\"" << -1 << "\" o=\"1\" r=\"1\" g=\"0.2\" b=\"0\"/>" << std::endl;
      color << "  <Point x=\"" << -2 << "\" o=\"1\" r=\"1\" g=\"0.6\" b=\"0\"/>" << std::endl;
      srand(time(NULL));
      for(int i = 0; i < 5000; i += 4) {
        color << "  <Point x=\"" << i << "\" o=\"1\" r=\"" << (rand() % 100) / 256.0 << "\" g=\"" << (rand() % 256) / 256.0 << "\" b=\"" << (rand() % 256) / 256.0 << "\"/>" << std::endl;
      }
      color << "</ColorMap>" << std::endl;
      color.close();
    }
    catch(std::exception& e) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Exceptions::RuntimeError,
               "VisualizationHelpers::buildColormap: Error while building colormap file: " << e.what());
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node> 
  std::vector<GeometryPoint<GlobalOrdinal>> VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getUniqueAggGeom(std::vector<GeometryPoint<GlobalOrdinal>>& geomPoints)
  {
    auto geomPointLess = [] (const GeomPoint& lhs, const GeomPoint& rhs) -> bool
    {
      return lhs.vert < rhs.vert || (lhs.vert == rhs.vert && lhs.agg < rhs.agg);
    };
    auto geomPointsEqual = [] (const GeomPoint& lhs, const GeomPoint& rhs) -> bool
    {
      return lhs.vert == rhs.vert && lhs.agg == rhs.agg;
    };
    auto copy = geomPoints;
    std::sort(copy.begin(), copy.end(), geomPointLess);
    auto end = std::unique(copy.begin(), copy.end(), geomPointsEqual);
    copy.erase(end, copy.end());
    //now replace all vert values in geomPoints with indices into copy
    for(auto& gp : geomPoints)
    {
      auto findPos = std::lower_bound(copy.begin(), copy.end(), gp, geomPointLess);
      gp.vert = findPos - copy.begin();
    }
    return copy;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  std::vector<GlobalOrdinal> VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getUniqueEdgeGeom(std::vector<GlobalOrdinal>& edges)
  {
    auto copy = edges;
    std::sort(copy.begin(), copy.end());
    auto end = std::unique(copy.begin(), copy.end());
    copy.erase(end, copy.end());
    for(auto& g : edges)
    {
      auto findPos = std::lower_bound(copy.begin(), copy.end(), g);
      g = findPos - copy.begin();
    }
    return copy;
  }

} // namespace VizHelpers
} // namespace MueLu

#endif /* MUELU_VISUALIZATIONHELPERS_DEF_HPP_ */

