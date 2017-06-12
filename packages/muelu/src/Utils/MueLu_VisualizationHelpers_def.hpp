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
#include <CGAL/convex_hull_2.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/convex_hull_3.h>
#endif

namespace MueLu {
namespace VizHelpers {

  /* Non-member utility functions */

  inline std::string replaceAll(std::string original, std::string replaceWhat, std::string replaceWithWhat)
  {
    while(1) {
      const int pos = result.find(replaceWhat);
      if (pos == -1)
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

  inline bool collinear(Vec2 v1, Vec2 v2, Vec3 v3)
  {
    Vec3 diff1(v2.x - v1.x, v2.y - v1.y, 0);
    Vec3 diff2(v3.x - v2.x, v3.y - v2.y, 0);
    //diff1 and diff2 share v2, so if they are parallel then all 3 points collinear
    //cross product of parallel vectors is 0
    return mag(cross(diff1, diff2)) <= 1e-8;
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
    return mag(vecSubtract(p1, p2));
  }

  inline Vec2 segmentNormal(Vec2 v) //"normal" to a 2D vector - just rotate 90 degrees to left
  {
    return Vec2(v.y, -v.x);
  }

  inline Vec3 triNormal(Vec3 v1, Vec3 v2, Vec3 v3) //normal to face of triangle (will be outward rel. to polyhedron) (v1, v2, v3 are in CCW order when normal is toward viewpoint)
  {
    return crossProduct(vecSubtract(v2, v1), vecSubtract(v3, v1));
  }

  //get minimum distance from 'point' to plane containing v1, v2, v3 (or the triangle with v1, v2, v3 as vertices)
  inline double pointDistFromTri(Vec3 point, Vec3 v1, Vec3 v2, Vec3 v3)
  {
    using namespace std;
    Vec3 norm = getNorm(v1, v2, v3);
    //must normalize the normal vector
    double normScl = mymagnitude(norm);
    double rv = 0.0;
    if (normScl > 1e-8) {
      norm.x /= normScl;
      norm.y /= normScl;
      norm.z /= normScl;
      rv = fabs(dotProduct(norm, vecSubtract(point, v1)));
    } else {
      // triangle is degenerated
      Vec3 test1 = vecSubtract(v3, v1);
      Vec3 test2 = vecSubtract(v2, v1);
      bool useTest1 = mymagnitude(test1) > 0.0 ? true : false;
      bool useTest2 = mymagnitude(test2) > 0.0 ? true : false;
      if(useTest1 == true) {
        double normScl1 = mymagnitude(test1);
        test1.x /= normScl1;
        test1.y /= normScl1;
        test1.z /= normScl1;
        rv = fabs(dotProduct(test1, vecSubtract(point, v1)));
      } else if (useTest2 == true) {
        double normScl2 = mymagnitude(test2);
        test2.x /= normScl2;
        test2.y /= normScl2;
        test2.z /= normScl2;
        rv = fabs(dotProduct(test2, vecSubtract(point, v1)));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, Exceptions::RuntimeError,
                 "VisualizationHelpers::pointDistFromTri: Could not determine the distance of a point to a triangle.");
      }
    }
    return rv;
  }

  std::vector<int> makeUnique(std::vector<int>& verts)
  {
    using std::vector;
    using std::sort;
    using std::unique;
    vector<int> uniqueVerts = verts;
    sort(uniqueVerts.begin(), uniqueVerts.end());
    vector<int>::iterator newUniqueEnd = unique(uniqueVerts.begin(), uniqueVerts.end());
    uniqueVerts.erase(newUniqueEnd, uniqueVerts.end());
    //uniqueNodes is now a sorted list of the nodes whose info actually goes in file
    //Now replace values in vertices with locations of the old values in uniqueFine
    for(size_t i = 0; i < verts.size(); i++)
    {
      int lo = 0;
      int hi = uniqueVerts.size() - 1;
      int mid = 0;
      int search = verts[i];
      while(lo <= hi)
      {
        mid = lo + (hi - lo) / 2;
        if(uniqueVerts[mid] == search)
          break;
        else if(uniqueVerts[mid] > search)
          hi = mid - 1;
        else
          lo = mid + 1;
      }
      if(uniqueVerts[mid] != search)
        throw runtime_error("Issue in makeUnique_() - a point wasn't found in list.");
      verts[i] = mid;
    }
    return uniqueVerts;
  }

  std::vector<int> mergeAndMakeUnique(std::vector<int>& verts1, std::vector<int>& verts2)
  {
    using namespace std;
    vector<int> uniqueNodes;
    uniqueNodes.reserve(verts1.size() + verts2.size());
    uniqueNodes.insert(uniqueNodes.end(), verts1.begin(), verts1.end());
    uniqueNodes.insert(uniqueNodes.end(), verts2.begin(), verts2.end());
    sort(uniqueNodes.begin(), uniqueNodes.end());
    vector<int>::iterator newUniqueFineEnd = unique(uniqueNodes.begin(), uniqueNodes.end());
    uniqueNodes.erase(newUniqueFineEnd, uniqueNodes.end());
    //uniqueNodes is now a sorted list of the nodes whose info actually goes in file
    //Now replace values in vertices with locations of the old values in uniqueFine
    for(int i = 0; i < int(geomVerts_.size()); i++)
    {
      int lo = 0;
      int hi = uniqueNodes.size() - 1;
      int mid = 0;
      int search = geomVerts_[i];
      while(lo <= hi)
      {
        mid = lo + (hi - lo) / 2;
        if(uniqueNodes[mid] == search)
          break;
        else if(uniqueNodes[mid] > search)
          hi = mid - 1;
        else
          lo = mid + 1;
      }
      if(uniqueNodes[mid] != search)
        throw runtime_error("Issue in makeUnique_() - a point wasn't found in list.");
      geomVerts_[i] = mid;
    }
    return uniqueNodes;
  }

  /*----------------------------*/
  /* AggGeometry implementation */
  /*----------------------------*/

  //Constructor for AggregationExportFactory
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  AggGeometry(const Teuchos::RCP<Aggregates>& aggs, const std::vector<bool>& isRoot, const Teuchos::RCP<const Teuchos::Comm>& comm,
      const Teuchos::RCP<MultiVector>& coords)
  {
    bubbles_ = false;
    numNodes_ = coords->getLocalLength();
    numLocalAggs_ = aggs->GetNumAggregates();
    dims = coords->getNumVectors();
    this->x_ = coords->getData(0);
    this->y_ = coords->getData(1);
    if(dims == 3)
    {
      this->z_ = coords->getData(2);
    }
    this->rank_ = comm->getRank();
    this->nprocs_ = comm->getSize();
    this->vertex2Agg_ = aggs->GetVertex2AggId()->getData(0);
    if(nprocs_ != 1)
    {
      //prepare for calculating global aggregate ids
      std::vector<GlobalOrdinal> numAggsGlobal (numProcs, 0);
      std::vector<GlobalOrdinal> numAggsLocal  (numProcs, 0);
      std::vector<GlobalOrdinal> minGlobalAggId(numProcs, 0);
      numAggsLocal[rank_] = aggregates->GetNumAggregates();
      Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, nprocs_, &numAggsLocal[0], &numAggsGlobal[0]);
      for(size_t i = 1; i < numAggsGlobal.size(); ++i)
      {
        numAggsGlobal [i] += numAggsGlobal[i-1];
        minGlobalAggId[i]  = numAggsGlobal[i-1];
      }
      firstAgg_ = minGlobalAggId[rank_];
    }
    else
    {
      firstAgg_ = 0;
    }
    //can already compute aggOffsets_, since sizes of all aggregates known
    aggOffsets_.reserve(aggSizes.size() + 1);
    LocalOrdinal accum = 0;
    for(size_t i = 0; i < aggSizes.size(); i++)
    {
      aggOffsets_[i] = accum;
      accum += aggSizes[i];
    }
    aggOffsets_[numLocalAggs] = accum;
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
  AggGeometry(const Teuchos::RCP<Matrix>& P, const Teuchos::RCP<Map>& map, const Teuchos::RCP<Teuchos::Comm>& comm,
      const Teuchos::RCP<MultiVector>& coords, LocalOrdinal dofsPerNode, LocalOrdinal colsPerNode, bool ptent)
  {
    bubbles_ = !ptent;
    //use P (possibly including non-local entries)
    //to populate aggVerts_, aggOffsets_, vertex2Agg_, firstAgg_
    dims = coords->getNumVectors();
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
    TEUCHOS_TEST_FOR_EXCEPTION(strDomainMap.is_null(), Exceptions::RuntimeException, "Aggregate geometry requires the strided domain map from P/Ptent, but it was null.");
    numLocalAggs_ = strDomainMap->getNodeNumElements() / columnsPerNode;
    numNodes_ = coords->getLocalLength();
    auto rowMap = P->getRowMap();
    auto colMap = P->getColMap();
    if(ptent)
    {
      //know that aggs are not overlapping, so save time by using only local row views
      std::vector< std::set<LocalOrdinal> > localAggs(numLocalAggs_);
      // do loop over all local rows of prolongator and extract column information
      for (LocalOrdinal row = 0; row < Teuchos::as<LocalOrdinal>(P->getRowMap()->getNodeNumElements()); ++row)
      {
        ArrayView<const LocalOrdinal> indices;
        ArrayView<const Scalar> vals;
        P->getLocalRowView(row, indices, vals);
        for(auto c = indices.begin(); c != indices.end(); ++c)
        {
          localAggs[(*c)/columnsPerNode].insert(row/dofsPerNode);
        }
      }
      //fill aggOffsets
      aggOffsets_.resize(numLocalAggs + 1);
      LocalOrdinal accum = 0;
      for(LocalOrdinal i = 0; i < numLocalAggs; i++)
      {
        aggOffsets_[i] = accum;
        accum += localAggs[i].size();
      }
      aggOffsets_[numLocalAggs] = accum;
      //fill aggVerts
      aggVerts_.resize(accum);
      //number of vertices inserted into each aggregate so far
      std::vector<int> numInserted(accum, 0);
      for(auto& agg : localAggs)
      {
        for(auto v : agg)
        {
          aggVerts_[aggOffsets_[i] + numInserted[i]++] = rowMap->getGlobalElement(v);
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
        VertexData(GlobalOrdinal agg, GlobalOrdinal vertex, double x, double y, double z = 0)
        {
          this->agg = agg;
          this->vertex = vertex;
          this->x = x;
          this->y = y;
          this->z = z;
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
      std::vector<std::set<GlobalOrdinal>> aggMembers;
      //First, get all local vert-agg pairs as VertexData
      vector<VertexData> localVerts;
      {
        //Do this by getting local row views of every local row and adding LocalOrdinals to a std::set representing agg
        std::vector< std::set<LocalOrdinal> > localAggs(numLocalAggs_);
        // do loop over all local rows of prolongator and extract column information
        for (LocalOrdinal row = 0; row < Teuchos::as<LocalOrdinal>(P->getRowMap()->getNodeNumElements()); ++row)
        {
          ArrayView<const LocalOrdinal> indices;
          ArrayView<const Scalar> valsUnused;
          P->getLocalRowView(row, indices, valsUnused);
          for(auto c = indices.begin(); c != indices.end(); ++c)
          {
            aggMembers[(*c) / columnsPerNode].insert(row / dofsPerNode);
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
                GlobalOrdinal thisAgg = entry / columnsPerNode;
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
        std::vector<GlobalOrdinal> thisAggVerts;
        thisAggVerts.reserve(aggMemebers[i].size());
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
      std::vector<GlobalOrdinal> myLocalAggsPerProc(comm->getSize(), 0);
      std::vector<GlobalOrdinal> numLocalAggsPerProc(comm->getSize(), 0);
      myLocalAggsPerProc[comm->getRank()] = numLocalAggs;
      Teuchos::reduceAll<int, GlobalOrdinal>(*comm, Teuchos::REDUCE_MAX, comm->getSize(), &myLocalAggsPerProc[0], &numLocalAggsPerProc[0]);
      firstAgg_ = 0;
      for(int i = 0; i < comm->getRank(); ++i) {
        firstAgg_ += numLocalAggsPerProc[i];
      }
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
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
      if(dims == 2)
        cgalConvexHulls2D();
      else
        cgalConvexHulls3D();
#else
      if(dims == 2)
        convexHulls2D();
      else
        convexHulls3D();
#endif
    }
#ifdef HAVE_MUELU_CGAL
    else if(style == "Alpha Hulls")
    {
      if(dims == 2)
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
    geomVerts_.reserve(numFineNodes);
    geomSizes_.reserve(numFineNodes);
    for(LocalOrdinal i = 0; i < numLocalAggs_; i++)
    {
      GlobalOrdinal numVerts = aggOffsets_[i + 1] - aggOffsets_[i];
      for(GlobalOrdinal j = 0; j < numVerts; j++)
      {
        GlobalOrdinal go = aggVerts_[aggOffsets_[i] + j];
        geomVerts_.push_back(go);
        geomAggs_.push_back(i);
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
        isRoot_(map->getGlobalElement(i)) = aggregates->IsRoot(i);
      }
    }
    else
    {
      //must fake the roots, assume root of each agg is probably
      //the vertex closest to average position of vertices in agg
      for(LocalOrdinal agg = 0; agg < numLocalAggs_; agg++)
      {
        Vec3 avgPos(0, 0, 0);
        int aggSize = aggOffsets_[agg + 1] - aggOffsets_[agg];
        for(LocalOrdinal i = aggOffsets_[agg]; i < aggOffsets_[agg + 1]; i++)
        {
          LocalOrdinal vert = verts_[aggVerts_[i]];
        }
        avgPos *= (1.0 / aggSize);
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
            "per aggregate, but aggregate ") + to_string(agg) + " on proc " + to_string(rank) + " does not have a root.");
      //for each vert in agg (except root), make a line segment between the vert and root
      for(GlobalOrdinal i = aggStart; i < aggEnd; i++)
      {
        int vert = aggVerts_[i];
        if(vert == root)
          continue;
        geomVerts_.push_back(vert);
        geomAggs_.push(back(agg));
        geomVerts_.push_back(root);
        geomAggs_.push(back(agg));
        geomSizes_.push_back(2);
      }
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  convexHulls2D() {
    for(LocalOrdinal agg = 0; agg < numLocalAggs_; agg++) {
      GlobalOrdinal aggStart = aggOffsets_[agg];
      GlobalOrdial aggEnd = aggOffsets_[agg + 1];
      GlobalOrdinal aggSize = aggEnd - aggStart;
      if(aggSize == 1) {
        //aggregate is a point
        geomVerts_.push_back(aggVerts_[aggStart]);
        geomAggs_.push_back(agg);
        geomSizes_.push_back(1);
        continue;
      }
      if(aggNodes.size() == 2) {
        //aggregate is a line
        geomVerts_.push_back(aggVerts_[aggStart]);
        geomAggs_.push_back(agg);
        geomVerts_.push_back(aggVerts_[aggStart + 1]);
        geomAggs_.push_back(agg);
        geomSizes_.push_back(2);
        continue;
      }
      //check if all points are collinear, need to explicitly draw a line in that case.
      {
        //find first pair (v1, v2) of vertices that are not the same, in case multiple nodes share 
        GlobalOrdinal iter = 2;
        GlobalOrdinal v1 = aggVerts_[aggStart];
        GlobalOrdinal v2 = aggVerts_[aggStart + 1];
        while(x_[v1] == x_[v2] && y_[v1] == y_[v2] && (iter < aggSize))
        {
          v2 = aggVerts_[aggStart + iter++];
        }
        if(x_[v1] == x_[v2] && y_[v1] == y_[v2])
        {
          //all vertices in agg have same position
          geomVerts_.push_back(aggVerts_[aggStart]);
          geomAggs_.push_back(agg);
          geomSizes_.push_back(1);
          continue;
        }
        //check if the rest of the vertices are collinear with v1, v2
        Vec2 vec1(x_[v1], y_[v1]);
        Vec2 vec2(x_[v2], y_[v2]);
        for(int i = iter; i < aggSize; i++)
        {
          Vec2 vec3(x_[aggVerts_[aggStart + iter]], y_[aggVerts_[aggStart + iter]]);
          if(!collinear(vec1, vec2, vec3))
          {
            break;
          }
        }
        if(iter == aggSize)
        {
          //all vertices collinear
          //find min and max coordinates in agg and make line segment between them
          GlobalOrdinal minVert = aggVerts_[aggStart];
          GlobalOrdinal maxVert = minVert;
          for(GlobalOrdinal i = 1; i < aggSize; i++)
          {
            GlobalOrdinal v = aggVerts_[aggStart + i];
            //compare X first and then use Y as a tiebreak (will always find 2 most distant points on the line)
            if(x_[v] < x_[minVert] || (x_[v] == x_[minVert] && y_[v] < y_[minVert]))
            {
              minVert = v;
            }
            if(x_[v] > x_[maxVert] || (x_[v] == x_[maxVert] && y_[v] > y_[maxVert]))
            {
              maxVert = v;
            }
          }
          geomVerts_.push_back(minVert);
          geomAggs_.push_back(agg);
          geomVerts_.push_back(maxVert);
          geomAggs_.push_back(agg);
          geomSizes_.push_back(2);
          continue;
        }
      }
      {
        //use "gift wrap" algorithm to find the convex hull
        //first, find vert with minimum x (and use min y as a tiebreak)
        GlobalOrdinal minVert = aggVerts_[aggStart];
        Vec3 minPos = verts_[minVert];
        for(GlobalOrdinal i = 1; i < aggSize; i++)
        {
          GlobalOrdinal test = aggVerts_[aggStart + i];
          Vec3 testPos = verts_[test];
          if(testPos.x < minPos.x || (testPos.x == minPos.x && testPos.y < minPos.y))
          {
            minVert = test;
            minPos = testPos;
          }
        }
        std::vector<GlobalOrdinal> loop;
        GlobalOrdinal loopIter = minVert;
        //loop until a closed loop (with > 1 vertices) has been found
        while(true)
        {
          //find another vertex "ray" in the aggregate 
          GlobalOrdinal ray = aggVerts_[aggStart];
          {
            GlobalOrdinal i = 0;
            while(ray == loopIter)
            {
              i++;
              ray = aggVerts_[aggStart + i];
            }
          }
          //sweep through all vertices, and if one is on the left side of loopIter->ray, change ray to it
          //"is point left of ray?" is answered by segmentNormal and dot prod
          Vec2 norm = segmentNormal(verts_[ray] - verts_[loopIter]);
          bool foundNext = false;
          while(!foundNext)
          {
            for(GlobalOrdinal i = aggStart; i < aggEnd; i++)
            {
              if(aggVerts_[i] == loopIter || aggVerts_[i] == ray)
                continue;
              if(dotProduct(norm, verts_[aggVerts_[i]] - verts_[loopIter]) > 0)
              {
                foundNext = true;
                loop.push_back(loopIter);
                loopIter = aggVerts_[i];
                break;
              }
            }
            foundNext = true;
          }
          if(loop.back() == loop.front() && loop.size() > 1)
          {
            //have a closed loop
            loop.pop_back();
            break;
          }
        }
        //loop now contains the vertex loop representing convex hull (with none repeated)
        for(auto vert : loop)
        {
          geomVerts_.push_back(vert);
          geomAggs_.push_back(agg);
        }
        geomSizes_.push_back(loop.size());
      }
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
      //have a list of nodes in the aggregate
      TEUCHOS_TEST_FOR_EXCEPTION(aggNodes.size() == 0, Exceptions::RuntimeError,
               "CoarseningVisualization::doCGALConvexHulls2D: aggregate contains zero nodes!");
      if(aggNodes.size() == 1) {
        geomVerts_.push_back(aggNodes.front());
        geomAggs_.push_back(agg);
        geomSizes_.push_back(1);
        continue;
      }
      if(aggNodes.size() == 2) {
        geomVerts_.push_back(aggNodes.front());
        geomAggs_.push_back(agg);
        geomVerts_.push_back(aggNodes.back());
        geomAggs_.push_back(agg);
        geomSizes_.push_back(2);
        continue;
      }
      //check if all points are collinear, need to explicitly draw a line in that case.
      bool collinear = true; //assume true at first, if a segment not parallel to others then clear
      {
        auto it = aggNodes.begin();
        Vec2 firstPoint = verts_[*it].toVec2();
        it++;
        Vec2 secondPoint = verts_[*it].toVec2();
        it++;  //it now points to third node in the aggregate
        Vec2 norm1 = segmentNormal(secondPoint - firstPoint);
        do {
          Vec2 thisNorm = segmentNormal(verts_[*it].toVec2() - firstPoint);
          //rotate one of the vectors by 90 degrees so that dot product is 0 if the two are parallel
          double temp = thisNorm.x;
          thisNorm.x = thisNorm.y;
          thisNorm.y = temp;
          double comp = dotProduct(norm1, thisNorm);
          if(-1e-8 > comp || comp > 1e-8) {
            collinear = false;
            break;
          }
          it++;
        }
        while(it != aggNodes.end());
      }
      if(collinear)
      {
        //find the most distant two points in the plane and use as endpoints of line representing agg
        GlobalOrdinal minNode = aggNodes[0];    //min X then min Y where x is equal
        GlobalOrdinal maxNode = aggNodes[0];    //max X then max Y where x is equal
        for(auto& it : aggNodes) {
          auto itPoint = verts_[it].toVec2();
          auto minPoint = verts_[minNode].toVec2();
          auto maxPoint = verts_[maxNode].toVec2();
          if(it.x < minPoint.x || (it.x == minPoint.x && it.y < minPoint.y))
          {
            minPoint = it;
          }
          if(it.x > maxPoint.x || (it.x == maxPoint.x && it.y > maxPoint.y))
          {
            maxPoint = it;
          }
        }
        //Just set up a line between nodes *min and *max
        geomVerts_.push_back(minNode);
        geomAggs_.push_back(agg);
        geomVerts_.push_back(maxNode);
        geomAggs_.push_back(agg);
        geomSizes_.push_back(2);
        continue; //jump to next aggregate (in outermost loop)
      }
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
            if(fabs(result[r].x() - aggPoints[l].x) < eps &&
               fabs(result[r].y() - aggPoints[l].y) < eps)
            {
              geomVerts_.push_back(aggNodes[l]);
              geomAggs_.push_back(agg);
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
    //Use 3D quickhull algo.
    //Vector of node indices representing triangle vertices
    //Note: Calculate the hulls first since will only include point data for points in the hulls
    //Effectively the size() of vertIndices after each hull is added to it
    typedef std::list<int>::iterator Iter;
    for(int agg = 0; agg < numLocalAggs; agg++) {
      std::list<int> aggNodes; //At first, list of all nodes in the aggregate. As nodes are enclosed or included by/in hull, remove them
      for(int i = 0; i < numFineNodes; i++) {
        if(vertex2AggId[i] == agg)
          aggNodes.push_back(i);
      }
      //First, check anomalous cases
      TEUCHOS_TEST_FOR_EXCEPTION(aggNodes.size() == 0, Exceptions::RuntimeError,
               "CoarseningVisualization::doConvexHulls3D: aggregate contains zero nodes!");
      if(aggNodes.size() == 1) {
        geomVerts_.push_back(aggNodes.front());
        geomSizes_.push_back(1);
        continue;
      } else if(aggNodes.size() == 2) {
        geomVerts_.push_back(aggNodes.front());
        geomVerts_.push_back(aggNodes.back());
        geomSizes_.push_back(2);
        continue;
      }
      //check for collinearity
      bool areCollinear = true;
      {
        Iter it = aggNodes.begin();
        Vec3 firstVec(xCoords[*it], yCoords[*it], zCoords[*it]);
        Vec3 comp;
        {
          it++;
          Vec3 secondVec(xCoords[*it], yCoords[*it], zCoords[*it]); //cross this with other vectors to compare
          comp = vecSubtract(secondVec, firstVec);
          it++;
        }
        for(; it != aggNodes.end(); it++) {
          Vec3 thisVec(xCoords[*it], yCoords[*it], zCoords[*it]);
          Vec3 cross = crossProduct(vecSubtract(thisVec, firstVec), comp);
          if(mymagnitude(cross) > 1e-10) {
            areCollinear = false;
            break;
          }
        }
      }
      if(areCollinear)
      {
        //find the endpoints of segment describing all the points
        //compare x, if tie compare y, if tie compare z
        Iter min = aggNodes.begin();
        Iter max = aggNodes.begin();
        Iter it = ++aggNodes.begin();
        for(; it != aggNodes.end(); it++) {
          if(xCoords[*it] < xCoords[*min]) min = it;
          else if(xCoords[*it] == xCoords[*min]) {
            if(yCoords[*it] < yCoords[*min]) min = it;
            else if(yCoords[*it] == yCoords[*min]) {
              if(zCoords[*it] < zCoords[*min]) min = it;
            }
          }
          if(xCoords[*it] > xCoords[*max]) max = it;
          else if(xCoords[*it] == xCoords[*max]) {
            if(yCoords[*it] > yCoords[*max]) max = it;
            else if(yCoords[*it] == yCoords[*max]) {
              if(zCoords[*it] > zCoords[*max])
                max = it;
            }
          }
        }
        geomVerts_.push_back(*min);
        geomVerts_.push_back(*max);
        geomSizes_.push_back(2);
        continue;
      }
      bool areCoplanar = true;
      {
        //number of points is known to be >= 3
        Iter vert = aggNodes.begin();
        Vec3 v1(xCoords[*vert], yCoords[*vert], zCoords[*vert]);
        vert++;
        Vec3 v2(xCoords[*vert], yCoords[*vert], zCoords[*vert]);
        vert++;
        Vec3 v3(xCoords[*vert], yCoords[*vert], zCoords[*vert]);
        vert++;
        //Make sure the first three points aren't also collinear (need a non-degenerate triangle to get a normal)
        while(mymagnitude(crossProduct(vecSubtract(v1, v2), vecSubtract(v1, v3))) < 1e-10) {
          //Replace the third point with the next point
          v3 = Vec3(xCoords[*vert], yCoords[*vert], zCoords[*vert]);
          vert++;
        }
        for(; vert != aggNodes.end(); vert++) {
          Vec3 pt(xCoords[*vert], yCoords[*vert], zCoords[*vert]);
          if(fabs(pointDistFromTri(pt, v1, v2, v3)) > 1e-12) {
            areCoplanar = false;
            break;
          }
        }
        if(areCoplanar) {
          //do 2D convex hull
          Vec3 planeNorm = getNorm(v1, v2, v3);
          planeNorm.x = fabs(planeNorm.x);
          planeNorm.y = fabs(planeNorm.y);
          planeNorm.z = fabs(planeNorm.z);
          std::vector<Vec2> points;
          std::vector<int> nodes;
          if(planeNorm.x >= planeNorm.y && planeNorm.x >= planeNorm.z) {
            //project points to yz plane to make hull
            for(Iter it = aggNodes.begin(); it != aggNodes.end(); it++) {
              nodes.push_back(*it);
              points.push_back(Vec2(yCoords[*it], zCoords[*it]));
            }
          }
          if(planeNorm.y >= planeNorm.x && planeNorm.y >= planeNorm.z) {
            //use xz
            for(Iter it = aggNodes.begin(); it != aggNodes.end(); it++) {
              nodes.push_back(*it);
              points.push_back(Vec2(xCoords[*it], zCoords[*it]));
            }
          }
          if(planeNorm.z >= planeNorm.x && planeNorm.z >= planeNorm.y) {
            for(Iter it = aggNodes.begin(); it != aggNodes.end(); it++) {
              nodes.push_back(*it);
              points.push_back(Vec2(xCoords[*it], yCoords[*it]));
            }
          }
          std::vector<int> convhull2d = giftWrap(points, nodes, xCoords, yCoords);
          geomSizes_.push_back(convhull2d.size());
          geomVerts_.reserve(geomVerts_.size() + convhull2d.size());
          for(size_t i = 0; i < convhull2d.size(); i++)
            geomVerts_.push_back(convhull2d[i]);
          continue;
        }
      }
      Iter exIt = aggNodes.begin(); //iterator to be used for searching for min/max x/y/z
      int extremeSix[] = {*exIt, *exIt, *exIt, *exIt, *exIt, *exIt}; //nodes with minimumX, maxX, minY ...
      exIt++;
      for(; exIt != aggNodes.end(); exIt++) {
        Iter it = exIt;
        if(xCoords[*it] < xCoords[extremeSix[0]] ||
          (xCoords[*it] == xCoords[extremeSix[0]] && yCoords[*it] < yCoords[extremeSix[0]]) ||
          (xCoords[*it] == xCoords[extremeSix[0]] && yCoords[*it] == yCoords[extremeSix[0]] && zCoords[*it] < zCoords[extremeSix[0]]))
            extremeSix[0] = *it;
        if(xCoords[*it] > xCoords[extremeSix[1]] ||
          (xCoords[*it] == xCoords[extremeSix[1]] && yCoords[*it] > yCoords[extremeSix[1]]) ||
          (xCoords[*it] == xCoords[extremeSix[1]] && yCoords[*it] == yCoords[extremeSix[1]] && zCoords[*it] > zCoords[extremeSix[1]]))
            extremeSix[1] = *it;
        if(yCoords[*it] < yCoords[extremeSix[2]] ||
          (yCoords[*it] == yCoords[extremeSix[2]] && zCoords[*it] < zCoords[extremeSix[2]]) ||
          (yCoords[*it] == yCoords[extremeSix[2]] && zCoords[*it] == zCoords[extremeSix[2]] && xCoords[*it] < xCoords[extremeSix[2]]))
            extremeSix[2] = *it;
        if(yCoords[*it] > yCoords[extremeSix[3]] ||
          (yCoords[*it] == yCoords[extremeSix[3]] && zCoords[*it] > zCoords[extremeSix[3]]) ||
          (yCoords[*it] == yCoords[extremeSix[3]] && zCoords[*it] == zCoords[extremeSix[3]] && xCoords[*it] > xCoords[extremeSix[3]]))
            extremeSix[3] = *it;
        if(zCoords[*it] < zCoords[extremeSix[4]] ||
          (zCoords[*it] == zCoords[extremeSix[4]] && xCoords[*it] < xCoords[extremeSix[4]]) ||
          (zCoords[*it] == zCoords[extremeSix[4]] && xCoords[*it] == xCoords[extremeSix[4]] && yCoords[*it] < yCoords[extremeSix[4]]))
            extremeSix[4] = *it;
        if(zCoords[*it] > zCoords[extremeSix[5]] ||
          (zCoords[*it] == zCoords[extremeSix[5]] && xCoords[*it] > xCoords[extremeSix[5]]) ||
          (zCoords[*it] == zCoords[extremeSix[5]] && xCoords[*it] == xCoords[extremeSix[5]] && yCoords[*it] > zCoords[extremeSix[5]]))
            extremeSix[5] = *it;
      }
      Vec3 extremeVectors[6];
      for(int i = 0; i < 6; i++) {
        Vec3 thisExtremeVec(xCoords[extremeSix[i]], yCoords[extremeSix[i]], zCoords[extremeSix[i]]);
        extremeVectors[i] = thisExtremeVec;
      }
      double maxDist = 0;
      int base1 = 0; //ints from 0-5: which pair out of the 6 extreme points are the most distant? (indices in extremeSix and extremeVectors)
      int base2 = 0;
      for(int i = 0; i < 5; i++) {
        for(int j = i + 1; j < 6; j++) {
          double thisDist = distance(extremeVectors[i], extremeVectors[j]);
          if(thisDist > maxDist) {
            maxDist = thisDist;
            base1 = i;
            base2 = j;
          }
        }
      }
      std::list<Triangle> hullBuilding;    //each Triangle is a triplet of nodes (int IDs) that form a triangle
      //remove base1 and base2 iters from aggNodes, they are known to be in the hull
      aggNodes.remove(extremeSix[base1]);
      aggNodes.remove(extremeSix[base2]);
      //extremeSix[base1] and [base2] still have the Vec3 representation
      Triangle tri;
      tri.v1 = extremeSix[base1];
      tri.v2 = extremeSix[base2];
      //Now find the point that is furthest away from the line between base1 and base2
      maxDist = 0;
      //need the vectors to do "quadruple product" formula
      Vec3 b1 = extremeVectors[base1];
      Vec3 b2 = extremeVectors[base2];
      Iter thirdNode;
      for(Iter node = aggNodes.begin(); node != aggNodes.end(); node++) {
        Vec3 nodePos(xCoords[*node], yCoords[*node], zCoords[*node]);
        double dist = mymagnitude(crossProduct(vecSubtract(nodePos, b1), vecSubtract(nodePos, b2))) / mymagnitude(vecSubtract(b2, b1));
        if(dist > maxDist) {
          maxDist = dist;
          thirdNode = node;
        }
      }
      //Now know the last node in the first triangle
      tri.v3 = *thirdNode;
      hullBuilding.push_back(tri);
      Vec3 b3(xCoords[*thirdNode], yCoords[*thirdNode], zCoords[*thirdNode]);
      aggNodes.erase(thirdNode);
      //Find the fourth node (most distant from triangle) to form tetrahedron
      maxDist = 0;
      int fourthVertex = -1;
      for(Iter node = aggNodes.begin(); node != aggNodes.end(); node++) {
        Vec3 thisNode(xCoords[*node], yCoords[*node], zCoords[*node]);
        double nodeDist = pointDistFromTri(thisNode, b1, b2, b3);
        if(nodeDist > maxDist) {
          maxDist = nodeDist;
          fourthVertex = *node;
        }
      }
      aggNodes.remove(fourthVertex);
      Vec3 b4(xCoords[fourthVertex], yCoords[fourthVertex], zCoords[fourthVertex]);
      //Add three new triangles to hullBuilding to form the first tetrahedron
      //use tri to hold the triangle info temporarily before being added to list
      tri = hullBuilding.front();
      tri.v1 = fourthVertex;
      hullBuilding.push_back(tri);
      tri = hullBuilding.front();
      tri.v2 = fourthVertex;
      hullBuilding.push_back(tri);
      tri = hullBuilding.front();
      tri.v3 = fourthVertex;
      hullBuilding.push_back(tri);
      //now orient all four triangles so that the vertices are oriented clockwise (so getNorm_ points outward for each)
      Vec3 barycenter((b1.x + b2.x + b3.x + b4.x) / 4.0, (b1.y + b2.y + b3.y + b4.y) / 4.0, (b1.z + b2.z + b3.z + b4.z) / 4.0);
      for(std::list<Triangle>::iterator tetTri = hullBuilding.begin(); tetTri != hullBuilding.end(); tetTri++) {
        Vec3 triVert1(xCoords[tetTri->v1], yCoords[tetTri->v1], zCoords[tetTri->v1]);
        Vec3 triVert2(xCoords[tetTri->v2], yCoords[tetTri->v2], zCoords[tetTri->v2]);
        Vec3 triVert3(xCoords[tetTri->v3], yCoords[tetTri->v3], zCoords[tetTri->v3]);
        Vec3 trinorm = getNorm(triVert1, triVert2, triVert3);
        Vec3 ptInPlane = tetTri == hullBuilding.begin() ? b1 : b4; //first triangle definitely has b1 in it, other three definitely have b4
        if(isInFront(barycenter, ptInPlane, trinorm)) {
          //don't want the faces of the tetrahedron to face inwards (towards barycenter) so reverse orientation
          //by swapping two vertices
          int temp = tetTri->v1;
          tetTri->v1 = tetTri->v2;
          tetTri->v2 = temp;
        }
      }
      //now, have starting polyhedron in hullBuilding (all faces are facing outwards according to getNorm_) and remaining nodes to process are in aggNodes
      //recursively, for each triangle, make a list of the points that are 'in front' of the triangle. Find the point with the maximum distance from the triangle.
      //Add three new triangles, each sharing one edge with the original triangle but now with the most distant point as a vertex. Remove the most distant point from
      //the list of all points that need to be processed. Also from that list remove all points that are in front of the original triangle but not in front of all three
      //new triangles, since they are now enclosed in the hull.
      //Construct point lists for each face of the tetrahedron individually.
      Vec3 trinorms[4]; //normals to the four tetrahedron faces, now oriented outwards
      int index = 0;
      for(std::list<Triangle>::iterator tetTri = hullBuilding.begin(); tetTri != hullBuilding.end(); tetTri++) {
        Vec3 triVert1(xCoords[tetTri->v1], yCoords[tetTri->v1], zCoords[tetTri->v1]);
        Vec3 triVert2(xCoords[tetTri->v2], yCoords[tetTri->v2], zCoords[tetTri->v2]);
        Vec3 triVert3(xCoords[tetTri->v3], yCoords[tetTri->v3], zCoords[tetTri->v3]);
        trinorms[index] = getNorm(triVert1, triVert2, triVert3);
        index++;
      }
      std::list<int> startPoints1;
      std::list<int> startPoints2;
      std::list<int> startPoints3;
      std::list<int> startPoints4;
      //scope this so that 'point' is not in function scope
      {
        Iter point = aggNodes.begin();
        while(!aggNodes.empty())  //this removes points one at a time as they are put in startPointsN or are already done
        {
          Vec3 pointVec(xCoords[*point], yCoords[*point], zCoords[*point]);
          //Note: Because of the way the tetrahedron faces are constructed above,
          //face 1 definitely contains b1 and faces 2-4 definitely contain b4.
          if(isInFront(pointVec, b1, trinorms[0])) {
            startPoints1.push_back(*point);
            point = aggNodes.erase(point);
          } else if(isInFront(pointVec, b4, trinorms[1])) {
            startPoints2.push_back(*point);
            point = aggNodes.erase(point);
          } else if(isInFront(pointVec, b4, trinorms[2])) {
            startPoints3.push_back(*point);
            point = aggNodes.erase(point);
          } else if(isInFront(pointVec, b4, trinorms[3])) {
            startPoints4.push_back(*point);
            point = aggNodes.erase(point);
          } else {
            point = aggNodes.erase(point); //points here are already inside tetrahedron.
          }
        }
        //Call processTriangle for each triangle in the initial tetrahedron, one at a time.
      }
      typedef std::list<Triangle>::iterator TriIter;
      TriIter firstTri = hullBuilding.begin();
      Triangle start1 = *firstTri;
      firstTri++;
      Triangle start2 = *firstTri;
      firstTri++;
      Triangle start3 = *firstTri;
      firstTri++;
      Triangle start4 = *firstTri;
      //kick off depth-first recursive filling of hullBuilding list with all triangles in the convex hull
      if(!startPoints1.empty())
        processTriangle(hullBuilding, start1, startPoints1, barycenter, xCoords, yCoords, zCoords);
      if(!startPoints2.empty())
        processTriangle(hullBuilding, start2, startPoints2, barycenter, xCoords, yCoords, zCoords);
      if(!startPoints3.empty())
        processTriangle(hullBuilding, start3, startPoints3, barycenter, xCoords, yCoords, zCoords);
      if(!startPoints4.empty())
        processTriangle(hullBuilding, start4, startPoints4, barycenter, xCoords, yCoords, zCoords);
      //hullBuilding now has all triangles that make up this hull.
      //Dump hullBuilding info into the list of all triangles for the scene.
      geomVerts_.reserve(geomVerts_.size() + 3 * hullBuilding.size());
      for(TriIter hullTri = hullBuilding.begin(); hullTri != hullBuilding.end(); hullTri++) {
        geomVerts_.push_back(hullTri->v1);
        geomVerts_.push_back(hullTri->v2);
        geomVerts_.push_back(hullTri->v3);
        geomSizes_.push_back(3);
      }
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  std::vector<Triangle> AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  processTriangle(std::list<Triangle>& tris, Triangle tri, std::list<int>& pointsInFront, Vec3& barycenter) {
    //*tri is in the tris list, and is the triangle to process here. tris is a complete list of all triangles in the hull so far. pointsInFront is only a list of the nodes in front of tri. Need coords also.
    //precondition: each triangle is already oriented so that getNorm_(v1, v2, v3) points outward (away from interior of hull)
    //First find the point furthest from triangle.
    using namespace std;
    typedef std::list<int>::iterator Iter;
    typedef std::list<Triangle>::iterator TriIter;
    typedef list<pair<int, int> >::iterator EdgeIter;
    double maxDist = 0;
    //Need vector representations of triangle's vertices
    Vec3 v1(xCoords[tri.v1], yCoords[tri.v1], zCoords[tri.v1]);
    Vec3 v2(xCoords[tri.v2], yCoords[tri.v2], zCoords[tri.v2]);
    Vec3 v3(xCoords[tri.v3], yCoords[tri.v3], zCoords[tri.v3]);
    Vec3 farPointVec; //useful to have both the point's coordinates and it's position in the list
    Iter farPoint = pointsInFront.begin();
    for(Iter point = pointsInFront.begin(); point != pointsInFront.end(); point++)
    {
      Vec3 pointVec(xCoords[*point], yCoords[*point], zCoords[*point]);
      double dist = pointDistFromTri(pointVec, v1, v2, v3);
      if(dist > maxDist)
      {
        dist = maxDist;
        farPointVec = pointVec;
        farPoint = point;
      }
    }
    //Find all the triangles that the point is in front of (can be more than 1)
    //At the same time, remove them from tris, as every one will be replaced later
    vector<Triangle> visible; //use a list of iterators so that the underlying object is still in tris
    for(TriIter it = tris.begin(); it != tris.end();)
    {
      Vec3 vec1(xCoords[it->v1], yCoords[it->v1], zCoords[it->v1]);
      Vec3 vec2(xCoords[it->v2], yCoords[it->v2], zCoords[it->v2]);
      Vec3 vec3(xCoords[it->v3], yCoords[it->v3], zCoords[it->v3]);
      Vec3 norm = getNorm(vec1, vec2, vec3);
      if(isInFront(farPointVec, vec1, norm))
      {
        visible.push_back(*it);
        it = tris.erase(it);
      }
      else
        it++;
    }
    //Figure out what triangles need to be destroyed/created
    //First create a list of edges (as std::pair<int, int>, where the two ints are the node endpoints)
    list<pair<int, int> > horizon;
    //For each triangle, add edges to the list iff the edge only appears once in the set of all
    //Have members of horizon have the lower node # first, and the higher one second
    for(vector<Triangle>::iterator it = visible.begin(); it != visible.end(); it++)
    {
      pair<int, int> e1(it->v1, it->v2);
      pair<int, int> e2(it->v2, it->v3);
      pair<int, int> e3(it->v1, it->v3);
      //"sort" the pair values
      if(e1.first > e1.second)
      {
        int temp = e1.first;
        e1.first = e1.second;
        e1.second = temp;
      }
      if(e2.first > e2.second)
      {
        int temp = e2.first;
        e2.first = e2.second;
        e2.second = temp;
      }
      if(e3.first > e3.second)
      {
        int temp = e3.first;
        e3.first = e3.second;
        e3.second = temp;
      }
      horizon.push_back(e1);
      horizon.push_back(e2);
      horizon.push_back(e3);
    }
    //sort based on lower node first, then higher node (lexicographical ordering built in to pair)
    horizon.sort();
    //Remove all edges from horizon, except those that appear exactly once
    {
      EdgeIter it = horizon.begin();
      while(it != horizon.end())
      {
        int occur = count(horizon.begin(), horizon.end(), *it);
        if(occur > 1)
        {
          pair<int, int> removeVal = *it;
          while(removeVal == *it && !(it == horizon.end()))
            it = horizon.erase(it);
        }
        else
          it++;
      }
    }
    //Now make a list of new triangles being created, each of which take 2 vertices from an edge and one from farPoint
    list<Triangle> newTris;
    for(EdgeIter it = horizon.begin(); it != horizon.end(); it++)
    {
      Triangle t(it->first, it->second, *farPoint);
      newTris.push_back(t);
    }
    //Ensure every new triangle is oriented outwards, using the barycenter of the initial tetrahedron
    vector<Triangle> trisToProcess;
    vector<list<int> > newFrontPoints;
    for(TriIter it = newTris.begin(); it != newTris.end(); it++)
    {
      Vec3 t1(xCoords[it->v1], yCoords[it->v1], zCoords[it->v1]);
      Vec3 t2(xCoords[it->v2], yCoords[it->v2], zCoords[it->v2]);
      Vec3 t3(xCoords[it->v3], yCoords[it->v3], zCoords[it->v3]);
      if(isInFront(barycenter, t1, getNorm(t1, t2, t3)))
      {
        //need to swap two vertices to flip orientation of triangle
        int temp = it->v1;
        Vec3 tempVec = t1;
        it->v1 = it->v2;
        t1 = t2;
        it->v2 = temp;
        t2 = tempVec;
      }
      Vec3 outwardNorm = getNorm(t1, t2, t3); //now definitely points outwards
      //Add the triangle to tris
      tris.push_back(*it);
      trisToProcess.push_back(tris.back());
      //Make a list of the points that are in front of nextToProcess, to be passed in for processing
      list<int> newInFront;
      for(Iter point = pointsInFront.begin(); point != pointsInFront.end();)
      {
        Vec3 pointVec(xCoords[*point], yCoords[*point], zCoords[*point]);
        if(isInFront(pointVec, t1, outwardNorm))
        {
          newInFront.push_back(*point);
          point = pointsInFront.erase(point);
        }
        else
          point++;
      }
      newFrontPoints.push_back(newInFront);
    }
    vector<Triangle> allRemoved; //list of all invalid iterators that were erased by calls to processTriangle below
    for(int i = 0; i < int(trisToProcess.size()); i++)
    {
      if(!newFrontPoints[i].empty())
      {
        //Comparing the 'triangle to process' to the one for this call prevents infinite recursion/stack overflow.
        //TODO: Why was it doing that? Rounding error? Make more robust fix. But this does work for the time being.
        if(find(allRemoved.begin(), allRemoved.end(), trisToProcess[i]) == allRemoved.end() && !(trisToProcess[i] == tri))
        {
          vector<Triangle> removedList = processTriangle(tris, trisToProcess[i], newFrontPoints[i], barycenter, xCoords, yCoords, zCoords);
          for(int j = 0; j < int(removedList.size()); j++)
            allRemoved.push_back(removedList[j]);
        }
      }
    }
    return visible;
  }

#ifdef HAVE_MUELU_CGAL
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  cgalConvexHulls3D() {

    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::Point_3 Point_3;
    typedef CGAL::Polyhedron_3<K> Polyhedron_3;
    typedef std::vector<int>::iterator Iter;
    for(int agg = 0; agg < numLocalAggs; agg++) {
      std::vector<int> aggNodes;
      std::vector<Point_3> aggPoints;
      for(int i = 0; i < numFineNodes; i++) {
        if(vertex2AggId[i] == agg) {
          Point_3 p(xCoords[i], yCoords[i], zCoords[i]);
          aggPoints.push_back(p);
          aggNodes.push_back(i);
        }
      }
      //First, check anomalous cases
      TEUCHOS_TEST_FOR_EXCEPTION(aggNodes.size() == 0, Exceptions::RuntimeError,
               "CoarseningVisualization::doCGALConvexHulls3D: aggregate contains zero nodes!");
      if(aggNodes.size() == 1) {
        geomVerts_.push_back(aggNodes.front());
        geomSizes_.push_back(1);
        continue;
      } else if(aggNodes.size() == 2) {
        geomVerts_.push_back(aggNodes.front());
        geomVerts_.push_back(aggNodes.back());
        geomSizes_.push_back(2);
        continue;
      }
      //check for collinearity
      bool areCollinear = true;
      {
        Iter it = aggNodes.begin();
        Vec3 firstVec(xCoords[*it], yCoords[*it], zCoords[*it]);
        Vec3 comp;
        {
          it++;
          Vec3 secondVec(xCoords[*it], yCoords[*it], zCoords[*it]); //cross this with other vectors to compare
          comp = vecSubtract(secondVec, firstVec);
          it++;
        }
        for(; it != aggNodes.end(); it++) {
          Vec3 thisVec(xCoords[*it], yCoords[*it], zCoords[*it]);
          Vec3 cross = crossProduct(vecSubtract(thisVec, firstVec), comp);
          if(mymagnitude(cross) > 1e-8) {
            areCollinear = false;
            break;
          }
        }
      }
      if(areCollinear)
      {
        //find the endpoints of segment describing all the points
        //compare x, if tie compare y, if tie compare z
        Iter min = aggNodes.begin();
        Iter max = aggNodes.begin();
        Iter it = ++aggNodes.begin();
        for(; it != aggNodes.end(); it++) {
          if(xCoords[*it] < xCoords[*min]) min = it;
          else if(xCoords[*it] == xCoords[*min]) {
            if(yCoords[*it] < yCoords[*min]) min = it;
            else if(yCoords[*it] == yCoords[*min]) {
              if(zCoords[*it] < zCoords[*min]) min = it;
            }
          }
          if(xCoords[*it] > xCoords[*max]) max = it;
          else if(xCoords[*it] == xCoords[*max]) {
            if(yCoords[*it] > yCoords[*max]) max = it;
            else if(yCoords[*it] == yCoords[*max]) {
              if(zCoords[*it] > zCoords[*max])
                max = it;
            }
          }
        }
        geomVerts_.push_back(*min);
        geomVerts_.push_back(*max);
        geomSizes_.push_back(2);
        continue;
      }
      // do not handle coplanar or general case here. Just let's use CGAL
      {
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
              if(fabs(pp.x() - xCoords[aggNodes[l]]) < 1e-12 &&
                 fabs(pp.y() - yCoords[aggNodes[l]]) < 1e-12 &&
                 fabs(pp.z() - zCoords[aggNodes[l]]) < 1e-12)
              {
                geomVerts_.push_back(aggNodes[l]);
                cntVertInAgg++;
                break;
              }
            }
          } while (++hit != fi->facet_begin());
          geomSizes_.push_back(cntVertInAgg);
        }
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
      vector<Point> aggPoints;
      LocalOrdinal aggStart = aggOffsets_[agg];
      LocalOrdinal aggEnd = aggOffsets_[agg + 1];
      LocalOrdinal aggSize = aggEnd - aggStart;
      //Handle point and line segment case
      //because make assumption later that alpha shape is a polygon
      if(aggSize == 1)
      {
        geomVerts_.push_back(aggVerts_[aggStart]);
        geomSizes_.push_back(1);
        continue;
      }
      else if(aggSize == 2)
      {
        geomVerts_.push_back(aggVerts_[aggStart]);
        geomVerts_.push_back(aggVerts_[aggStart + 1]);
        geomSizes_.push_back(2);
        continue;
      }
      aggPoints.reserve(aggSize);
      for(LocalOrdinal vi = aggStart; vi < aggEnd; vi++)
      {
        LocalOrdinal v = aggVerts_[vi];
        aggPoints.push_back(Point(x_[v], y_[v]));
      }
      Alpha_shape_2 hull(aggPoints.begin(), aggPoints.end());
      //Find smallest alpha value where alpha shape is one contiguous polygon
      Alpha_iterator it = hull.find_optimal_alpha(1);
      hull.set_alpha(*it);
      vector<Segment> cgalSegments;
      CGAL::alpha_edges(hull, back_inserter(segments));
      //map points back to vertices
      vector<pair<LocalOrdinal, LocalOrdinal>> segments;
      segments.reserve(cgalSegments.size());
      for(size_t j = 0; j < cgalSegments.size(); j++)
      {
        bool foundFirst = false;
        pair<LocalOrdinal, LocalOrdinal> seg(-1, -1);
        for(LocalOrdinal k = aggStart; k < aggEnd; k++)
        {
          LocalOrdinal v = aggVerts_[k];
          if(cgalSegments[j][0].x == x_[v] && cgalSegments[j][0].y == y_[v])
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
        geomVerts_.push_back(polyVerts[i]);
      }
      geomSizes_.push_back(polyVerts.size());
    }
  }

  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  cgalAlphaHulls3D()
  {
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Gt;
    typedef CGAL::Alpha_shape_cell_base_3<Gt> Fb;
    typedef CGAL::Triangulation_data_structure_3<Vb,Fb> Tds;
    typedef CGAL::Delaunay_triangulation_3<Gt,Tds> Triangulation_3;
    typedef Gt::Point_3 Point;
    typedef Alpha_shape_3::Alpha_iterator Alpha_iterator;
    typedef Alpha_shape_3::Cell_handle Cell_handle;
    typedef Alpha_shape_3::Vertex_handle Vertex_handle;
    typedef Alpha_shape_3::Facet Facet;
    typedef Alpha_shape_3::Edge Edge;
    typedef Gt::Weighted_point Weighted_point;
    typedef Gt::Bare_point Bare_point;
    using std::vector;
    for(LocalOrdinal agg = 0; agg < numLocalAggs_; agg++)
    {
      vector<Point> aggPoints;
      LocalOrdinal aggStart = aggOffsets_[agg];
      LocalOrdinal aggEnd = aggOffsets_[agg + 1];
      LocalOrdinal aggSize = aggEnd - aggStart;
      if(aggSize == 1)
      {
        geomVerts_.push_back(aggVerts_[aggStart]);
        geomSizes_.push_back(1);
        continue;
      }
      else if(aggSize == 2)
      {
        geomVerts_.push_back(aggVerts_[aggStart]);
        geomVerts_.push_back(aggVerts_[aggStart + 1]);
        geomSizes_.push_back(2);
        continue;
      }
      for(LocalOrdinal i = aggStart; i < aggEnd; i++)
      {
        LocalOrdinal v = aggVerts_[i];
        aggPoints.push_back(Point(x_[v], y_[v], z_[v]));
      }
      Fixed_alpha_shape_3 hull(aggPoints.begin(), aggPoints.end());
      vector<Cell_handle> cells;
      vector<Facet> facets;
      vector<Edge> edges;
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
              geomVerts_.push_back(aggVerts_[aggStart + ap]);
          }
        }
        geomSizes_.push_back(3);
      }
    }
  }

#endif

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  std::vector<int> AggGeometry<LocalOrdinal>::
  giftWrap(int aggStart, int aggEnd) {

    TEUCHOS_TEST_FOR_EXCEPTION(points.size() < 3, Exceptions::RuntimeError,
             "CoarseningVisualization::giftWrap: Gift wrap algorithm input has to have at least 3 points!");

    std::vector<int> hull;
    int aggSize = aggEnd - aggStart;
    //Find "minimum" point definitely in the convex hull
    int minPt = aggVerts_[aggStart];
    for(int i = 1; i < aggSize; i++)
    {
      int v = aggVerts_[aggStart + i];
      if(x_[v] < x_[minPt] || (x_[v] == x_[minPt] && y_[v] < y_[minPt]))
      {
        minPt = v;
      }
    }
    hull.push_back(minPt);
    std::vector<bool> inHull(aggSize, false);
    //walk around the boundary of the convex hull
    int current = minPt;
    do
    {
      //vertex that might be in the hull
      int cand;
      //pick any vertex not already in the hull
      for(int i = 0; i < aggSize; i++)
      {
        if(!inHull[i])
        {
          cand = aggVerts_[aggStart + i];
          break;
        }
      }
      //draw a line from current to cand,
      //and switch cand to any vertex not in hull that is "left" of that line
      //note: "left" defined by segmentNormal, which rotates a 2D vector 90 degrees to left
      Vec2 curVec(x_[current], y_[current]);
      for(int i = 0; i < aggSize; i++)
      {
        Vec2 candVec(x_[cand], y_[cand]);
        int iter = aggVerts_[aggStart + i];
        if(inHull[i] || iter == cand)
          continue;
        Vec2 iterVec(x_[iter], y_[iter]);
        Vec2 lineLeft = segmentNormal(candVec - curVec);
        //check if v is left of the line from current to cand
        double dotProd = dot(lineLeft, iterVec - curVec);
        if(dotProd > 1e-8)
        {
          //iter is definitely left of the line
          cand = iter;
        }
        else if(dotProd > -1e-8)
        {
          //iter is on the line, use it if it is further from current than cand
          if(dist(iterVec, curVec) > dist(candVec, curVec))
          {
            cand = iter;
          }
        }
      }
      //cand is the next vertex, add it to hull
      hull.push_back(cand);
    }
    while(current != minPt);
    //add hull to geometry list as one polygon
    for(size_t i = 0; i < hull.size(); i++)
    {
      geomVerts_.push_back(hull[i]);
    }
    geomSizes_.push_back(hull.size());

    /*
    double min_x =points[0].x;
    double min_y =points[0].y;
    for(std::vector<int>::iterator it = nodes.begin(); it != nodes.end(); it++) {
      int i = it - nodes.begin();
      if(points[i].x < min_x) min_x = points[i].x;
      if(points[i].y < min_y) min_y = points[i].y;
    }
    // create dummy min coordinates
    min_x -= 1.0;
    min_y -= 1.0;
    Vec2 dummy_min(min_x, min_y);

    // loop over all nodes and determine nodes with minimal distance to (min_x, min_y)
    std::vector<int> hull;
    Vec2 min = points[0];
    double mindist = distance(min,dummy_min);
    std::vector<int>::iterator minNode = nodes.begin();
    for(std::vector<int>::iterator it = nodes.begin(); it != nodes.end(); it++) {
      int i = it - nodes.begin();
      if(distance(points[i],dummy_min) < mindist) {
        mindist = distance(points[i],dummy_min);
        min = points[i];
        minNode = it;
      }
    }
    hull.push_back(*minNode);
    bool includeMin = false;
    //int debug_it = 0;
    while(1)
    {
      std::vector<int>::iterator leftMost = nodes.begin();
      if(!includeMin && leftMost == minNode)
      {
        leftMost++;
      }
      std::vector<int>::iterator it = leftMost;
      it++;
      for(; it != nodes.end(); it++)
      {
        if(it == minNode && !includeMin) //don't compare to min on very first sweep
          continue;
        if(*it == hull.back())
          continue;
        //see if it is in front of line containing nodes thisHull.back() and leftMost
        //first get the left normal of leftMost - thisHull.back() (<dy, -dx>)
        Vec2 leftMostVec = points[leftMost - nodes.begin()];
        Vec2 lastVec(xCoords[hull.back()], yCoords[hull.back()]);
        Vec2 testNorm = getNorm(vecSubtract(leftMostVec, lastVec));
        //now dot testNorm with *it - leftMost. If dot is positive, leftMost becomes it. If dot is zero, take one further from thisHull.back().
        Vec2 itVec(xCoords[*it], yCoords[*it]);
        double dotProd = dotProduct(testNorm, vecSubtract(itVec, lastVec));
        if(-1e-8 < dotProd && dotProd < 1e-8)
        {
          //thisHull.back(), it and leftMost are collinear.
          //Just sum the differences in x and differences in y for each and compare to get further one, don't need distance formula
          Vec2 itDist = vecSubtract(itVec, lastVec);
          Vec2 leftMostDist = vecSubtract(leftMostVec, lastVec);
          if(fabs(itDist.x) + fabs(itDist.y) > fabs(leftMostDist.x) + fabs(leftMostDist.y)) {
            leftMost = it;
          }
        }
        else if(dotProd > 0) {
          leftMost = it;

        }
      }
      //if leftMost is min, then the loop is complete.
      if(*leftMost == *minNode)
        break;
      hull.push_back(*leftMost);
      includeMin = true; //have found second point (the one after min) so now include min in the searches
      //debug_it ++;
      //if(debug_it > 100) exit(0); //break;
    }
    return hull;
    */
  }

  /*-----------------------------*/
  /* EdgeGeometry implementation */
  /*-----------------------------*/

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void EdgeGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  EdgeGeometry(Teuchos::RCP<GraphBase> G, int dofs, Teuchos::RCP<Matrix> A = Teuchos::null)
  {
    G_ = G;
    A_ = A;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void EdgeGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  build() {
    ArrayView<const Scalar> values;
    ArrayView<const LocalOrdinal> neighbors;

    std::vector<std::pair<int, int> > vert1; //vertices (node indices)

    ArrayView<const LocalOrdinal> indices;
    for(LocalOrdinal locRow = 0; locRow < LocalOrdinal(G->GetNodeNumVertices()); locRow++) {
      neighbors = G->getNeighborVertices(locRow);
      //Add those local indices (columns) to the list of connections (which are pairs of the form (localM, localN))
      for(int gEdge = 0; gEdge < int(neighbors.size()); ++gEdge) {
        vert1.push_back(std::pair<int, int>(locRow, neighbors[gEdge]));
      }
    }
    for(size_t i = 0; i < vert1.size(); i ++) {
      if(vert1[i].first > vert1[i].second) {
        int temp = vert1[i].first;
        vert1[i].first = vert1[i].second;
        vert1[i].second = temp;
      }
    }
    std::sort(vert1.begin(), vert1.end());
    std::vector<std::pair<int, int> >::iterator newEnd = unique(vert1.begin(), vert1.end()); //remove duplicate edges
    vert1.erase(newEnd, vert1.end());
    //std::vector<int> points1;
    geomVerts_.reserve(2 * vert1.size());
    geomSizes_.reserve(vert1.size());
    for(size_t i = 0; i < vert1.size(); i++) {
      geomVerts_.push_back(vert1[i].first);
      geomVerts_.push_back(vert1[i].second);
      geomSizes_.push_back(2);
    }
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void EdgeGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::build()
  {
    using namespace std;
    ArrayView<const Scalar> values;
    ArrayView<const LocalOrdinal> neighbors;
    if(A->isGloballyIndexed())
    {
      ArrayView<const GlobalOrdinal> indices;
      for(GlobalOrdinal globRow = 0; globRow < GlobalOrdinal(A->getGlobalNumRows()); globRow++)
      {
        if(dofs == 1)
          A->getGlobalRowView(globRow, indices, values);
        neighbors = G->getNeighborVertices((LocalOrdinal) globRow);
        int gEdge = 0;
        int aEdge = 0;
        while(gEdge != int(neighbors.size()))
        {
          if(dofs == 1)
          {
            if(neighbors[gEdge] == indices[aEdge])
            {
              //graph and matrix both have this edge, wasn't filtered
              vertsNonFilt_.push_back(pair<int, int>(int(globRow), neighbors[gEdge]));
              gEdge++;
              aEdge++;
            }
            else
            {
              //graph contains an edge at gEdge which was filtered from A
              vertsFilt_.push_back(pair<int, int>(int(globRow), neighbors[gEdge]));
              gEdge++;
            }
          }
          else 
          {
            //for multiple DOF problems, don't try to detect filtered edges and ignore A
            //TODO bmk: do detect them
            vertsNonFilt_.push_back(pair<int, int>(int(globRow), neighbors[gEdge]));
            gEdge++;
          }
        }
      }
    }
    else
    {
      ArrayView<const LocalOrdinal> indices;
      for(LocalOrdinal locRow = 0; locRow < LocalOrdinal(A->getNodeNumRows()); locRow++)
      {
        if(dofs == 1)
          A->getLocalRowView(locRow, indices, values);
        neighbors = G->getNeighborVertices(locRow);
        //Add those local indices (columns) to the list of connections (which are pairs of the form (localM, localN))
        int gEdge = 0;
        int aEdge = 0;
        while(gEdge != int(neighbors.size()))
        {
          if(dofs == 1)
          {
            if(neighbors[gEdge] == indices[aEdge])
            {
              vertsNonFilt_.push_back(pair<int, int>(locRow, neighbors[gEdge]));
              gEdge++;
              aEdge++;
            }
            else
            {
              vertsFilt_.push_back(pair<int, int>(locRow, neighbors[gEdge]));
              gEdge++;
            }
          }
          else
          {
            vertsNonFilt_.push_back(pair<int, int>(locRow, neighbors[gEdge]));
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
  VTKEmitter(const Teuchos::ParameterList>& pL, int numProcs, int level, int rank,
      const Teuchos::RCP<Map>& fineMap = Teuchos::null, const Teuchos::RCP<Map>& coarseMap = Teuchos::null);
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
    baseName_ = this->replaceAll(baseName_, "%LEVELID", toString(level));
    baseName_ = this->replaceAll(baseName_, "%TIMESTEP", toString(timeStep));
    baseName_ = this->replaceAll(baseName_, "%ITER", toString(iter));
    this->rank = rank;
    this->fineMap = fineMap;
    this->coarseMap = coarseMap;
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
  writeAggGeom(AggGeometry& ag)
  {
    //note: makeUnique modifies ag.geomVerts_ in place, but OK because it won't be used again
    std::vector<int> uniqueVerts = makeUnique(ag.geomVerts_);
    std::ofstream fout(getAggFilename());
    writeOpening(fout, uniqueVerts.size(), ag.geomSizes_.size());
    writeNodes(fout, uniqueVerts, fineMap_);
    writeAggData(fout, uniqueVerts, ag.vertex2Agg_);
    writeCoordinates(fout, uniqueVerts, ag.x_, ag.y_, ag.z_);
    writeCells(fout, ag.geomVerts_, ag.geomSizes_);
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
  writeEdgeGeom(EdgeGeometry& eg, bool fine)
  {
    std::vector<int> uniqueVerts = makeUnique(eg.geomVerts_);
    //get filename
    string filename = fine ? getFineEdgeFilename() : getCoarseEdgeFilename();
    std::ofstream fout(filename);
    writeOpening(fout, uniqueVerts.size(), eg.geomSizes_.size());
    writeNodes(fout, uniqueVerts, fine ? fineMap_ : coarseMap_);
    writeEdgeData(fout, uniqueVerts, ag.vertex2Agg_);
    writeCoordinates(fout, uniqueVerts, eg.x_, eg.y_, eg.z_);
    writeCells(fout, eg.geomVerts_, eg.geomSizes_);
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
  writeOpening(std::ofstream& fout, int numVerts, int numCells) {
    std::string indent = "      ";
    fout << "<!--MueLu Aggregates/Coarsening Visualization-->" << std::endl;
    fout << "<VTKFile type=\"UnstructuredGrid\" byte_order=\"LittleEndian\">" << std::endl;
    fout << "  <UnstructuredGrid>" << std::endl;
    fout << "    <Piece NumberOfPoints=\"" << numVerts << "\" NumberOfCells=\"" << numCells << "\">" << std::endl;
    fout << "      <PointData Scalars=\"Node Aggregate Processor\">" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeNodes(std::ofstream& fout, const std::vector<int>& uniqueVerts, const Teuchos::RCP<Map>& map) {
    std::string indent = "      ";
    fout << "        <DataArray type=\"Int32\" Name=\"Node\" format=\"ascii\">" << std::endl;
    indent = "          ";
    bool localIsGlobal = GlobalOrdinal(map->getGlobalNumElements()) == GlobalOrdinal(map->getNodeNumElements());
    for(size_t i = 0; i < uniqueVerts.size(); i++)
    {
      if(localIsGlobal)
      {
        fout << uniqueVerts[i] << " "; //if all nodes are on this processor, do not map from local to global
      }
      else
        fout << map->getGlobalElement(uniqueVerts[i]) << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeAggData(std::ofstream& fout, const std::vector<int>& uniqueVerts, const ArrayRCP<const LocalOrdinal>& vertex2AggId) {
    std::string indent = "          ";
    fout << "        <DataArray type=\"Int32\" Name=\"Aggregate\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < uniqueFine.size(); i++)
    {
      fout << myAggOffset + vertex2AggId[uniqueVerts[i]] << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"Processor\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < uniqueFine.size(); i++)
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
  writeEdgeData(std::ofstream& fout, const std::vector<int>& uniqueVerts)
  {
    std::string indent = "          ";
    fout << "        <DataArray type=\"Int32\" Name=\"Aggregate\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < uniqueFine.size(); i++)
    {
      fout << myAggOffset + vertex2AggId[uniqueVerts[i]] << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"Processor\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < uniqueFine.size(); i++)
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
  writeCoordinates(std::ofstream& fout, const std::vector<int>& uniqueVerts, const Teuchos::ArrayRCP<const double>& x, const Teuchos::ArrayRCP<const double>& y, const Teuchos::ArrayRCP<const double>& z) const {
    std::string indent = "      ";
    fout << "      <Points>" << std::endl;
    fout << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < uniqueFine.size(); i++)
    {
      fout << x[uniqueVerts[i]] << " " << y[uniqueVerts[i]] << " ";
      if(z.is_null())
        fout << "0 ";
      else
        fout << z[uniqueVerts[i]] << " ";
      //write 3 coordinates per line
      if(i % 3 == 2)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "      </Points>" << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  writeCells(std::ofstream& fout, const std::vector<int>& geomVerts, const std::vector<int>& geomSizes) const {
    std::string indent = "      ";
    fout << "      <Cells>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
    fout << indent;
    for(size_t i = 0; i < geomVerts_.size(); i++)
    {
      fout << geomVerts[i] << " ";
      if(i % 10 == 9)
        fout << std::endl << indent;
    }
    fout << std::endl;
    fout << "        </DataArray>" << std::endl;
    fout << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
    fout << indent;
    int accum = 0;
    for(size_t i = 0; i < geomSize.size(); i++)
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
  writeFileVTKClosing(std::ofstream& fout) const {
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
          return;
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
  void VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>::buildColormap() const {
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

} // namespace VizHelpers
} // namespace MueLu

#endif /* MUELU_VISUALIZATIONHELPERS_DEF_HPP_ */

