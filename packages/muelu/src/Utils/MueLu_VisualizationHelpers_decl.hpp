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

#ifndef MUELU_VISUALIZATIONHELPERS_DECL_HPP_
#define MUELU_VISUALIZATIONHELPERS_DECL_HPP_

#include <Xpetra_Matrix_fwd.hpp>
#include <Xpetra_CrsMatrixWrap_fwd.hpp>

#include "MueLu_ConfigDefs.hpp"
#include "MueLu_TwoLevelFactoryBase.hpp"
#include "MueLu_VisualizationHelpers_fwd.hpp"
#include "MueLu_Aggregates_fwd.hpp"
#include "MueLu_Graph_fwd.hpp"
#include "MueLu_GraphBase.hpp"
#include "MueLu_AmalgamationFactory_fwd.hpp"
#include "MueLu_AmalgamationInfo_fwd.hpp"
#include "MueLu_Utilities_fwd.hpp"

#include <list>

namespace MueLu {
namespace VizHelpers {
  //Utility classes used in convex hull algorithm

  struct Triangle
  {
    Triangle() : v1(0), v2(0), v3(0) {}
    Triangle(int v1in, int v2in, int v3in) : v1(v1in), v2(v2in), v3(v3in) {}
    ~Triangle() {}
    bool operator==(const Triangle& l)
    {
      if(l.v1 == v1 && l.v2 == v2 && l.v3 == v3)
        return true;
      return false;
    }
    int v1;
    int v2;
    int v3;
  };

  class Vec3
  {
    public:
      Vec3() : x(0), y(0), z(0) {}
      Vec3(double xin, double yin, double zin) : x(xin), y(yin), z(zin) {}
      ~Vec3() {}
      double x;
      double y;
      double z;
  };

  Vec3 operator-(const Vec3 lhs, const vec3 rhs)
  {
    return Vec3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
  }

  class Vec2
  {
    public:
      Vec2() : x(0), y(0) {}
      Vec2(double xin, double yin) : x(xin), y(yin) {}
      ~Vec2() {}
      double x;
      double y;
  };

  Vec2 operator-(const Vec2 lhs, const vec2 rhs)
  {
    return Vec2(lhs.x - rhs.x, lhs.y - rhs.y);
  }

  Vec3 crossProduct(Vec3 v1, Vec3 v2);
  double dotProduct(Vec2 v1, Vec2 v2);
  double dotProduct(Vec3 v1, Vec3 v2);
  bool isInFront(Vec3 point, Vec3 inPlane, Vec3 n);
  bool collinear(Vec2 v1, Vec2 v2, Vec2 v3);
  double mag(Vec2 vec);
  double mag(Vec3 vec);
  double dist(Vec2 p1, Vec2 p2);
  double dist(Vec3 p1, Vec3 p2);
  //! Get "normal" to given 2D vector - rotate left 90 degrees
  Vec2 segmentNormal(Vec2 v);
  //! Get normal to triangle (oriented outward)
  Vec3 triNormal(Vec3 v1, Vec3 v2, Vec3 v3);
  double pointDistFromTri(Vec3 point, Vec3 v1, Vec3 v2, Vec3 v3);

  std::string replaceAll(std::string original, std::string replaceWhat, std::string replaceWithWhat);

  //! Replaces node indices in vertices in place with compressed unique indices, and returns list of unique points
  //! Elements of vertices (originally local rows/vertices) are replaced by indices of returned vector
  std::vector<int> makeUnique(std::vector<int>& verts); 

  //! Make list of unique vertices from union of two vertex lists, and replace elements of the two
  //! lists with indices of the unique list
  std::vector<int> mergeAndMakeUnique(std::vector<int>& verts1, std::vector<int>& verts2); 

  /*!
    @class AggGeoemtry class.
    @brief Generates geometry for visualizing aggregates and coarsening information.

    This class is used by the CoarseningVisualizationFactory as well as the AggregationExporterFactory to
    visualize aggregates or coarsening information from the transfer operators.
  */

  /* vertex -> agg mapping is not a function for "bubbles" in CoarseningVisualizationFactory (vertices may be included in >1 aggregates for viz purposes)
   * vertex2AggId from Aggregates doesn't represent that mapping
   * instead, have cumulative "aggregate offset" list with each unique local agg id, which index a long list of vertices ("aggVerts")
   * vertices can appear multiple times in aggVerts
   * adapt all geometry code to be correct with this format
   * in old usage case, aggVerts is effectively same as vertex2AggId (but in different order) and aggregate offset is just cumulative sum of agg sizes
   */

  template <class Scalar = double, class LocalOrdinal = int, class GlobalOrdinal = LocalOrdinal, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
  class AggGeometry {
#undef MUELU_VISUALIZATIONHELPERS_SHORT
#include "MueLu_UseShortNames.hpp"
    friend class VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    public:
      //! @name Constructors/Destructors.
      //@{

      //! Constructor used by AggregationExportFactory
      //! Map is Coordinates non-overlapping map that describes vertex-processor association
      //! Number of dimensions inferred from whether z is null
      AggGeometry(const Teuchos::RCP<Aggregates>& aggs, const Teuchos::RCP<Map>& map, const Teuchos::RCP<Teuchos::Comm>& comm,
          const Teuchos::RCP<MultiVector>& coords);

      //! Constructor used by CoarseningVisualizationFactory 
      //! P can be either smoothed or tentative prolongator
      //! Map is Coordinates non-overlapping map that describes vertex-processor association
      //! Aggregates can be overlapping, in the case of smoothed P
      //! Number of dimensions inferred from whether z is null
      AggGeometry(const Teuchos::RCP<Matrix>& P, const Teuchos::RCP<Map>& map, const Teuchos::RCP<Teuchos::Comm>& comm,
          const Teuchos::RCP<MultiVector>& coords, LocalOrdinal dofsPerNode, LocalOrdinal colsPerNode);

      //! Generate the geometry. style is the "visualization: agg style" parameter value, and doesn't need to be valid.
      //! If style not valid, default to Point Cloud and return false.
      bool build(std::string& style);

      //@}
    private:
      std::vector<int> geomVerts_;
      std::vector<int> geomSizes_;
      Teuchos::RCP<Aggregates> aggs_;
      Teuchos::RCP<Map> map_;
      Teuchos::ArrayRCP<const LocalOrdinal> vertex2Agg_;
      Teuchos::ArrayRCP<LocalOrdinal> aggVerts_;
      Teuchos::ArrayRCP<LocalOrdinal> aggOffsets_;
      //! For each local vertex, whether it is the root of its aggregate
      //! Only used by jacks(), so will only be populated if jacks() is called.
      std::vector<bool> isRoot_;
      LocalOrdinal numLocalAggs_;
      LocalOrdinal numNodes_;
      //! Global id of local aggregate 0
      GlobalOrdinal firstAgg_;
      Teuchos::ArrayRCP<const double> x_;
      Teuchos::ArrayRCP<const double> y_;
      Teuchos::ArrayRCP<const double> z_;    //null if 2D
      int dims_;
      int rank_;
      int nprocs_;
      //! false if doing geometry based on aggregates (AggExport) or Ptent (CoarseningViz)
      //! true if doing geometry based on smoothed P nonzero entries (CoarseningViz)
      bool bubbles_;

      // algorithm implementations
      void pointCloud();
      void jacks();
      //call cgalConvexHulls if available
      void convexHulls2D();
      void convexHulls3D();

      //used by convexHulls2D
      std::vector<int> giftWrap(int aggStart, int aggEnd);

      //used by jacks
      void computeIsRoot();

#ifdef HAVE_MUELU_CGAL
      void cgalConvexHulls2D();
      void cgalConvexHulls3D();
      void cgalAlphaHulls2D();
      void cgalAlphaHulls3D();
#endif
      //Internal geometry utilities
      std::vector<Triangle> processTriangle(std::list<Triangle>& tris, Triangle tri, std::list<int>& pointsInFront, Vec3& barycenter);
  };

  template <class Scalar = double, class LocalOrdinal = int, class GlobalOrdinal = LocalOrdinal, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
  class EdgeGeometry {
#undef MUELU_VISUALIZATIONHELPERS_SHORT
#include "MueLu_UseShortNames.hpp"
    friend class VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    public:
      EdgeGeometry(Teuchos::RCP<GraphBase> G, Teuchos::RCP<Matrix> A = Teuchos::null, int dofs);
      //! Compute graph edge geometry. If the matrix A was available and passed to constructor, filtered edges will be given a different color than non-filtered edges.
      void build();
    private:
      const Teuchos::RCP<GraphBase> G_;
      const Teuchos::RCP<Matrix> A_;
      //! Vertices for non-filtered edges
      std::vector<int> vertsNonFilt_;
      //! Vertices for filtered edges 
      std::vector<int> vertsFilt_;
      int aggsOffset_;            
      //! Special node index value representing non-filtered edges (in both graph and matrix)
      static const int contrast1_ = -1;
      //! Special node index value representing filtered edges (in graph but not filtered matrix)
      static const int contrast2_ = -2;
  };

  //! Class for writing geometry into VTK files.
  template <class Scalar = double, class LocalOrdinal = int, class GlobalOrdinal = LocalOrdinal, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
  class VTKEmitter {
#undef MUELU_VISUALIZATIONHELPERS_SHORT
#include "MueLu_UseShortNames.hpp"
    private:
      typedef AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeometry;
      typedef EdgeGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> EdgeGeometry;
    public:
      VTKEmitter(const Teuchos::ParameterList>& pL, int numProcs, int level, int rank, const Teuchos::RCP<Map>& fineMap = Teuchos::null, const Teuchos::RCP<Map>& coarseMap = Teuchos::null);
      //! Write one VTK file per process with ag's geometry data (also does bubbles)
      //! Note: filename automatically generated with proc id, level id
      void writeAggGeom(AggGeometry& ag);
      //! Write one VTK file per process with eg's geometry data, using fine or coarse map
      void writeEdgeGeom(EdgeGeometry& eg, bool fine);
      //! Write PVTU linking to all previously written VTU files, allowing all geometry to be viewed together
      void writePVTU();
      void buildColormap();
    private:
      void writeOpening(std::ofstream& fout, int numVerts, int numCells);
      void writeNodes(std::ofstream& fout, const std::vector<int>& uniqueVerts, const Teuchos::RCP<Map>& map);
      void writeAggData(std::ofstream& fout, const std::vector<int>& uniqueVerts, const Teuchos::ArrayRCP<LocalOrdinal>& vertex2AggId);
      void writeEdgeData(std::ofstream& fout, const std::vector<int>& uniqueVerts);
      void writeCoordinates(std::ofstream& fout, const std::vector<int>& uniqueVerts, const Teuchos::ArrayRCP<const double>& x, const Teuchos::ArrayRCP<const double>& y, const Teuchos::ArrayRCP<const double>& z);
      void writeCells(std::ofstream& fout, const std::vector<int>& geomVerts, const std::vector<int>& geomSizes);
      void writeClosing(std::ofstream& fout);
      //! Generate filename to use for main aggregate geometry file.
      std::string getAggFilename(int proc = rank_);
      //! Generate filename to use for bubble (secondary aggregate) geometry file.
      std::string getBubbleFilename(int proc = rank_);
      //! Generate filename to use for fine edge geometry file.
      std::string getFineEdgeFilename(int proc = rank_);
      //! Generate filename to use for coarse edge geometry file.
      std::string getCoarseEdgeFilename(int proc = rank_);
      //! Generate filename for main PVTU file.
      std::string getPVTUFilename();
      //! Base filename (no file extension) used for all VTU output files
      //! Will have %LEVELID, %TIMESTEP and %ITER substituted with values, but not %PROCID
      std::string baseName_;
      int rank_;
      int nprocs_;
      const Teuchos::RCP<Map>& fineMap_;
      const Teuchos::RCP<Map>& coarseMap_;
      bool didAggs_;
      bool didBubbles_;
      bool didFineEdges_;
      bool didCoarseEdges_;
  };
} // namespace VizHelpers
} // namespace MueLu

#define MUELU_VISUALIZATIONHELPERS_SHORT

#endif /* MUELU_VISUALIZATIONHELPERS_DECL_HPP_ */
