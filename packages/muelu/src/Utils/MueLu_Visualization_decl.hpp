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
#include "MueLu_Aggregates.hpp"
#include "MueLu_Graph_fwd.hpp"
#include "MueLu_GraphBase.hpp"
#include "MueLu_AmalgamationFactory_fwd.hpp"
#include "MueLu_AmalgamationInfo_fwd.hpp"
#include "MueLu_Utilities_fwd.hpp"

#include <list>

namespace MueLu {
namespace VizHelpers {
  //Geometry utility classes (used in convex hull algorithm)

  struct Vec2;
  struct Vec3;

  double mag(Vec2 vec);
  double mag(Vec3 vec);

  struct Vec3
  {
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double xin, double yin, double zin) : x(xin), y(yin), z(zin) {}
    ~Vec3() {}
    double x;
    double y;
    double z;
    inline Vec2 toVec2();
    //Get a unit vector in direction of *this
    Vec3 normalize()
    {
      if(x != 0 || y != 0 || z != 0)
      {
        double magnitude = mag(*this);
        return Vec3(x / magnitude, y / magnitude, z / magnitude);
      }
      else
      {
        return *this;
      }
    }
  };

  inline Vec3 operator+(const Vec3 lhs, const Vec3 rhs)
  {
    return Vec3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
  }

  inline Vec3 operator-(const Vec3 lhs, const Vec3 rhs)
  {
    return Vec3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
  }

  inline Vec3 operator*(const Vec3 vec, double scale)
  {
    return Vec3(vec.x * scale, vec.y * scale, vec.z * scale);
  }

  inline void operator+=(Vec3& lhs, const Vec3 rhs)
  {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
  }

  inline void operator*=(Vec3& vec, double scale)
  {
    vec.x *= scale;
    vec.y *= scale;
    vec.z *= scale;
  }

  struct Vec2
  {
    Vec2() : x(0), y(0) {}
    Vec2(double xin, double yin) : x(xin), y(yin) {}
    ~Vec2() {}
    double x;
    double y;
    //Get a unit vector in direction of *this
    Vec2 normalize()
    {
      if(x != 0 || y != 0)
      {
        double magnitude = mag(*this);
        return Vec2(x / magnitude, y / magnitude);
      }
      else
      {
        return *this;
      }
    }
  };

  inline Vec2 Vec3::toVec2()
  {
    return Vec2(x, y);
  }

  inline Vec2 operator+(const Vec2 lhs, const Vec2 rhs)
  {
    return Vec2(lhs.x + rhs.x, lhs.y + rhs.y);
  }

  inline Vec2 operator-(const Vec2 lhs, const Vec2 rhs)
  {
    return Vec2(lhs.x - rhs.x, lhs.y - rhs.y);
  }

  inline bool operator==(const Vec2 lhs, const Vec2 rhs)
  {
    return lhs.x == rhs.x && lhs.y == rhs.y;
  }

  inline bool operator!=(const Vec2 lhs, const Vec2 rhs)
  {
    return lhs.x != rhs.x || lhs.y != rhs.y;
  }

  inline Teuchos::RCP<Teuchos::ParameterList> GetVizParameterList();

  inline Vec3 crossProduct(Vec3 v1, Vec3 v2);
  inline double dotProduct(Vec2 v1, Vec2 v2);
  inline double dotProduct(Vec3 v1, Vec3 v2);
  inline bool isInFront(Vec3 point, Vec3 inPlane, Vec3 n);
  inline bool collinear(Vec2 v1, Vec2 v2, Vec2 v3);
  inline double dist(Vec2 p1, Vec2 p2);
  inline double dist(Vec3 p1, Vec3 p2);
  //! Get "normal" to given 2D vector - rotate left 90 degrees
  inline Vec2 segmentNormal(Vec2 v);
  //! Get normal to triangle (oriented outward)
  inline Vec3 triNormal(Vec3 v1, Vec3 v2, Vec3 v3);
  inline double pointDistFromTri(Vec3 point, Vec3 v1, Vec3 v2, Vec3 v3);
  inline double pointDistFromLine(Vec3 point, Vec3 line1, Vec3 line2);

  inline std::string replaceAll(std::string original, std::string replaceWhat, std::string replaceWithWhat);

  //used by convexHulls2D and convexHulls3D (in case of agg with all-coplanar points)
  //precondition: all points are coplanar (but the plane is allowed to have any 3D orientation)
  template<typename GlobalOrdinal>
  std::vector<GlobalOrdinal> giftWrap(std::vector<GlobalOrdinal>& points, std::map<GlobalOrdinal, Vec3>& verts);

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

  template<typename GlobalOrdinal>
  struct GeometryPoint
  {
    GeometryPoint()
    {
      vert = 0;
      agg = 0;
    }
    GeometryPoint(GlobalOrdinal geomVert, GlobalOrdinal geomAgg)
    {
      vert = geomVert;
      agg = geomAgg;
    }
    GlobalOrdinal vert;
    GlobalOrdinal agg;
  };

  template <class Scalar = double, class LocalOrdinal = int, class GlobalOrdinal = LocalOrdinal, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
  class AggGeometry {
#undef MUELU_VISUALIZATIONHELPERS_SHORT
#include "MueLu_UseShortNames.hpp"
    public:
      //! Type of coordinates array (for input only).
      //! Always has double scalar type, no matter the matrix scalar type.
      typedef Xpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> CoordArray;
      typedef MueLu::Aggregates<LocalOrdinal, GlobalOrdinal, Node> Aggs;
      typedef GeometryPoint<GlobalOrdinal> GeomPoint;
      //! @name Constructors/Destructors.
      //@{

      //! Constructor used by AggregationExportFactory
      //! Map is Coordinates non-overlapping map that describes vertex-processor association
      //! Number of dimensions inferred from whether z is null
      AggGeometry(const Teuchos::RCP<Aggs>& aggs, const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
          const Teuchos::RCP<CoordArray>& coords);

      //! Constructor used by CoarseningVisualizationFactory 
      //! P can be either smoothed or tentative prolongator
      //! Map is Coordinates non-overlapping map that describes vertex-processor association
      //! Aggregates can be overlapping, in the case of smoothed P
      //! Number of dimensions inferred from whether z is null
      AggGeometry(const Teuchos::RCP<Matrix>& P, const Teuchos::RCP<const Map>& map, const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
          const Teuchos::RCP<CoordArray>& coords, LocalOrdinal dofsPerNode, LocalOrdinal colsPerNode, bool ptent);

      //! Constructor used to create a single artificial aggregate (for testing)
      AggGeometry(std::vector<Vec3>& coords, int dims);

      //! Generate the geometry. style is the "visualization: agg style" parameter value, and doesn't need to be valid.
      //! If style not valid, default to Point Cloud and return false.
      bool build(std::string& style);

      //@}

      std::vector<GeomPoint> geomVerts_;
      std::vector<int> geomSizes_;
      //The row map of A
      Teuchos::RCP<const Map> map_;
      //The aggregates, if available
      Teuchos::RCP<Aggs> aggs_;
      Teuchos::Array<GlobalOrdinal> aggVerts_;
      Teuchos::Array<GlobalOrdinal> aggOffsets_;
      //Locally used set of global rows is not contiguous, so use a map to look up positions
      std::map<GlobalOrdinal, Vec3> verts_;
      //! For each locally used vertex, whether it is the root of its aggregate
      //! Only used by jacks(), so will only be populated if jacks() is called.
      //! If roots not available from aggregates, use vertex nearest to centroid of aggregate
      std::map<GlobalOrdinal, bool> isRoot_;
      LocalOrdinal numLocalAggs_;
      GlobalOrdinal numNodes_;
      //! Global id of local aggregate 0 (note: aggregate indices are always contiguous)
      GlobalOrdinal firstAgg_;
      int dims_;
      int rank_;
      int nprocs_;
      //! false if doing geometry based on aggregates (AggExport) or Ptent (CoarseningViz)
      //! true if doing geometry based on smoothed P (CoarseningViz)
      bool bubbles_;

      // algorithm implementations
      void pointCloud();
      void jacks();
      //call cgalConvexHulls if available, otherwise use custom gift wrapping implementation
      void convexHulls2D();
      //quickhull algorithm, as described at thomasdiewald.com/blog/?p=1888
      void convexHulls3D();

      //used by all 3D convex hull and alpha shape functions to deal with collinear/coplanar nodes
      bool handleDegenerate(std::vector<GlobalOrdinal>& points, int agg, bool is3D = true);

      //used by jacks only (called lazily when roots are needed)
      void computeIsRoot();

#ifdef HAVE_MUELU_CGAL
      void cgalConvexHulls2D();
      void cgalConvexHulls3D();
      void cgalAlphaHulls2D();
      void cgalAlphaHulls3D();
#endif
      struct Triangle
      {
        Triangle() : v1(-1), v2(-1), v3(-1), valid(false)
        {
          valid = false;
          nei[0] = -1;
          nei[1] = -1;
          nei[2] = -1;
        }
        Triangle(GlobalOrdinal v1in, GlobalOrdinal v2in, GlobalOrdinal v3in, int nei1, int nei2, int nei3) : v1(v1in), v2(v2in), v3(v3in)
        {
          valid = false;
          nei[0] = nei1;
          nei[1] = nei2;
          nei[2] = nei3;
        }
        ~Triangle() {}
        bool operator==(const Triangle& l)
        {
          if(l.v1 == v1 && l.v2 == v2 && l.v3 == v3)
            return true;
          return false;
        }
        void setPointList(std::vector<GlobalOrdinal>& pts)
        {
          frontPoints = pts;
        }
        bool hasNeighbor(int n)
        {
          for(int i = 0; i < 3; i++)
          {
            if(nei[i] == n)
              return true;
          }
          return false;
        }
        void replaceNeighbor(int toReplace, int with)
        {
          //if already have with, don't duplicate it
          if(hasNeighbor(with))
            return;
          //if don't already have toReplace, can't replace it
          if(!hasNeighbor(toReplace))
            return;
          for(int i = 0; i < 3; i++)
          {
            if(nei[i] == toReplace)
            {
              nei[i] = with;
              return;
            }
          }
          //throw std::runtime_error("Tri: tried to replace neighbor " +
          //    std::to_string(toReplace) + " with " + std::to_string(with) +
          //    " but didn't have the one to replace.");
        }
        void removeNeighbor(int n)
        {
          replaceNeighbor(n, -1);
        }
        void addNeighbor(int n)
        {
          replaceNeighbor(-1, n);
        }
        bool adjacent(Triangle& tri)
        {
          int shared = 0;
          GlobalOrdinal thisVerts[] = {v1, v2, v3};
          GlobalOrdinal otherVerts[] = {tri.v1, tri.v2, tri.v3};
          for(int i = 0; i < 3; i++)
          {
            for(int j = 0; j < 3; j++)
            {
              if(thisVerts[i] == otherVerts[j])
                shared++;
            }
          }
          if(shared == 2)
          {
            return true;
          }
          else
          {
            return false;
          }
        }
        GlobalOrdinal v1;
        GlobalOrdinal v2;
        GlobalOrdinal v3;
        int nei[3];
        std::vector<GlobalOrdinal> frontPoints;
        bool valid;
      };
      bool pointInFront(Triangle& t, GlobalOrdinal p)
      {
        return pointDistFromTri(verts_[p], verts_[t.v1], verts_[t.v2], verts_[t.v3]) > 0;
      }
  };

  template <class Scalar = double, class LocalOrdinal = int, class GlobalOrdinal = LocalOrdinal, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
  class EdgeGeometry {
#undef MUELU_VISUALIZATIONHELPERS_SHORT
#include "MueLu_UseShortNames.hpp"
    public:
      //EdgeGeometry has the same coordinates format as AggGeometry
      typedef typename AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>::CoordArray CoordArray;
      EdgeGeometry(Teuchos::RCP<CoordArray>& coords, Teuchos::RCP<GraphBase>& G, int dofs, Teuchos::RCP<Matrix> A = Teuchos::null);
      //! Compute graph edge geometry. If the matrix A was available and passed to constructor, filtered edges will be given a different color than non-filtered edges.
      void build();
      Teuchos::RCP<GraphBase> G_;
      Teuchos::RCP<Matrix> A_;
      int dofs_;
      //Note: vertsFilt/vertsNonFilt are not accompanied by local aggregate index because all edge data attribs are contrast1/contrast2
      //! Vertices for non-filtered edges
      std::vector<GlobalOrdinal> vertsNonFilt_;
      //! Vertices for filtered edges 
      std::vector<GlobalOrdinal> vertsFilt_;
      std::map<GlobalOrdinal, Vec3> verts_;
  };

  //! Class for writing geometry into VTK files.
  template <class Scalar = double, class LocalOrdinal = int, class GlobalOrdinal = LocalOrdinal, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
  class VTKEmitter {
#undef MUELU_VISUALIZATIONHELPERS_SHORT
#include "MueLu_UseShortNames.hpp"
    public:
      typedef AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeom;
      typedef EdgeGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> EdgeGeom;
      typedef GeometryPoint<GlobalOrdinal> GeomPoint;
      VTKEmitter(const Teuchos::ParameterList& pL, int numProcs, int level, int rank, Teuchos::RCP<const Map> fineMap = Teuchos::null, Teuchos::RCP<const Map> coarseMap = Teuchos::null);
      //! Write one VTK file per process with ag's geometry data (also does bubbles)
      //! Note: filename automatically generated with proc id, level id
      void writeAggGeom(AggGeom& ag);
      //! Write one VTK file per process with eg's geometry data, using fine or coarse map
      void writeEdgeGeom(EdgeGeom& eg, bool fine);
      //! Write PVTU linking to all previously written VTU files, allowing all geometry to be viewed together
      void writePVTU();
      void buildColormap();
    private:
      //Replace global vertex IDs with indices into indices for minimal local set of vertices
      //the minimal set is returned and will become the set of points written to VTK
      std::vector<GeomPoint> getUniqueAggGeom(std::vector<GeomPoint>& geomPoints);
      std::vector<GlobalOrdinal> getUniqueEdgeGeom(std::vector<GlobalOrdinal>& edges);  //0,1 form edge, 2,3 form edge, etc.
      void writeOpening(std::ofstream& fout, size_t numVerts, size_t numCells);
      //! Write out GIDs for agg geometry
      void writeAggNodes(std::ofstream& fout, std::vector<GeomPoint>& uniqueVerts);
      //! Write out GIDs for edge geometry
      void writeEdgeNodes(std::ofstream& fout, std::vector<GlobalOrdinal>& uniqueVertsNonFilt, std::vector<GlobalOrdinal>& uniqueVertsFilt);
      //! Write out aggregate and process information for each vertex (firstAgg is the minimum global agg ID that is locally owned)
      void writeAggData(std::ofstream& fout, std::vector<GeomPoint>& uniqueVerts, GlobalOrdinal firstAgg);
      void writeEdgeData(std::ofstream& fout, size_t vertsNonFilt, size_t vertsFilt);
      void writeCoordinates(std::ofstream& fout, std::vector<GeomPoint>& uniqueVerts, std::map<GlobalOrdinal, Vec3>& positions);
      void writeCoordinates(std::ofstream& fout,
          std::vector<GlobalOrdinal>& uniqueVertsNonFilt, std::vector<GlobalOrdinal>& uniqueVertsFilt,
          std::map<GlobalOrdinal, Vec3>& positions);
      void writeAggCells(std::ofstream& fout, std::vector<GeomPoint>& geomVerts, std::vector<int>& geomSizes);
      void writeEdgeCells(std::ofstream& fout, std::vector<GlobalOrdinal>& vertsNonFilt, std::vector<GlobalOrdinal>& vertsFilt, size_t numUniqueNonFilt);
      void writeClosing(std::ofstream& fout);
      //! Generate filename to use for main aggregate geometry file.
      std::string getAggFilename(int proc);
      //! Generate filename to use for bubble (secondary aggregate) geometry file.
      std::string getBubbleFilename(int proc);
      //! Generate filename to use for fine edge geometry file.
      std::string getFineEdgeFilename(int proc);
      //! Generate filename to use for coarse edge geometry file.
      std::string getCoarseEdgeFilename(int proc);
      //! Generate filename for main PVTU file.
      std::string getPVTUFilename();
      //! Base filename (no file extension) used for all VTU output files
      //! Will have %LEVELID, %TIMESTEP and %ITER substituted with values, but not %PROCID
      std::string baseName_;
      int rank_;
      int nprocs_;
      Teuchos::RCP<const Map> fineMap_;
      Teuchos::RCP<const Map> coarseMap_;
      bool didAggs_;
      bool didBubbles_;
      bool didFineEdges_;
      bool didCoarseEdges_;
  };
} // namespace VizHelpers
} // namespace MueLu

#define MUELU_VISUALIZATIONHELPERS_SHORT

#endif /* MUELU_VISUALIZATIONHELPERS_DECL_HPP_ */

