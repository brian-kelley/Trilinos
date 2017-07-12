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
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_TestingHelpers.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <MueLu_TestHelpers.hpp>
#include <MueLu_Version.hpp>

#include <MueLu_FactoryManagerBase.hpp>
#include <MueLu_CoupledAggregationFactory.hpp>
#include <MueLu_UncoupledAggregationFactory.hpp>
#include <MueLu_Aggregates.hpp>
#include <MueLu_Visualization_def.hpp>

#include <random>

namespace MueLuTests {

  template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
  void emitAggGeom(MueLu::VizHelpers::AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node>& ag)
  {
    using Teuchos::ParameterList;
    using std::string;
    typedef MueLu::VizHelpers::VTKEmitter<Scalar, LocalOrdinal, GlobalOrdinal, Node> VTK;
    ParameterList p;
    string vizFnameParam = "visualization: output filename";
    string vizFname = "ConvexHull_Test.vtu";
    p.set<string>(vizFnameParam, vizFname);
    VTK vtk(p, 1, 0, 0);
    vtk.writeAggGeom(ag);
  }

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(VizGeom, PointCloud, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    using MueLu::VizHelpers::Vec3;
    typedef MueLu::VizHelpers::AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeom;
    //Generate 1000 points in a cubic 3D grid representing a single aggregate
    std::vector<Vec3> pts;
    for(int i = 0; i < 1000; i++)
    {
      pts.emplace_back(i % 10, (i / 10) % 10, i / 100);
    }
    AggGeom ag(pts, 3);
    std::string style = "Point Cloud";
    ag.build(style);
    //check geometry
    //make sure there are 1000 total geometry pieces, each with exactly one point
    TEST_EQUALITY(ag.geomVerts_.size(), 1000);
    TEST_EQUALITY(ag.geomSizes_.size(), 1000);
    for(int i = 0; i < 1000; i++)
    {
      TEST_EQUALITY(ag.geomSizes_[i], 1);
    }
  }

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(VizGeom, Jacks, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    using MueLu::VizHelpers::Vec3;
    typedef MueLu::VizHelpers::AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeom;
    //Make an agg out of the origin and nonRoot points around the unit circle
    const int nonRoot = 20;
    std::vector<Vec3> pts;
    pts.emplace_back(0, 0, 0);
    srand(42);
    const double pi2 = 2 * 3.14159265;
    for(int i = 0; i < 20; i++)
    {
      double theta = pi2 * rand() / RAND_MAX;
      pts.emplace_back(cos(theta), sin(theta), 0);
    }
    AggGeom ag(pts, 2);
    std::string style = "Jacks";
    ag.build(style);
    //make sure the vertex at the origin was chosen as the only root in agg
    int numRoots = 0;
    size_t root;
    for(size_t i = 0; i < pts.size(); i++)
    {
      if(ag.isRoot_[i])
      {
        numRoots++;
        root = i;
      }
    }
    TEST_EQUALITY(numRoots, 1);
    TEST_EQUALITY(root, 0);
    //check geometry - should have exactly nonRoot line segments
    TEST_EQUALITY(ag.geomSizes_.size(), nonRoot);
    for(int i = 0; i < nonRoot; i++)
    {
      TEST_EQUALITY(ag.geomSizes_[i], 2);
    }
  }

  //Test 2D convex hull consisting of many points around a circle
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(VizGeom, ConvHull2D_1, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    using MueLu::VizHelpers::Vec3;
    typedef MueLu::VizHelpers::AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeom;
    //Make an agg out of n points around the unit circle
    const int n = 5;
    std::vector<Vec3> pts;
    srand(42);
    const double pi2 = 2 * 3.14159265;
    for(int i = 0; i < n; i++)
    {
      double theta = pi2 * rand() / RAND_MAX;
      pts.emplace_back(cos(theta), sin(theta), 0);
    }
    AggGeom ag(pts, 2);
    std::string style = "Convex Hulls";
    ag.build(style);
    //check geometry - should have exactly one geometry element (polygon) with exactly n points
    TEST_EQUALITY(ag.geomSizes_.size(), 1);
    TEST_EQUALITY(ag.geomSizes_[0], n);
    emitAggGeom(ag);
  }

  //Test 2D convex hull consisting of 4 points in a square and many points inside the square and on the boundary
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(VizGeom, ConvHull2D_2, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    using MueLu::VizHelpers::Vec3;
    typedef MueLu::VizHelpers::AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeom;
    //Make an agg out of the square [0, 1] x [0, 1], plus n points inside and on boundary
    const int n = 100;
    std::vector<Vec3> pts;
    srand(42);
    pts.emplace_back(0, 0, 0);
    pts.emplace_back(1, 0, 0);
    pts.emplace_back(1, 1, 0);
    pts.emplace_back(0, 1, 0);
    const int resolution = 100000;
    for(int i = 0; i < n; i++)
    {
      pts.emplace_back((rand() % (resolution + 1)) / (double) resolution, (rand() % (resolution + 1)) / (double) resolution, 0);
    }
    AggGeom ag(pts, 2);
    std::string style = "Convex Hulls";
    ag.build(style);
    //check geometry - should have exactly one geometry element (square) with exactly 4 points
    TEST_EQUALITY(ag.geomSizes_.size(), 1);
    TEST_EQUALITY(ag.geomSizes_[0], 4);
    emitAggGeom(ag);
  }

  //Minimal test of 3D convex hull geometry and VTK output
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(VizGeom, ConvHull3D, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    using MueLu::VizHelpers::Vec3;
    typedef MueLu::VizHelpers::AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeom;
    //Make an agg out of n random points in [0,1) x [0,1) x [0,1)
    const int n = 1000;
    std::vector<Vec3> pts;
    srand(42);
    for(int i = 0; i < n; i++)
    {
      pts.emplace_back(1.0 * rand() / RAND_MAX, 1.0 * rand() / RAND_MAX, 1.0 * rand() / RAND_MAX);
    }
    AggGeom ag(pts, 3);
    std::string style = "Convex Hulls";
    ag.build(style);
    //check geometry - should have exactly one geometry element (some polyhedron) with 4 <= faces <= n
    TEST_EQUALITY(ag.geomSizes_.size() >= 4, true);
    //every geometry element should be a triangle
    bool areAllTriangles = true;
    for(auto gs : ag.geomSizes_)
    {
      if(gs != 3)
      {
        areAllTriangles = false;
        break;
      }
    }
    TEST_EQUALITY(areAllTriangles, true);
    emitAggGeom(ag);
  }

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(VizGeom, ConvHull3D_Cube, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    using MueLu::VizHelpers::Vec3;
    using std::string;
    using Teuchos::ParameterList;
    typedef MueLu::VizHelpers::AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeom;
    //Make an agg out of the square [0, 1] x [0, 1], plus n points inside and on boundary
    const int n = 10000;
    std::vector<Vec3> pts;
    srand(42);
    pts.emplace_back(0, 0, 0);
    pts.emplace_back(1, 0, 0);
    pts.emplace_back(0, 1, 0);
    pts.emplace_back(1, 1, 0);
    pts.emplace_back(0, 0, 1);
    pts.emplace_back(1, 0, 1);
    pts.emplace_back(0, 1, 1);
    pts.emplace_back(1, 1, 1);
    const int resolution = 100000;
    for(int i = 0; i < n; i++)
    {
      pts.emplace_back(
          (rand() % (resolution + 1)) / (double) resolution,
          (rand() % (resolution + 1)) / (double) resolution,
          (rand() % (resolution + 1)) / (double) resolution);
    }
    AggGeom ag(pts, 3);
    std::string style = "Convex Hulls";
    ag.build(style);
    //check geometry - should have exactly 12 geometry elements (cube made of triangles) with exactly 3 points each
    TEST_EQUALITY(ag.geomSizes_.size(), 12);
    for(int i = 0; i < 12; i++)
    {
      TEST_EQUALITY(ag.geomSizes_[i], 3);
    }
    emitAggGeom(ag);
  }

  //Make an agg consisting of many points on the unit sphere
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(VizGeom, ConvHull3D_Large, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    using MueLu::VizHelpers::Vec3;
    using std::vector;
    typedef MueLu::VizHelpers::AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeom;
    //Make an agg out of the square [0, 1] x [0, 1], plus n points inside and on boundary
    const int n = 50000;
    vector<Vec3> pts;
    srand(42);
    for(int i = 0; i < n; i++)
    {
      //to get even distribution of points on sphere, just pick normal components uniformly and normalize
      Vec3 v(0, 0, 0);
      while(MueLu::VizHelpers::mag(v) == 0)
      {
        v.x = -1.0 + rand() * 2.0 / RAND_MAX;
        v.y = -1.0 + rand() * 2.0 / RAND_MAX;
        v.z = -1.0 + rand() * 2.0 / RAND_MAX;
        v = v.normalize();
      }
      pts.emplace_back(v);
    }
    AggGeom ag(pts, 3);
    std::string style = "Convex Hulls";
    ag.build(style);
    emitAggGeom(ag);
  }

  //Make an agg consisting of many points on a line in 3D (with random orientation and fixed length)
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(VizGeom, ConvHull3D_Collinear, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    using MueLu::VizHelpers::Vec3;
    typedef MueLu::VizHelpers::AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeom;
    //Make an agg out of the square [0, 1] x [0, 1], plus n points inside and on boundary
    const int n = 30;
    std::vector<Vec3> pts;
    srand(42);
    Vec3 direction((1.0 + rand()) / RAND_MAX, (1.0 + rand()) / RAND_MAX, (1.0 + rand()) / RAND_MAX);
    direction = direction.normalize();
    for(int i = 0; i < n; i++)
    {
      double mult = (20.0 * rand()) / RAND_MAX;
      //scale direction vector by random multiplier
      pts.emplace_back(direction * mult);
    }
    AggGeom ag(pts, 3);
    std::string style = "Convex Hulls";
    ag.build(style);
    //Should have exactly one line segment as the only geometry
    TEST_EQUALITY(ag.geomSizes_.size(), 1);
    TEST_EQUALITY(ag.geomSizes_[0], 2);
    //make sure the endpoints of segment are not the same
    TEST_EQUALITY(ag.geomVerts_[0].vert == ag.geomVerts_[1].vert, false);
    emitAggGeom(ag);
  }

  //Make an agg from many points on a plane with random 3D orientation
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(VizGeom, ConvHull3D_Coplanar, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    using MueLu::VizHelpers::Vec3;
    typedef MueLu::VizHelpers::AggGeometry<Scalar, LocalOrdinal, GlobalOrdinal, Node> AggGeom;
    //Make an agg out of the square [0, 1] x [0, 1], plus n points inside and on boundary
    const int n = 30;
    std::vector<Vec3> pts;
    srand(42);
    //get two tangent vectors to represent plane
    Vec3 tan1((1.0 + rand()) / RAND_MAX, (1.0 + rand()) / RAND_MAX, (1.0 + rand()) / RAND_MAX);
    tan1 = tan1.normalize();
    Vec3 tan2((1.0 + rand()) / RAND_MAX, (1.0 + rand()) / RAND_MAX, (1.0 + rand()) / RAND_MAX);
    tan2 = tan2.normalize();
    for(int i = 0; i < n; i++)
    {
      double mult1 = (20.0 * rand()) / RAND_MAX;
      double mult2 = (20.0 * rand()) / RAND_MAX;
      //place point as linear combination of the tangent vectors
      pts.emplace_back(tan1 * mult1 + tan2 * mult2);
    }
    AggGeom ag(pts, 3);
    std::string style = "Convex Hulls";
    ag.build(style);
    //Should have one polygon as geometry
    TEST_EQUALITY(ag.geomSizes_.size(), 1);
    emitAggGeom(ag);
  }

#define MUELU_ETI_GROUP(Scalar,LO,GO,Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(VizGeom, PointCloud, Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(VizGeom, Jacks, Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(VizGeom, ConvHull2D_1, Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(VizGeom, ConvHull2D_2, Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(VizGeom, ConvHull3D, Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(VizGeom, ConvHull3D_Cube, Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(VizGeom, ConvHull3D_Large, Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(VizGeom, ConvHull3D_Collinear, Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(VizGeom, ConvHull3D_Coplanar, Scalar, LO, GO, Node)

#include <MueLu_ETI_4arg.hpp>

} // namespace MueLuTests

