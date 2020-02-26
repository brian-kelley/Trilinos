//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
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
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_DDINTERFACE_DECL_HPP
#define _FROSCH_DDINTERFACE_DECL_HPP

//#define INTERFACE_OUTPUT

#include <Xpetra_Operator_fwd.hpp>
#include <Xpetra_MapFactory_fwd.hpp>
#include <Xpetra_ExportFactory_fwd.hpp>
#include <Xpetra_CrsGraphFactory.hpp>

#include <FROSch_EntitySet_def.hpp>
#include <FROSch_InterfaceEntity_decl.hpp>

#include <FROSch_ExtractSubmatrices_def.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    enum CommunicationStrategy {CommCrsMatrix,CommCrsGraph,CreateOneToOneMap};

    template <class SC = double,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              class LO = int,
              class GO = DefaultGlobalOrdinal,
#endif
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
    class DDInterface {

    protected:

#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using LO = typename Tpetra::Map<>::local_ordinal_type;
        using GO = typename Tpetra::Map<>::global_ordinal_type;
#endif
        using CommPtr                   = RCP<const Comm<int> >;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMap                      = Map<LO,GO,NO>;
#else
        using XMap                      = Map<NO>;
#endif
        using XMapPtr                   = RCP<XMap>;
        using ConstXMapPtr              = RCP<const XMap>;
        using XMapPtrVecPtr             = ArrayRCP<XMapPtr>;
        using ConstXMapPtrVecPtr        = ArrayRCP<ConstXMapPtr>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMatrix                   = Matrix<SC,LO,GO,NO>;
#else
        using XMatrix                   = Matrix<SC,NO>;
#endif
        using XMatrixPtr                = RCP<XMatrix>;
        using ConstXMatrixPtr           = RCP<const XMatrix>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XCrsGraph                 = CrsGraph<LO,GO,NO>;
#else
        using XCrsGraph                 = CrsGraph<NO>;
#endif
        using XCrsGraphPtr              = RCP<XCrsGraph>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMultiVector              = MultiVector<SC,LO,GO,NO>;
#else
        using XMultiVector              = MultiVector<SC,NO>;
#endif
        using XMultiVectorPtr           = RCP<XMultiVector>;
        using ConstXMultiVectorPtr      = RCP<const XMultiVector>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XImport                   = Import<LO,GO,NO>;
#else
        using XImport                   = Import<NO>;
#endif
        using XImportPtr                = RCP<XImport>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XExport                   = Export<LO,GO,NO>;
#else
        using XExport                   = Export<NO>;
#endif
        using XExportPtr                = RCP<XExport>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using EntitySetPtr              = RCP<EntitySet<SC,LO,GO,NO> >;
#else
        using EntitySetPtr              = RCP<EntitySet<SC,NO> >;
#endif
        using EntitySetConstPtr         = const EntitySetPtr;
        using EntitySetPtrVecPtr        = ArrayRCP<EntitySetPtr>;
        using EntitySetPtrConstVecPtr   = const EntitySetPtrVecPtr;

        using EntityFlagVecPtr          = ArrayRCP<EntityFlag>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using InterfaceEntityPtr        = RCP<InterfaceEntity<SC,LO,GO,NO> >;
#else
        using InterfaceEntityPtr        = RCP<InterfaceEntity<SC,NO> >;
#endif
        using InterfaceEntityPtrVecPtr  = ArrayRCP<InterfaceEntityPtr>;

        using UN                        = unsigned;
        using ConstUN                   = const UN;
        using UNVecPtr                  = ArrayRCP<UN>;

        using IntVec                    = Array<int>;
        using IntVecVec                 = Array<IntVec>;
        using IntVecVecPtr              = ArrayRCP<IntVec>;

        using LOVec                     = Array<LO>;
        using LOVecPtr                  = ArrayRCP<LO>;

        using GOVec                     = Array<GO>;
        using ConstGOVecView            = ArrayView<const GO>;
        using GOVecPtr                  = ArrayRCP<GO>;
        using GOVecView                 = ArrayView<GO>;
        using GOVecVec                  = Array<GOVec>;
        using GOVecVecPtr               = ArrayRCP<GOVec>;

        using SCVec                     = Array<SC>;
        using SCVecPtr                  = ArrayRCP<SC>;

    public:

        DDInterface(UN dimension,
                    UN dofsPerNode,
                    ConstXMapPtr localToGlobalMap,
                    Verbosity verbosity = All,
                    UN levelID = 1,
                    CommunicationStrategy commStrategy = CommCrsGraph);

        ~DDInterface();

        int resetGlobalDofs(ConstXMapPtrVecPtr dofsMaps);

        int removeDirichletNodes(GOVecView dirichletBoundaryDofs);

        int divideUnconnectedEntities(ConstXMatrixPtr matrix);

        int flagEntities(ConstXMultiVectorPtr nodeList = null);

        int removeEmptyEntities();

        int sortVerticesEdgesFaces(ConstXMultiVectorPtr nodeList = null);

        int buildEntityMaps(bool buildVerticesMap = true,
                            bool buildShortEdgesMap = true,
                            bool buildStraightEdgesMap = true,
                            bool buildEdgesMap = true,
                            bool buildFacesMap = true,
                            bool buildRootsMap = false,
                            bool buildLeafsMap = false);

        int buildEntityHierarchy();

        int computeDistancesToRoots(UN dimension,
                                    ConstXMultiVectorPtr &nodeList = null,
                                    DistanceFunction distanceFunction = ConstantDistanceFunction);

        //! This function extracts those entities which are to be used to build a connectivity graph on the subdomain
        //! level. By default, we identify all entities with multiplicity 2. Afterwards, the corresponding entities can
        //! be obtained using the function getConnectivityEntities().
        //! If short or straight edges should be omitted, the function flagEntities() has to be called in advance.
        int identifyConnectivityEntities(UNVecPtr multiplicities = null,
                                         EntityFlagVecPtr flags = null);

        UN getDimension() const;

        UN getDofsPerNode() const;

        LO getNumMyNodes() const;

        //
        // Remove the references below?
        //

        EntitySetConstPtr & getVertices() const;

        EntitySetConstPtr & getShortEdges() const;

        EntitySetConstPtr & getStraightEdges() const;

        EntitySetConstPtr & getEdges() const;

        EntitySetConstPtr & getFaces() const;

        EntitySetConstPtr & getInterface() const;

        EntitySetConstPtr & getInterior() const;

        EntitySetConstPtr & getRoots() const;
        
        EntitySetConstPtr & getLeafs() const;

        EntitySetPtrConstVecPtr & getEntitySetVector() const;

        //! This function returns those entities which are to be used to build a connectivity graph on the subdomain
        //! level. They have to identified first using the function identifyConnectivityEntities().
        EntitySetConstPtr & getConnectivityEntities() const;

        ConstXMapPtr getNodesMap() const;


    protected:

        int communicateLocalComponents(IntVecVecPtr &componentsSubdomains,
                                       IntVecVec &componentsSubdomainsUnique,
                                       CommunicationStrategy commStrategy = CommCrsGraph);

        int identifyLocalComponents(IntVecVecPtr &componentsSubdomains,
                                    IntVecVec &componentsSubdomainsUnique);


        CommPtr MpiComm_;

        UN Dimension_ = 3;
        UN DofsPerNode_ = 1;
        LO NumMyNodes_ = 0;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        EntitySetPtr Vertices_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(VertexType));
        EntitySetPtr ShortEdges_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(EdgeType));
        EntitySetPtr StraightEdges_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(EdgeType));
        EntitySetPtr Edges_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(EdgeType));
        EntitySetPtr Faces_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(FaceType));
        EntitySetPtr Interface_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(InterfaceType));
        EntitySetPtr Interior_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(InteriorType));
        EntitySetPtr Roots_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(DefaultType));
        EntitySetPtr Leafs_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(DefaultType));
        EntitySetPtr ConnectivityEntities_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(DefaultType));
#else
        EntitySetPtr Vertices_ = EntitySetPtr(new EntitySet<SC,NO>(VertexType));
        EntitySetPtr ShortEdges_ = EntitySetPtr(new EntitySet<SC,NO>(EdgeType));
        EntitySetPtr StraightEdges_ = EntitySetPtr(new EntitySet<SC,NO>(EdgeType));
        EntitySetPtr Edges_ = EntitySetPtr(new EntitySet<SC,NO>(EdgeType));
        EntitySetPtr Faces_ = EntitySetPtr(new EntitySet<SC,NO>(FaceType));
        EntitySetPtr Interface_ = EntitySetPtr(new EntitySet<SC,NO>(InterfaceType));
        EntitySetPtr Interior_ = EntitySetPtr(new EntitySet<SC,NO>(InteriorType));
        EntitySetPtr Roots_ = EntitySetPtr(new EntitySet<SC,NO>(DefaultType));
        EntitySetPtr Leafs_ = EntitySetPtr(new EntitySet<SC,NO>(DefaultType));
        EntitySetPtr ConnectivityEntities_ = EntitySetPtr(new EntitySet<SC,NO>(DefaultType));
#endif
        EntitySetPtrVecPtr EntitySetVector_;

        ConstXMapPtr NodesMap_;
        ConstXMapPtr UniqueNodesMap_;

        bool Verbose_ = false;

        Verbosity Verbosity_ = All;

        ConstUN LevelID_ = 1;
    };

}

#endif
