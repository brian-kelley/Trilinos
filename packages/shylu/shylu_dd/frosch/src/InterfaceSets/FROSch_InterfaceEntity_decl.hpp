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

#ifndef _FROSCH_INTERFACEENTITY_DECL_HPP
#define _FROSCH_INTERFACEENTITY_DECL_HPP

#include <Xpetra_VectorFactory_fwd.hpp>

#include <FROSch_ExtractSubmatrices_def.hpp>
#include <FROSch_Tools_def.hpp>


namespace FROSch {
    
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC = double,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              class LO = int,
              class GO = DefaultGlobalOrdinal,
#endif
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
    class EntitySet;

    enum EntityType {DefaultType,VertexType,EdgeType,FaceType,InteriorType,InterfaceType};
    enum EntityFlag {DefaultFlag,StraightFlag,ShortFlag,NodeFlag};
    enum DistanceFunction {ConstantDistanceFunction,InverseEuclideanDistanceFunction};

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal>
#else
    template <class SC = double,>
#endif
    struct Node {
#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using LO = typename Tpetra::Map<>::local_ordinal_type;
        using GO = typename Tpetra::Map<>::global_ordinal_type;
#endif
        LO NodeIDGamma_;
        LO NodeIDLocal_;
        GO NodeIDGlobal_;

        ArrayRCP<LO> DofsGamma_;
        ArrayRCP<LO> DofsLocal_;
        ArrayRCP<GO> DofsGlobal_;

        bool operator< (const Node &n) const;

        bool operator== (const Node &n) const;
    };

    template <class SC = double,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              class LO = int,
              class GO = DefaultGlobalOrdinal,
#endif
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
    class InterfaceEntity {

    protected:

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMatrix               = Matrix<SC,LO,GO,NO>;
#else
        using LO = typename Tpetra::Map<>::local_ordinal_type;
        using GO = typename Tpetra::Map<>::global_ordinal_type;
        using XMatrix               = Matrix<SC,NO>;
#endif
        using XMatrixPtr            = RCP<XMatrix>;
        using ConstXMatrixPtr       = RCP<const XMatrix>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XVector               = Vector<SC,LO,GO,NO>;
#else
        using XVector               = Vector<SC,NO>;
#endif
        using XVectorPtr            = RCP<XVector>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMultiVector          = MultiVector<SC,LO,GO,NO>;
#else
        using XMultiVector          = MultiVector<SC,NO>;
#endif
        using XMultiVectorPtr       = RCP<XMultiVector>;
        using ConstXMultiVectorPtr  = RCP<const XMultiVector>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using EntitySetPtr          = RCP<EntitySet<SC,LO,GO,NO> >;
#else
        using EntitySetPtr          = RCP<EntitySet<SC,NO> >;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using InterfaceEntityPtr    = RCP<InterfaceEntity<SC,LO,GO,NO> >;
#else
        using InterfaceEntityPtr    = RCP<InterfaceEntity<SC,NO> >;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using NodeVec               = Array<Node<SC,LO,GO> >;
        using NodePtr               = RCP<Node<SC,LO,GO> >;
#else
        using NodeVec               = Array<Node<SC> >;
        using NodePtr               = RCP<Node<SC> >;
#endif
        using NodePtrVec            = Array<NodePtr>;

        using UN                    = unsigned;

        using IntVec                = Array<int>;

        using LOVecPtr              = ArrayRCP<LO>;

        using GOVec                 = Array<GO>;
        using GOVecPtr              = ArrayRCP<GO>;

        using SCVecPtr              = ArrayRCP<SC>;
        using SCVecPtrVec           = Array<SCVecPtr>;

    public:

        InterfaceEntity(EntityType type,
                        UN dofsPerNode,
                        UN multiplicity,
                        const int *subdomains,
                        EntityFlag flag = DefaultFlag);

        ~InterfaceEntity();

        int addNode(LO nodeIDGamma,
                    LO nodeIDLocal,
                    GO nodeIDGlobal,
                    UN nDofs,
                    const LOVecPtr dofsGamma,
                    const LOVecPtr dofsLocal,
                    const GOVecPtr dofsGlobal);

        int addNode(const NodePtr &node);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        int addNode(const Node<SC,LO,GO> &node);
#else
        int addNode(const Node<SC> &node);
#endif

        int resetGlobalDofs(UN iD,
                            UN nDofs,
                            UN *dofIDs,
                            GO *dofsGlobal);

        int removeNode(UN iD);

        int sortByGlobalID();

        int setUniqueID(GO uniqueID);

        int setLocalID(LO localID);

        int setRootID(LO rootID);
        
        int setLeafID(LO leafID);

        int setUniqueIDToFirstGlobalID();

        int resetEntityType(EntityType type);

        int resetEntityFlag(EntityFlag flag);

        int findAncestorsInSet(EntitySetPtr entitySet);

        int clearAncestors();

        int addOffspring(InterfaceEntityPtr interfaceEntity);

        int clearOffspring();

        EntitySetPtr findRoots();

        int clearRoots();

        int computeDistancesToRoots(UN dimension,
                                    ConstXMultiVectorPtr &nodeList = null,
                                    DistanceFunction distanceFunction = ConstantDistanceFunction);

        InterfaceEntityPtr divideEntity(ConstXMatrixPtr matrix,
                                        int pID);

        /////////////////
        // Get Methods //
        /////////////////

        EntityType getEntityType() const;

        EntityFlag getEntityFlag() const;

        UN getDofsPerNode() const;

        UN getMultiplicity() const;

        GO getUniqueID() const;

        LO getLocalID() const;

        LO getRootID() const;
        
        LO getLeafID() const;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        const Node<SC,LO,GO>& getNode(UN iDNode) const;
#else
        const Node<SC>& getNode(UN iDNode) const;
#endif

        LO getGammaNodeID(UN iDNode) const;

        LO getLocalNodeID(UN iDNode) const;

        GO getGlobalNodeID(UN iDNode) const;

        LO getGammaDofID(UN iDNode, UN iDDof) const;

        LO getLocalDofID(UN iDNode, UN iDDof) const;

        GO getGlobalDofID(UN iDNode, UN iDDof) const;

        const IntVec & getSubdomainsVector() const;

        UN getNumNodes() const;

        const EntitySetPtr getAncestors() const;

        const EntitySetPtr getOffspring() const;

        const EntitySetPtr getRoots() const;

        SC getDistanceToRoot(UN iDNode,
                             UN iDRoot) const;

    protected:

        EntityType Type_ = DefaultType;

        EntityFlag Flag_ = DefaultFlag;

        NodeVec NodeVector_ = NodeVec(0);

        IntVec SubdomainsVector_ = IntVec(0);

        EntitySetPtr Ancestors_;
        EntitySetPtr Offspring_;
        EntitySetPtr Roots_;

        SCVecPtrVec DistancesVector_ = SCVecPtrVec(0); // AH 08/08/2019 TODO: make a MultiVector out of this

        UN DofsPerNode_ = 1;
        UN Multiplicity_ = 1;
        GO UniqueID_ = -1;
        LO LocalID_ = -1;
        LO RootID_ = -1;
        LO LeafID_ = -1;
    };

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    bool compareInterfaceEntities(RCP<InterfaceEntity<SC,LO,GO,NO> > iEa,
                                  RCP<InterfaceEntity<SC,LO,GO,NO> > iEb);

    template <class SC,class LO,class GO,class NO>
    bool equalInterfaceEntities(RCP<InterfaceEntity<SC,LO,GO,NO> > iEa,
                                RCP<InterfaceEntity<SC,LO,GO,NO> > iEb);
#else
    template <class SC,class NO>
    bool compareInterfaceEntities(RCP<InterfaceEntity<SC,NO> > iEa,
                                  RCP<InterfaceEntity<SC,NO> > iEb);

    template <class SC,class NO>
    bool equalInterfaceEntities(RCP<InterfaceEntity<SC,NO> > iEa,
                                RCP<InterfaceEntity<SC,NO> > iEb);
#endif

}

#endif
