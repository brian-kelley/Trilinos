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

#ifndef _FROSCH_INTERFACEENTITY_DEF_HPP
#define _FROSCH_INTERFACEENTITY_DEF_HPP

#include <FROSch_InterfaceEntity_decl.hpp>


namespace FROSch {
    
    using namespace Teuchos;
    using namespace Xpetra;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO>
    bool Node<SC,LO,GO>::operator< (const Node &n) const
#else
    template <class SC,>
    bool Node<SC>::operator< (const Node &n) const
#endif
    {
        return NodeIDGlobal_<n.NodeIDGlobal_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO>
    bool Node<SC,LO,GO>::operator== (const Node &n) const
#else
    template <class SC,>
    bool Node<SC>::operator== (const Node &n) const
#endif
    {
        return NodeIDGlobal_==n.NodeIDGlobal_;
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    InterfaceEntity<SC,LO,GO,NO>::InterfaceEntity(EntityType type,
#else
    template <class SC,class NO>
    InterfaceEntity<SC,NO>::InterfaceEntity(EntityType type,
#endif
                                                  UN dofsPerNode,
                                                  UN multiplicity,
                                                  const int *subdomains,
                                                  EntityFlag flag) :
    Type_ (type),
    Flag_ (flag),
    SubdomainsVector_ (multiplicity),
    DofsPerNode_ (dofsPerNode),
    Multiplicity_ (multiplicity)
    {
        for (UN i=0; i<multiplicity; i++) {
            SubdomainsVector_[i] = subdomains[i];
        }
        sortunique(SubdomainsVector_);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Ancestors_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
        Offspring_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
        Roots_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
#else
        Ancestors_.reset(new EntitySet<SC,NO>(DefaultType));
        Offspring_.reset(new EntitySet<SC,NO>(DefaultType));
        Roots_.reset(new EntitySet<SC,NO>(DefaultType));
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    InterfaceEntity<SC,LO,GO,NO>::~InterfaceEntity()
#else
    template <class SC,class NO>
    InterfaceEntity<SC,NO>::~InterfaceEntity()
#endif
    {

    } // Do we need sth here?

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::addNode(LO nodeIDGamma,
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::addNode(LO nodeIDGamma,
#endif
                                              LO nodeIDLocal,
                                              GO nodeIDGlobal,
                                              UN nDofs,
                                              const LOVecPtr dofsGamma,
                                              const LOVecPtr dofsLocal,
                                              const GOVecPtr dofsGlobal)
    {
        FROSCH_ASSERT(nDofs<=DofsPerNode_,"nDofs>NumDofs_.");

        FROSCH_ASSERT(dofsGamma.size()==nDofs,"dofIDs.size()!=nDofs");
        FROSCH_ASSERT(dofsLocal.size()==nDofs,"dofIDs.size()!=nDofs");
        FROSCH_ASSERT(dofsGlobal.size()==nDofs,"dofIDs.size()!=nDofs");

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Node<SC,LO,GO> node;
#else
        Node<SC> node;
#endif

        node.NodeIDGamma_ = nodeIDGamma;
        node.NodeIDLocal_ = nodeIDLocal;
        node.NodeIDGlobal_ = nodeIDGlobal;

        node.DofsGamma_.deepCopy(dofsGamma());
        node.DofsLocal_.deepCopy(dofsLocal());
        node.DofsGlobal_.deepCopy(dofsGlobal());

        NodeVector_.push_back(node);
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::addNode(const NodePtr &node)
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::addNode(const NodePtr &node)
#endif
    {
        return addNode(*node);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::addNode(const Node<SC,LO,GO> &node)
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::addNode(const Node<SC> &node)
#endif
    {
        FROSCH_ASSERT(node.DofsGamma_.size()<=DofsPerNode_,"node.DofsGamma_ is too large.");
        FROSCH_ASSERT(node.DofsLocal_.size()<=DofsPerNode_,"node.DofsLocal_ is too large.");
        FROSCH_ASSERT(node.DofsGlobal_.size()<=DofsPerNode_,"node.DofsGlobal_ is too large.");

        NodeVector_.push_back(node);

        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::resetGlobalDofs(UN iD,
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::resetGlobalDofs(UN iD,
#endif
                                                      UN nDofs,
                                                      UN *dofIDs,
                                                      GO *dofsGlobal)
    {
        FROSCH_ASSERT(iD<getNumNodes(),"iD=>getNumNodes()");
        FROSCH_ASSERT(nDofs<=DofsPerNode_,"nDofs>DofsPerNode_.");

        for (unsigned i=0; i<nDofs; i++) {
            if ((0<=dofIDs[i])&&(dofIDs[i]<=DofsPerNode_)) {
                NodeVector_[iD].DofsGlobal_[dofIDs[i]] = dofsGlobal[dofIDs[i]];
            } else {
                FROSCH_ASSERT(false,"dofIDs[i] is out of range.");
            }
        }

        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::removeNode(UN iD)
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::removeNode(UN iD)
#endif
    {
        FROSCH_ASSERT(iD<getNumNodes(),"iD=>getNumNodes()");
        NodeVector_.erase(NodeVector_.begin()+iD);
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::sortByGlobalID()
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::sortByGlobalID()
#endif
    {
        sortunique(NodeVector_);
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::setUniqueID(GO uniqueID)
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::setUniqueID(GO uniqueID)
#endif
    {
        UniqueID_ = uniqueID;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::setLocalID(LO localID)
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::setLocalID(LO localID)
#endif
    {
        LocalID_ = localID;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::setRootID(LO rootID)
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::setRootID(LO rootID)
#endif
    {
        RootID_ = rootID;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::setLeafID(LO leafID)
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::setLeafID(LO leafID)
#endif
    {
        LeafID_ = leafID;
        return 0;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::setUniqueIDToFirstGlobalID()
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::setUniqueIDToFirstGlobalID()
#endif
    {
        UniqueID_ = NodeVector_[0].NodeIDGlobal_;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::resetEntityType(EntityType type)
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::resetEntityType(EntityType type)
#endif
    {
        Type_ = type;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::resetEntityFlag(EntityFlag flag)
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::resetEntityFlag(EntityFlag flag)
#endif
    {
        Flag_ = flag;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::findAncestorsInSet(EntitySetPtr entitySet)
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::findAncestorsInSet(EntitySetPtr entitySet)
#endif
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        EntitySetPtr ancestors(new EntitySet<SC,LO,GO,NO>(*entitySet));
#else
        EntitySetPtr ancestors(new EntitySet<SC,NO>(*entitySet));
#endif
        IntVec tmpVector;
        for (UN i=0; i<Multiplicity_; i++) {
            UN length = ancestors->getNumEntities();
            for (UN j=0; j<length; j++) {
                tmpVector = ancestors->getEntity(length-1-j)->getSubdomainsVector();
                if (ancestors->getEntity(length-1-j)->getMultiplicity()<=this->getMultiplicity() || !std::binary_search(tmpVector.begin(),tmpVector.end(),SubdomainsVector_[i])) {
                    ancestors->removeEntity(length-1-j);
                }
            }
        }

        for (UN i=0; i<ancestors->getNumEntities(); i++) {
            Ancestors_->addEntity(ancestors->getEntity(i));
        }
        Ancestors_->sortUnique();
        
        // this is offspring of each ancestor
        for (UN i=0; i<Ancestors_->getNumEntities(); i++) {
            InterfaceEntityPtr thisEntity = rcpFromRef(*this);
            FROSCH_ASSERT(!thisEntity.is_null(),"FROSch::InterfaceEntity : ERROR: thisEntity.is_null()");
            Ancestors_->getEntity(i)->addOffspring(thisEntity);
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::clearAncestors()
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::clearAncestors()
#endif
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Ancestors_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
#else
        Ancestors_.reset(new EntitySet<SC,NO>(DefaultType));
#endif
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::addOffspring(InterfaceEntityPtr interfaceEntity)
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::addOffspring(InterfaceEntityPtr interfaceEntity)
#endif
    {
        Offspring_->addEntity(interfaceEntity);
        Offspring_->sortUnique();
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::clearOffspring()
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::clearOffspring()
#endif
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Offspring_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
#else
        Offspring_.reset(new EntitySet<SC,NO>(DefaultType));
#endif
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename InterfaceEntity<SC,LO,GO,NO>::EntitySetPtr InterfaceEntity<SC,LO,GO,NO>::findRoots()
#else
    template <class SC,class NO>
    typename InterfaceEntity<SC,NO>::EntitySetPtr InterfaceEntity<SC,NO>::findRoots()
#endif
    {
        if (Roots_->getNumEntities()) {
            FROSCH_ASSERT(Ancestors_->getNumEntities()!=0,"Ancestors_->getNumEntities()==0");
            return Roots_;
        }
        for (UN i=0; i<Ancestors_->getNumEntities(); i++) {
            EntitySetPtr tmpRoots = Ancestors_->getEntity(i)->findRoots();
            if (tmpRoots.is_null()) {
                FROSCH_ASSERT(Ancestors_->getEntity(i)->getAncestors()->getNumEntities()==0,"EntityVector_[i]->getAncestors()->getNumEntities()!=0");
                Roots_->addEntity(Ancestors_->getEntity(i));
            } else {
                FROSCH_ASSERT(Ancestors_->getEntity(i)->getAncestors()->getNumEntities()!=0,"EntityVector_[i]->getAncestors()->getNumEntities()==0");
                FROSCH_ASSERT(tmpRoots->getNumEntities()>0,"tmpRoots->getNumEntities()<=0");
                Roots_->addEntitySet(tmpRoots);
            }
        }
        Roots_->sortUnique();
        if (Roots_->getNumEntities()) {
            FROSCH_ASSERT(Ancestors_->getNumEntities()!=0,"Ancestors_->getNumEntities()==0");
            return Roots_;
        } else {
            return null;
        }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::clearRoots()
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::clearRoots()
#endif
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Roots_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
#else
        Roots_.reset(new EntitySet<SC,NO>(DefaultType));
#endif
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::computeDistancesToRoots(UN dimension,
#else
    template <class SC,class NO>
    int InterfaceEntity<SC,NO>::computeDistancesToRoots(UN dimension,
#endif
                                                              ConstXMultiVectorPtr &nodeList,
                                                              DistanceFunction distanceFunction)
    {
        if (Roots_->getNumEntities()>0) {
            DistancesVector_.resize(getNumNodes());
            for (UN i=0; i<getNumNodes(); i++) {
                DistancesVector_[i].resize(Roots_->getNumEntities()+1,std::numeric_limits<SC>::max());
            }

            switch (distanceFunction) {
                case ConstantDistanceFunction:
                    for (UN i=0; i<NodeVector_.size(); i++) {
                        for (UN j=0; j<Roots_->getNumEntities(); j++) {
                            DistancesVector_[i][j] = ScalarTraits<SC>::one(); // AH 08/08/2019 TODO: Make a MultiVector out of this and use putScalar()
                        }
                    }
                    break;
                case InverseEuclideanDistanceFunction:
                    FROSCH_ASSERT(!nodeList.is_null(),"FROSch::InterfaceEntity : ERROR: The inverse euclidean distance cannot be calculated without coordinates of the nodes!");
                    FROSCH_ASSERT(dimension==nodeList->getNumVectors(),"FROSch::InterfaceEntity : ERROR: Inconsistent Dimension.");
                    for (UN i=0; i<Roots_->getNumEntities(); i++) {
                        for (UN j=0; j<Roots_->getEntity(i)->getNumNodes(); j++) {
                            // Coordinates of the nodes of the coarse node
                            SCVecPtr CN(dimension);
                            for (UN k=0; k<dimension; k++) {
                                CN[k] = nodeList->getData(k)[Roots_->getEntity(i)->getLocalNodeID(j)];
                            }
                            for (UN k=0; k<NodeVector_.size(); k++) {
                                // Coordinates of the nodes of the entity
                                SCVecPtr E(dimension);
                                SC distance = ScalarTraits<SC>::zero();
                                // Compute quadratic distance
                                for (UN l=0; l<dimension; l++) {
                                    distance += (nodeList->getData(l)[this->getLocalNodeID(k)]-CN[l]) * (nodeList->getData(l)[this->getLocalNodeID(k)]-CN[l]);
                                }
                                // Compute inverse euclidean distance
                                distance = std::sqrt(distance);
                                DistancesVector_[k][i] = std::min(DistancesVector_[k][i],distance);
                            }
                        }
                    }
                    for (UN i=0; i<NodeVector_.size(); i++) {
                        for (UN j=0; j<Roots_->getNumEntities(); j++) {
                            DistancesVector_[i][j] = ScalarTraits<SC>::one()/DistancesVector_[i][j];
                        }
                    }
                    break;
                default:
                    FROSCH_ASSERT(false,"FROSch::InterfaceEntity : ERROR: Specify a valid Distance Function.");
                    break;
            }

            // In the last "row", we store the sum of the distances for all coarse nodes
            for (UN i=0; i<NodeVector_.size(); i++) {
                DistancesVector_[i][Roots_->getNumEntities()] = ScalarTraits<SC>::zero();
                for (UN j=0; j<Roots_->getNumEntities(); j++) {
                    DistancesVector_[i][Roots_->getNumEntities()] += DistancesVector_[i][j];
                }
            }
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename InterfaceEntity<SC,LO,GO,NO>::InterfaceEntityPtr InterfaceEntity<SC,LO,GO,NO>::divideEntity(ConstXMatrixPtr matrix,
#else
    template <class SC,class NO>
    typename InterfaceEntity<SC,NO>::InterfaceEntityPtr InterfaceEntity<SC,NO>::divideEntity(ConstXMatrixPtr matrix,
#endif
                                                                                                         int pID)
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        InterfaceEntityPtr entity(new InterfaceEntity<SC,LO,GO,NO>(Type_,DofsPerNode_,Multiplicity_,&(SubdomainsVector_[0])));
#else
        InterfaceEntityPtr entity(new InterfaceEntity<SC,NO>(Type_,DofsPerNode_,Multiplicity_,&(SubdomainsVector_[0])));
#endif

        if (getNumNodes()>=2) {
            sortByGlobalID();

            GOVecPtr mapVector(getNumNodes());
            for (UN i=0; i<getNumNodes(); i++) {
                mapVector[i] = NodeVector_[i].DofsGamma_[0];
            }
            XMatrixPtr localMatrix,mat1,mat2,mat3;
            BuildSubmatrices(matrix,mapVector(),localMatrix,mat1,mat2,mat3);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            XVectorPtr iterationVector = VectorFactory<SC,LO,GO,NO>::Build(localMatrix->getRowMap());
#else
            XVectorPtr iterationVector = VectorFactory<SC,NO>::Build(localMatrix->getRowMap());
#endif
            iterationVector->getDataNonConst(0)[0] = ScalarTraits<SC>::one();
            for (UN i=0; i<getNumNodes()-1; i++) {
                localMatrix->apply(*iterationVector,*iterationVector);
            }

            for (UN i=0; i<getNumNodes(); i++) {
                if (fabs(iterationVector->getData(0)[i])<1.0e-10) {
                    entity->addNode(getNode(i));
                    removeNode(i);
                }
            }
        }
        return entity;
    }

    /////////////////
    // Get Methods //
    /////////////////

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    EntityType InterfaceEntity<SC,LO,GO,NO>::getEntityType() const
#else
    template <class SC,class NO>
    EntityType InterfaceEntity<SC,NO>::getEntityType() const
#endif
    {
        return Type_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    EntityFlag InterfaceEntity<SC,LO,GO,NO>::getEntityFlag() const
#else
    template <class SC,class NO>
    EntityFlag InterfaceEntity<SC,NO>::getEntityFlag() const
#endif
    {
        return Flag_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename InterfaceEntity<SC,LO,GO,NO>::UN InterfaceEntity<SC,LO,GO,NO>::getDofsPerNode() const
#else
    template <class SC,class NO>
    typename InterfaceEntity<SC,NO>::UN InterfaceEntity<SC,NO>::getDofsPerNode() const
#endif
    {
        return DofsPerNode_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename InterfaceEntity<SC,LO,GO,NO>::UN InterfaceEntity<SC,LO,GO,NO>::getMultiplicity() const
#else
    template <class SC,class NO>
    typename InterfaceEntity<SC,NO>::UN InterfaceEntity<SC,NO>::getMultiplicity() const
#endif
    {
        return Multiplicity_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    GO InterfaceEntity<SC,LO,GO,NO>::getUniqueID() const
#else
    template <class SC,class NO>
    GO InterfaceEntity<SC,NO>::getUniqueID() const
#endif
    {
        return UniqueID_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getLocalID() const
#else
    template <class SC,class NO>
    LO InterfaceEntity<SC,NO>::getLocalID() const
#endif
    {
        return LocalID_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getRootID() const
#else
    template <class SC,class NO>
    LO InterfaceEntity<SC,NO>::getRootID() const
#endif
    {
        return RootID_;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getLeafID() const
#else
    template <class SC,class NO>
    LO InterfaceEntity<SC,NO>::getLeafID() const
#endif
    {
        return LeafID_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    const Node<SC,LO,GO>& InterfaceEntity<SC,LO,GO,NO>::getNode(UN iDNode) const
#else
    template <class SC,class NO>
    const Node<SC>& InterfaceEntity<SC,NO>::getNode(UN iDNode) const
#endif
    {
        return NodeVector_[iDNode];
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getGammaNodeID(UN iDNode) const
#else
    template <class SC,class NO>
    LO InterfaceEntity<SC,NO>::getGammaNodeID(UN iDNode) const
#endif
    {
        return NodeVector_[iDNode].NodeIDGamma_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getLocalNodeID(UN iDNode) const
#else
    template <class SC,class NO>
    LO InterfaceEntity<SC,NO>::getLocalNodeID(UN iDNode) const
#endif
    {
        return NodeVector_[iDNode].NodeIDLocal_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    GO InterfaceEntity<SC,LO,GO,NO>::getGlobalNodeID(UN iDNode) const
#else
    template <class SC,class NO>
    GO InterfaceEntity<SC,NO>::getGlobalNodeID(UN iDNode) const
#endif
    {
        return NodeVector_[iDNode].NodeIDGlobal_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getGammaDofID(UN iDNode, UN iDDof) const
#else
    template <class SC,class NO>
    LO InterfaceEntity<SC,NO>::getGammaDofID(UN iDNode, UN iDDof) const
#endif
    {
        return NodeVector_[iDNode].DofsGamma_[iDDof];
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getLocalDofID(UN iDNode, UN iDDof) const
#else
    template <class SC,class NO>
    LO InterfaceEntity<SC,NO>::getLocalDofID(UN iDNode, UN iDDof) const
#endif
    {
        return NodeVector_[iDNode].DofsLocal_[iDDof];
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    GO InterfaceEntity<SC,LO,GO,NO>::getGlobalDofID(UN iDNode, UN iDDof) const
#else
    template <class SC,class NO>
    GO InterfaceEntity<SC,NO>::getGlobalDofID(UN iDNode, UN iDDof) const
#endif
    {
        return NodeVector_[iDNode].DofsGlobal_[iDDof];
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    const typename InterfaceEntity<SC,LO,GO,NO>::IntVec & InterfaceEntity<SC,LO,GO,NO>::getSubdomainsVector() const
#else
    template <class SC,class NO>
    const typename InterfaceEntity<SC,NO>::IntVec & InterfaceEntity<SC,NO>::getSubdomainsVector() const
#endif
    {
        return SubdomainsVector_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename InterfaceEntity<SC,LO,GO,NO>::UN InterfaceEntity<SC,LO,GO,NO>::getNumNodes() const
#else
    template <class SC,class NO>
    typename InterfaceEntity<SC,NO>::UN InterfaceEntity<SC,NO>::getNumNodes() const
#endif
    {
        return NodeVector_.size();
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    const typename InterfaceEntity<SC,LO,GO,NO>::EntitySetPtr InterfaceEntity<SC,LO,GO,NO>::getAncestors() const
#else
    template <class SC,class NO>
    const typename InterfaceEntity<SC,NO>::EntitySetPtr InterfaceEntity<SC,NO>::getAncestors() const
#endif
    {
        return Ancestors_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    const typename InterfaceEntity<SC,LO,GO,NO>::EntitySetPtr InterfaceEntity<SC,LO,GO,NO>::getOffspring() const
#else
    template <class SC,class NO>
    const typename InterfaceEntity<SC,NO>::EntitySetPtr InterfaceEntity<SC,NO>::getOffspring() const
#endif
    {
        return Offspring_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    const typename InterfaceEntity<SC,LO,GO,NO>::EntitySetPtr InterfaceEntity<SC,LO,GO,NO>::getRoots() const
#else
    template <class SC,class NO>
    const typename InterfaceEntity<SC,NO>::EntitySetPtr InterfaceEntity<SC,NO>::getRoots() const
#endif
    {
        return Roots_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    SC InterfaceEntity<SC,LO,GO,NO>::getDistanceToRoot(UN iDNode,
#else
    template <class SC,class NO>
    SC InterfaceEntity<SC,NO>::getDistanceToRoot(UN iDNode,
#endif
                                                       UN iDRoot) const
    {
        FROSCH_ASSERT(iDNode<getNumNodes(),"iDNode>=getNumNodes()");
        FROSCH_ASSERT(iDRoot<Roots_->getNumEntities()+1,"iDNode>=Roots_->getNumEntities()+1");
        return DistancesVector_[iDNode][iDRoot];
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    bool compareInterfaceEntities(RCP<InterfaceEntity<SC,LO,GO,NO> > iEa,
                                  RCP<InterfaceEntity<SC,LO,GO,NO> > iEb)
#else
    template <class SC,class NO>
    bool compareInterfaceEntities(RCP<InterfaceEntity<SC,NO> > iEa,
                                  RCP<InterfaceEntity<SC,NO> > iEb)
#endif
    {
        return iEa->getUniqueID()<iEb->getUniqueID();
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    bool equalInterfaceEntities(RCP<InterfaceEntity<SC,LO,GO,NO> > iEa,
                                RCP<InterfaceEntity<SC,LO,GO,NO> > iEb)
#else
    template <class SC,class NO>
    bool equalInterfaceEntities(RCP<InterfaceEntity<SC,NO> > iEa,
                                RCP<InterfaceEntity<SC,NO> > iEb)
#endif
    {
        return iEa->getUniqueID()==iEb->getUniqueID();
    }
}

#endif
