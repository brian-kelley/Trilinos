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

#ifndef _FROSCH_ENTITYSET_DEF_HPP
#define _FROSCH_ENTITYSET_DEF_HPP

#include <FROSch_EntitySet_decl.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    EntitySet<SC,LO,GO,NO>::EntitySet(EntityType type) :
#else
    template<class SC,class NO>
    EntitySet<SC,NO>::EntitySet(EntityType type) :
#endif
    Type_ (type)
    {

    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    EntitySet<SC,LO,GO,NO>::EntitySet(const EntitySet<SC,LO,GO,NO> &entitySet) :
#else
    template<class SC,class NO>
    EntitySet<SC,NO>::EntitySet(const EntitySet<SC,NO> &entitySet) :
#endif
    Type_ (entitySet.getEntityType()),
    EntityVector_ (entitySet.getEntityVector())
    {

    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    EntitySet<SC,LO,GO,NO>::~EntitySet()
#else
    template<class SC,class NO>
    EntitySet<SC,NO>::~EntitySet()
#endif
    {

    } // Do we need sth here?

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::addEntity(InterfaceEntityPtr entity)
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::addEntity(InterfaceEntityPtr entity)
#endif
    {
        FROSCH_ASSERT(Type_==DefaultType||entity->getEntityType()==Type_,"FROSch::EntitySet : ERROR: Entity to add is of wrong type.");
        EntityVector_.push_back(entity);
        EntityMapIsUpToDate_ = false;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::addEntitySet(EntitySetPtr entitySet)
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::addEntitySet(EntitySetPtr entitySet)
#endif
    {
        for (UN i=0; i<entitySet->getNumEntities(); i++) {
            addEntity(entitySet->getEntity(i));
        }
        return 0;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    typename EntitySet<SC,LO,GO,NO>::EntitySetPtr EntitySet<SC,LO,GO,NO>::deepCopy()
#else
    template<class SC,class NO>
    typename EntitySet<SC,NO>::EntitySetPtr EntitySet<SC,NO>::deepCopy()
#endif
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        EntitySetPtr copy(new EntitySet<SC,LO,GO,NO>(Type_));
#else
        EntitySetPtr copy(new EntitySet<SC,NO>(Type_));
#endif
        for (UN i=0; i<getNumEntities(); i++) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            InterfaceEntityPtr entity(new InterfaceEntity<SC,LO,GO,NO>(getEntity(i)->getEntityType(),
#else
            InterfaceEntityPtr entity(new InterfaceEntity<SC,NO>(getEntity(i)->getEntityType(),
#endif
                                                                       getEntity(i)->getDofsPerNode(),
                                                                       getEntity(i)->getMultiplicity(),
                                                                       getEntity(i)->getSubdomainsVector().getRawPtr()));
            for (UN j=0; j<getEntity(i)->getNumNodes(); j++) {
                entity->addNode(getEntity(i)->getNode(j).NodeIDGamma_,
                                getEntity(i)->getNode(j).NodeIDLocal_,
                                getEntity(i)->getNode(j).NodeIDGlobal_,
                                getEntity(i)->getNode(j).DofsGamma_.size(),
                                getEntity(i)->getNode(j).DofsGamma_,
                                getEntity(i)->getNode(j).DofsLocal_,
                                getEntity(i)->getNode(j).DofsGlobal_);
            }
            copy->addEntity(entity);
        }
        return copy;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::buildEntityMap(ConstXMapPtr localToGlobalNodesMap)
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::buildEntityMap(ConstXMapPtr localToGlobalNodesMap)
#endif
    {
        if (!EntityMapIsUpToDate_) {
            LO localNumberEntities = getNumEntities();
            LO globalNumberEntities = 0; // AH 10/13/2017: Can we stick with LO here
            LO maxLocalNumberEntities = 0;
            reduceAll(*localToGlobalNodesMap->getComm(),REDUCE_SUM,localNumberEntities,ptr(&globalNumberEntities));
            reduceAll(*localToGlobalNodesMap->getComm(),REDUCE_MAX,localNumberEntities,ptr(&maxLocalNumberEntities));

            GOVec localToGlobalVector(0);
            if (globalNumberEntities>0) {
                // Set the Unique iD
                setUniqueIDToFirstGlobalNodeID();

                GOVec entities(maxLocalNumberEntities);
                for (UN i=0; i<getNumEntities(); i++) {
                    entities[i] = getEntity(i)->getUniqueID()+1;
                    getEntity(i)->setLocalID(i);
                }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                XMapPtr entityMapping = MapFactory<LO,GO,NO>::Build(localToGlobalNodesMap->lib(),-1,entities(),0,localToGlobalNodesMap->getComm());
#else
                XMapPtr entityMapping = MapFactory<NO>::Build(localToGlobalNodesMap->lib(),-1,entities(),0,localToGlobalNodesMap->getComm());
#endif

                GOVec allEntities(maxLocalNumberEntities*localToGlobalNodesMap->getComm()->getSize(),0);
                //localToGlobalNodesMap->getComm().GatherAll(&(entities->at(0)),&(allEntities->at(0)),maxLocalNumberEntities);
                gatherAll(*localToGlobalNodesMap->getComm(),maxLocalNumberEntities,entities.getRawPtr(),maxLocalNumberEntities*localToGlobalNodesMap->getComm()->getSize(),allEntities.getRawPtr());

                allEntities.push_back(0); // Um sicherzugehen, dass der erste Eintrag nach sort_unique eine 0 ist.

                sortunique(allEntities);

                localToGlobalVector.resize(localNumberEntities);
                int LocalID;
                for (UN i=1; i<allEntities.size(); i++) { // Wir fangen bei 1 an, weil wir am Anfang 1 auf die ID addiert haben
                    LocalID = entityMapping->getLocalElement(allEntities[i]);
                    if ( LocalID != -1) {
                        localToGlobalVector[LocalID] = i-1;
                    }
                }

            }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            EntityMap_ = MapFactory<LO,GO,NO>::Build(localToGlobalNodesMap->lib(),-1,localToGlobalVector(),0,localToGlobalNodesMap->getComm());
#else
            EntityMap_ = MapFactory<NO>::Build(localToGlobalNodesMap->lib(),-1,localToGlobalVector(),0,localToGlobalNodesMap->getComm());
#endif
            EntityMapIsUpToDate_ = true;
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::findAncestorsInSet(EntitySetPtr entitySet)
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::findAncestorsInSet(EntitySetPtr entitySet)
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            getEntity(i)->findAncestorsInSet(entitySet);
        }
        return 0;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    typename EntitySet<SC,LO,GO,NO>::EntitySetPtr EntitySet<SC,LO,GO,NO>::findRoots()
#else
    template<class SC,class NO>
    typename EntitySet<SC,NO>::EntitySetPtr EntitySet<SC,NO>::findRoots()
#endif
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        EntitySetPtr Roots(new EntitySet<SC,LO,GO,NO>(DefaultType));
#else
        EntitySetPtr Roots(new EntitySet<SC,NO>(DefaultType));
#endif
        for (UN i=0; i<getNumEntities(); i++) {
            EntitySetPtr tmpRoots = getEntity(i)->findRoots();
            if (tmpRoots.is_null()) {
                FROSCH_ASSERT(getEntity(i)->getAncestors()->getNumEntities()==0,"FROSch::EntitySet : ERROR: getEntity(i)->getAncestors()->getNumEntities()!=0");
                Roots->addEntity(getEntity(i));
            } else {
                FROSCH_ASSERT(getEntity(i)->getAncestors()->getNumEntities()!=0,"FROSch::EntitySet : ERROR: getEntity(i)->getAncestors()->getNumEntities()==0");
                FROSCH_ASSERT(tmpRoots->getNumEntities()>0,"FROSch::EntitySet : ERROR: tmpRoots->getNumEntities()<=0");
                Roots->addEntitySet(tmpRoots);
            }
        }
        Roots->sortUnique();
        return Roots;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    typename EntitySet<SC,LO,GO,NO>::EntitySetPtr EntitySet<SC,LO,GO,NO>::findLeafs()
#else
    template<class SC,class NO>
    typename EntitySet<SC,NO>::EntitySetPtr EntitySet<SC,NO>::findLeafs()
#endif
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        EntitySetPtr Leafs(new EntitySet<SC,LO,GO,NO>(DefaultType));
#else
        EntitySetPtr Leafs(new EntitySet<SC,NO>(DefaultType));
#endif
        for (UN i=0; i<getNumEntities(); i++) {
            if (getEntity(i)->getOffspring()->getNumEntities()==0) {
                Leafs->addEntity(getEntity(i));
            }
        }
        Leafs->sortUnique();
        return Leafs;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::clearAncestors()
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::clearAncestors()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            getEntity(i)->clearAncestors();
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::clearOffspring()
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::clearOffspring()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            getEntity(i)->clearOffspring();
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::clearRoots()
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::clearRoots()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            getEntity(i)->clearRoots();
        }
        return 0;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::clearLeafs()
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::clearLeafs()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            getEntity(i)->clearLeafs();
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::computeDistancesToRoots(UN dimension,
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::computeDistancesToRoots(UN dimension,
#endif
                                                        ConstXMultiVectorPtr &nodeList,
                                                        DistanceFunction distanceFunction)
    {
        for (UN i=0; i<getNumEntities(); i++) {
            getEntity(i)->computeDistancesToRoots(dimension,nodeList,distanceFunction);
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::divideUnconnectedEntities(ConstXMatrixPtr matrix,
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::divideUnconnectedEntities(ConstXMatrixPtr matrix,
#endif
                                                          int pID)
    {
        UN before = getNumEntities();
        UN i=0;
        while (i<getNumEntities()) {
            InterfaceEntityPtr tmpEntity = getEntity(i)->divideEntity(matrix,pID);
            if (tmpEntity->getNumNodes()>0) {
                addEntity(tmpEntity);
            }
            i++;
        }
        if (getNumEntities()-before>0) EntityMapIsUpToDate_ = false;
        return getNumEntities()-before;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::flagNodes()
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::flagNodes()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            if (getEntity(i)->getNumNodes()==1) {
                getEntity(i)->resetEntityFlag(NodeFlag);
            }
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::flagShortEntities()
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::flagShortEntities()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            if (getEntity(i)->getNumNodes()==2) {
                getEntity(i)->resetEntityFlag(ShortFlag);
            }
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::flagStraightEntities(UN dimension,
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::flagStraightEntities(UN dimension,
#endif
                                                     ConstXMultiVectorPtr &nodeList)
    {
        FROSCH_ASSERT(dimension==nodeList->getNumVectors(),"FROSch::EntitySet : ERROR: Inconsistent Dimension.");

        bool straight;
        LO length,j;
        SCVec pt1(dimension);
        SCVec dir1(dimension);
        SCVec dir2(dimension);

        for (UN i=0; i<getNumEntities(); i++) {
            straight = true;
            length = getEntity(i)->getNumNodes();

            j=2;

            if (length>2) {
                // Anfangssteigung berechnen
                for (UN k=0; k<dimension; k++) {
                    pt1[k] = nodeList->getData(k)[getEntity(i)->getLocalNodeID(0)];
                }

                for (UN k=0; k<dimension; k++) {
                    dir1[k] = nodeList->getData(k)[getEntity(i)->getLocalNodeID(1)]-pt1[k];
                }

                while (j<length) {
                    // Steigung zum zweiten Punkt berechnen
                    for (UN k=0; k<dimension; k++) {
                        dir2[k] = nodeList->getData(k)[getEntity(i)->getLocalNodeID(j)]-pt1[k];
                    }

                    if (!ismultiple<SC,LO>(dir1(),dir2())) {
                        straight = false;
                        break;
                    }
                    j++;
                }
                if (straight) {
                    getEntity(i)->resetEntityFlag(StraightFlag);
                }
            }
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    typename EntitySet<SC,LO,GO,NO>::EntitySetPtr EntitySet<SC,LO,GO,NO>::sortOutEntities(EntityFlag flag)
#else
    template<class SC,class NO>
    typename EntitySet<SC,NO>::EntitySetPtr EntitySet<SC,NO>::sortOutEntities(EntityFlag flag)
#endif
    {
        UN before = getNumEntities();
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        EntitySetPtr removedEntities(new EntitySet<SC,LO,GO,NO>(DefaultType));
#else
        EntitySetPtr removedEntities(new EntitySet<SC,NO>(DefaultType));
#endif
        for (UN i=0; i<getNumEntities(); i++) {
            if (getEntity(i)->getEntityFlag()==flag) {
                removedEntities->addEntities(getEntity(i));
                EntityVector_.erase(EntityVector_.begin()+i);
                i--;
            }
        }
        if (getNumEntities()-before>0) EntityMapIsUpToDate_ = false;
        return arcp(removedEntities);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::removeEntity(UN iD)
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::removeEntity(UN iD)
#endif
    {
        FROSCH_ASSERT(iD<getNumEntities(),"FROSch::EntitySet : ERROR: Cannot access Entity because iD>=getNumEntities().");
        EntityVector_.erase(EntityVector_.begin()+iD);
        EntityMapIsUpToDate_ = false;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::removeNodesWithDofs(GOVecView dirichletBoundaryDofs)
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::removeNodesWithDofs(GOVecView dirichletBoundaryDofs)
#endif
    {
        UN dofsPerNode = 0;
        if (getNumEntities()>0) dofsPerNode = EntityVector_[0]->getDofsPerNode();
        for (UN i=0; i<getNumEntities(); i++) {
            UN length = getEntity(i)->getNumNodes();
            for (UN j=0; j<length; j++) {
                UN itmp = length-1-j;
                UN k = 0;
                while (k<dofsPerNode) {
                    GO dofGlobal = getEntity(i)->getGlobalDofID(itmp,k);
                    if (std::binary_search(dirichletBoundaryDofs.begin(),dirichletBoundaryDofs.end(),dofGlobal)) {
                        getEntity(i)->removeNode(itmp);
                        break;
                    }
                    k++;
                }
            }
        }
        return 0;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::removeEmptyEntities()
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::removeEmptyEntities()
#endif
    {
        UN before = getNumEntities();
        for (UN i=0; i<getNumEntities(); i++) {
            if (getEntity(i)->getNumNodes()==0) {
                EntityVector_.erase(EntityVector_.begin()+i);
                i--;
            }
        }
        if (getNumEntities()-before>0) EntityMapIsUpToDate_ = false;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::sortUnique()
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::sortUnique()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            getEntity(i)->sortByGlobalID();
        }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        std::sort(EntityVector_.begin(),EntityVector_.end(),compareInterfaceEntities<SC,LO,GO,NO>);
        EntityVector_.erase(std::unique(EntityVector_.begin(),EntityVector_.end(),equalInterfaceEntities<SC,LO,GO,NO>),EntityVector_.end());
#else
        std::sort(EntityVector_.begin(),EntityVector_.end(),compareInterfaceEntities<SC,NO>);
        EntityVector_.erase(std::unique(EntityVector_.begin(),EntityVector_.end(),equalInterfaceEntities<SC,NO>),EntityVector_.end());
#endif
        EntityMapIsUpToDate_ = false;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    bool EntitySet<SC,LO,GO,NO>::checkForVertices()
#else
    template<class SC,class NO>
    bool EntitySet<SC,NO>::checkForVertices()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            if (getEntity(i)->getNumNodes()==1) {
                i--;
                return true;
            }
        }
        return false;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    bool EntitySet<SC,LO,GO,NO>::checkForShortEdges()
#else
    template<class SC,class NO>
    bool EntitySet<SC,NO>::checkForShortEdges()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            if (getEntity(i)->getNumNodes()==2) {
                i--;
                return true;
            }
        }
        return false;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    bool EntitySet<SC,LO,GO,NO>::checkForStraightEdges(UN dimension,
#else
    template<class SC,class NO>
    bool EntitySet<SC,NO>::checkForStraightEdges(UN dimension,
#endif
                                                       ConstXMultiVectorPtr &nodeList)
    {
        FROSCH_ASSERT(dimension==nodeList->getNumVectors(),"FROSch::EntitySet : ERROR: Inconsistent Dimension.");

        bool straight;
        LO length,j;
        SCVec pt1(dimension);
        SCVec dir1(dimension);
        SCVec dir2(dimension);

        for (UN i=0; i<getNumEntities(); i++) {
            straight = true;
            length = getEntity(i)->getNumNodes();
            j=2;
            if (length>2) {
                // Anfangssteigung berechnen
                for (UN k=0; k<dimension; k++) {
                    pt1[k] = nodeList->getData(k)[getEntity(i)->getLocalNodeID(0)];
                }
                for (UN k=0; k<dimension; k++) {
                    dir1[k] = nodeList->getData(k)[getEntity(i)->getLocalNodeID(1)]-pt1[k];
                }

                while (j<length) {
                    for (UN k=0; k<dimension; k++) {
                        dir2[k] = nodeList->getData(k)[getEntity(i)->getLocalNodeID(j)]-pt1[k];
                    }
                    if (!ismultiple<SC,LO>(dir1(),dir2())) {
                        straight = false;
                        break;
                    }
                    j++;
                }
                if (straight) {
                    i--;
                    return true;
                }
            }
        }
        return false;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    bool EntitySet<SC,LO,GO,NO>::checkForEmptyEntities()
#else
    template<class SC,class NO>
    bool EntitySet<SC,NO>::checkForEmptyEntities()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            if (getEntity(i)->getNumNodes()==0) {
                i--;
                return true;
            }
        }
        return false;
    }

    /////////////////
    // Set Methods //
    /////////////////

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::setUniqueIDToFirstGlobalNodeID()
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::setUniqueIDToFirstGlobalNodeID()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            getEntity(i)->sortByGlobalID();
            getEntity(i)->setUniqueIDToFirstGlobalID();
        }
        EntityMapIsUpToDate_ = false;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::setRootID()
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::setRootID()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            getEntity(i)->setRootID(i);
        }
        return 0;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::setLeafID()
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::setLeafID()
#endif
    {
        for (UN i=0; i<getNumEntities(); i++) {
            getEntity(i)->setLeafID(i);
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int EntitySet<SC,LO,GO,NO>::resetEntityType(EntityType type)
#else
    template<class SC,class NO>
    int EntitySet<SC,NO>::resetEntityType(EntityType type)
#endif
    {
        Type_ = type;
        for (UN i=0; i<getNumEntities(); i++) {
            getEntity(i)->resetEntityType(type);
        }
        return 0;
    }

    /////////////////
    // Get Methods //
    /////////////////

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    EntityType EntitySet<SC,LO,GO,NO>::getEntityType() const
#else
    template<class SC,class NO>
    EntityType EntitySet<SC,NO>::getEntityType() const
#endif
    {
        return Type_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    typename EntitySet<SC,LO,GO,NO>::UN EntitySet<SC,LO,GO,NO>::getNumEntities() const
#else
    template<class SC,class NO>
    typename EntitySet<SC,NO>::UN EntitySet<SC,NO>::getNumEntities() const
#endif
    {
        return EntityVector_.size();
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    const typename EntitySet<SC,LO,GO,NO>::InterfaceEntityPtrVec & EntitySet<SC,LO,GO,NO>::getEntityVector() const
#else
    template<class SC,class NO>
    const typename EntitySet<SC,NO>::InterfaceEntityPtrVec & EntitySet<SC,NO>::getEntityVector() const
#endif
    {
        return EntityVector_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    const typename EntitySet<SC,LO,GO,NO>::InterfaceEntityPtr EntitySet<SC,LO,GO,NO>::getEntity(UN iD) const
#else
    template<class SC,class NO>
    const typename EntitySet<SC,NO>::InterfaceEntityPtr EntitySet<SC,NO>::getEntity(UN iD) const
#endif
    {
        FROSCH_ASSERT(iD<getNumEntities(),"FROSch::EntitySet : ERROR: Cannot access Entity because iD>=getNumEntities().");
        return EntityVector_[iD];
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    const typename EntitySet<SC,LO,GO,NO>::XMapPtr EntitySet<SC,LO,GO,NO>::getEntityMap() const
#else
    template<class SC,class NO>
    const typename EntitySet<SC,NO>::XMapPtr EntitySet<SC,NO>::getEntityMap() const
#endif
    {
        FROSCH_ASSERT(EntityMapIsUpToDate_,"FROSch::EntitySet : ERROR:  the entity map has not been built or is not up to date.");
        return EntityMap_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    const typename EntitySet<SC,LO,GO,NO>::SCVecPtr EntitySet<SC,LO,GO,NO>::getDirection(UN dimension,
#else
    template<class SC,class NO>
    const typename EntitySet<SC,NO>::SCVecPtr EntitySet<SC,NO>::getDirection(UN dimension,
#endif
                                                                                         ConstXMultiVectorPtr &nodeList,
                                                                                         UN iD) const
    {
        FROSCH_ASSERT(iD<getNumEntities(),"FROSch::EntitySet : ERROR: Cannot access Entity because iD>=getNumEntities().");

        if (getEntity(iD)->getEntityFlag()==StraightFlag) {

            LO length = getEntity(iD)->getNumNodes();
            LO j=2;

            FROSCH_ASSERT(length>2,"Edge is not a straight edge!");

            bool straight=true;

            SCVec pt1(dimension);
            SCVec dir2(dimension);
            SCVecPtr dir1(dimension);

            // Anfangssteigung berechnen
            for (UN k=0; k<dimension; k++) {
                pt1[k] = nodeList->getData(k)[getEntity(iD)->getLocalNodeID(0)];
            }
            for (UN k=0; k<dimension; k++) {
                dir1[k] = nodeList->getData(k)[getEntity(iD)->getLocalNodeID(1)]-pt1[k];
            }

            while (j<length) {
                for (UN k=0; k<dimension; k++) {
                    dir2[k] = nodeList->getData(k)[getEntity(iD)->getLocalNodeID(j)]-pt1[k];
                }

                if (!ismultiple<SC,LO>(dir1(),dir2())) {
                    straight = false;
                    break;
                }
                j++;
            }

            FROSCH_ASSERT(straight,"FROSch::EntitySet : ERROR: Edge is not straight!");

            return dir1;

        } else if (getEntity(iD)->getEntityFlag()==ShortFlag) {

            int length = getEntity(iD)->getNumNodes();

            FROSCH_ASSERT(length==2,"FROSch::EntitySet : ERROR: Edge is not a short edge!");

            SCVec pt1(dimension);
            SCVecPtr dir1(dimension);

            // Anfangssteigung berechnen
            for (UN k=0; k<dimension; k++) {
                pt1[k] = nodeList->getData(k)[getEntity(iD)->getLocalNodeID(0)];
            }
            for (UN k=0; k<dimension; k++) {
                dir1[k] = nodeList->getData(k)[getEntity(iD)->getLocalNodeID(1)]-pt1[k];
            }

            return dir1;

        } else {
            FROSCH_ASSERT(false,"FROSch::EntitySet : ERROR: There is a problem while computing the direction of an edge!");
        }
    }
}

#endif
