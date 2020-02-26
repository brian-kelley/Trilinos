#ifndef __IFPACK2_BASKERILU_HPP__
#define __IFPACK2_BASKERILU_HPP__
//Include needed Trilinos
#include <Tpetra_CrsMatrix.hpp>
#include <Kokkos_DefaultNode.hpp>
#include <Kokkos_CrsMatrix.hpp>
#include <Ifpack2_Preconditioner.hpp>

//Include needed Basker
#include "../src/shylubasker_decl.hpp"

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node, class ExecSpace>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
class Baskerilu : public Ifpack2::Preconditioner<Scalar,LocalOrdinal, GlobalOrdinal, Node>
#else
class Baskerilu : public Ifpack2::Preconditioner<Scalar, Node>
#endif
{
public:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal,GlobalOrdinal,Node> TCrsMatrix;
  typedef Kokkos::CrsMatrix<Scalar, LocalOrdinal,GlobalOrdinal,Node> KCrsMatrix;
#else
  typedef Tpetra::CrsMatrix<Scalar,Node> TCrsMatrix;
  typedef Kokkos::CrsMatrix<Scalar,Node> KCrsMatrix;
#endif
  typedef Kokkos::View<LocalOrdinal*, ExecSpace> OrdinalArray;
  typedef Kokkos::View<Scalar*, ExecSpace> ScalarArray;
 
private:
  bool initFlag;
  bool computedFlag;
  int nInit;
  int nApply;
  int nComputed;
  double initTime;
  double computeTime;
  double applyTime;
  Teuchos::RCP<TCrsMatrix> mat;
  
public:
  Baskerilu()
  {
    //Made to fit the needs of your solve
  }
  
  //required by IfPack2
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  Teuchos::RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >
#else
  Teuchos::RCP<const Tpetra::Map<Node> >
#endif
  getDomainMap() const
  {
    return mat->getDomainMap();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  Teuchos::RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >
#else
  Teuchos::RCP<const Tpetra::Map<Node> >
#endif
  getRangeMap() const
  {
    return mat->getRangeMap();
  }

  void
  apply
  (
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   const Tpetra::MultVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &X,
   Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &Y,
#else
   const Tpetra::MultVector<Scalar,Node> &X,
   Tpetra::MultiVector<Scalar,Node> &Y,
#endif
   Teuchos::ETransp mode = Teuchos::NO_TRANS,
   Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
   Scalar beta   =Teuchos::ScalarTraits<Scalar>::two()
   )const
  {

    return;
  }//end of apply()

  void setParameters(const Teuchos::ParameterList& List)
  {
     return;
  }//end setParameters()
	
  void initialize()
  {
    
    return;
  }//end initialize()

  bool isInitialized() const
  {
    return false;
  }//end isInitialized()

  void compute()
  {
    return;
  }//end compute()

  bool isComputed() const
  {
    return false;
  }//end isComputed()

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  Teuchos::RCP<const Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > 
#else
  Teuchos::RCP<const Tpetra::RowMatrix<Scalar,Node> > 
#endif
  getMatrix() const
  {

    
  }//end getMatrix()

  int getNumInitialize() const
  {

  }//end getNumInitialize()

  int getNumCompute() const
  {  
    
  }//end getNumCompute()

  int getNumApply() const
  {

  }//end getNumApply()

  double getInitializeTime() const
  {

  }//end getInitializeTime

  double getComputeTime() const
  {

  }//end getComputeTime

  double getApplyTime() const
  {
    
  }//end getApplyTime()

  void checkLocalILU()
  {
    
  }//end checkLocalILU()

  void checkLocalIC()
  {
    
  }//end checkIC()

  void printStatus()
  {

  }//end printStat()

};//end class Baskerilu

#endif //end __IFPACK2_BASKERILU_HPP__
