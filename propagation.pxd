from eigency.core cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport pair
ctypedef unsigned int uint

cdef extern from "instantAlg.cpp":
	pass

cdef extern from "instantAlg_transpose.cpp":
	pass
	
cdef extern from "instantAlg.h" namespace "propagation":
	cdef cppclass Instantgnn:
		Instantgnn() except+
		double init_graph(string,string,const Map[MatrixXi] &,uint,double,const vector[double]&,uint ) except +
		double init_push_graph(string,string,const Map[MatrixXd] &,const Map[MatrixXi] &,uint,double,const vector[double]&,uint,double ) except +
		void PushMethod(Map[MatrixXd]&, double)  except +
		double PowerMethod(Map[MatrixXd]&)  except +
		double UpdateEdges(const vector[pair[uint, uint]] &, Map[MatrixXd]&, uint, double) except +
		double UpdateStruct(const vector[pair[uint, uint]] &) except +
		double UpdateNodes(const vector[uint] &, Map[MatrixXd]&, uint, double) except +
		double UpdateFeatures(const vector[uint] &, Map[MatrixXd]&, uint, double) except +
		double GetResidueSum(Map[VectorXd]&) except +
		double UpdateStructNodes(const vector[uint] &) except +

	cdef cppclass Instantgnn_transpose:
		Instantgnn_transpose() except+
		double init_graph(string,string,const Map[MatrixXi] &,uint,double,const vector[double]&,uint ) except +
		double init_push_graph(string,string,const Map[MatrixXd] &,const Map[MatrixXi] &,uint,double,const vector[double]&,uint,double ) except +
		void PushMethod(Map[MatrixXd]&, double)  except +
		double PowerMethod(Map[MatrixXd]&)  except +
		double UpdateEdges(const vector[pair[uint, uint]] &, Map[MatrixXd]&, uint, double) except +
		double UpdateStruct(const vector[pair[uint, uint]] &) except +
		double UpdateNodes(const vector[uint] &, Map[MatrixXd]&, uint, double) except +
		double UpdateFeatures(const vector[uint] &, Map[MatrixXd]&, uint, double) except +
		double GetResidueSum(Map[VectorXd]&) except +
		double UpdateStructNodes(const vector[uint] &) except +


# cdef extern from "sampling/sampling.cpp":
#	  pass
