from propagation cimport Instantgnn,Instantgnn_transpose
import numpy as np
from libcpp.utility cimport pair
cdef class InstantGNN:
	cdef Instantgnn c_instantgnn

	def __cinit__(self):
		self.c_instantgnn=Instantgnn()

	def init_graph(self,path,dataset,np.ndarray array1,layer,r,weights,num_thread):
		return self.c_instantgnn.init_graph(path.encode(),dataset.encode(),Map[MatrixXi](array1),layer,r,weights,num_thread)

	def init_push_graph(self,path,dataset,np.ndarray array0,np.ndarray array1,layer,r,weights,num_thread,rmax):
		return self.c_instantgnn.init_push_graph(path.encode(),dataset.encode(),Map[MatrixXd](array0),Map[MatrixXi](array1),layer,r,weights,num_thread,rmax)

	def PushMethod(self, np.ndarray result,rmax):
		self.c_instantgnn.PushMethod(Map[MatrixXd](result),rmax)

	def PowerMethod(self,np.ndarray result):
		return self.c_instantgnn.PowerMethod(Map[MatrixXd](result))

	def UpdateEdges(self,pairs, np.ndarray result,num_thread,rmax):
		cdef vector[pair[uint,uint]] cpp_pairs=[(x[0],x[1]) for x in pairs] 
		return self.c_instantgnn.UpdateEdges(cpp_pairs,Map[MatrixXd](result),num_thread,rmax)

	def UpdateStruct(self,pairs):
		cdef vector[pair[uint,uint]] cpp_pairs=[(x[0],x[1]) for x in pairs] 
		return self.c_instantgnn.UpdateStruct(cpp_pairs)
	
	def UpdateNodes(self,nodes, np.ndarray result,num_thread,rmax):
		return self.c_instantgnn.UpdateNodes(nodes,Map[MatrixXd](result),num_thread,rmax)
	
	def UpdateFeatures(self, nodes, np.ndarray result,num_thread,rmax):
		return self.c_instantgnn.UpdateFeatures(nodes,Map[MatrixXd](result),num_thread,rmax)

	def GetResidueSum(self,res):
		return self.c_instantgnn.GetResidueSum(Map[VectorXd](res))
	
	def UpdateStructNodes(self,nodes):
		return self.c_instantgnn.UpdateStructNodes(nodes)

cdef class InstantGNN_transpose:
	cdef Instantgnn_transpose c_instantgnn_transpose
	
	def __cinit__(self):
		self.c_instantgnn_transpose=Instantgnn_transpose()

	def init_graph(self,path,dataset,np.ndarray array1,layer,r,weights,num_thread):
		return self.c_instantgnn_transpose.init_graph(path.encode(),dataset.encode(),Map[MatrixXi](array1),layer,r,weights,num_thread)
	
	def init_push_graph(self,path,dataset,np.ndarray array0,np.ndarray array1,layer,r,weights,num_thread,rmax):
		return self.c_instantgnn_transpose.init_push_graph(path.encode(),dataset.encode(),Map[MatrixXd](array0),Map[MatrixXi](array1),layer,r,weights,num_thread,rmax)
	
	def PushMethod(self, np.ndarray result,rmax):
		self.c_instantgnn_transpose.PushMethod(Map[MatrixXd](result),rmax)

	def PowerMethod(self,np.ndarray result):
		return self.c_instantgnn_transpose.PowerMethod(Map[MatrixXd](result))	
	
	def UpdateEdges(self,pairs, np.ndarray result,num_thread,rmax):
		cdef vector[pair[uint,uint]] cpp_pairs=[(x[0],x[1]) for x in pairs] 
		return self.c_instantgnn_transpose.UpdateEdges(cpp_pairs,Map[MatrixXd](result),num_thread,rmax)

	def UpdateStruct(self,pairs):
		cdef vector[pair[uint,uint]] cpp_pairs=[(x[0],x[1]) for x in pairs] 
		return self.c_instantgnn_transpose.UpdateStruct(cpp_pairs)
	
	def UpdateNodes(self,nodes, np.ndarray result,num_thread,rmax):
		return self.c_instantgnn_transpose.UpdateNodes(nodes,Map[MatrixXd](result),num_thread,rmax)
	
	def UpdateFeatures(self, nodes, np.ndarray result,num_thread,rmax):
		return self.c_instantgnn_transpose.UpdateFeatures(nodes,Map[MatrixXd](result),num_thread,rmax)

	def GetResidueSum(self,res):
		return self.c_instantgnn_transpose.GetResidueSum(Map[VectorXd](res))

	def UpdateStructNodes(self,nodes):
		return self.c_instantgnn_transpose.UpdateStructNodes(nodes)