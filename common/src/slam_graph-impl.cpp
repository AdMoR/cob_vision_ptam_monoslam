// This file is part of ScaViSLAM.
//
// Copyright 2011 Hauke Strasdat (Imperial College London)
//
// ScaViSLAM is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// any later version.
//
// ScaViSLAM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with ScaViSLAM.  If not, see <http://www.gnu.org/licenses/>.

#include "slam_graph.cpp"

#include "stereo_camera.h"
#include "transformations.h"
#include "utilities.h"

#include <g2o/core/robust_kernel_impl.h>

int current_id=0;
extern Matrix<double,3,3> g_camera_matrix;

namespace ScaViSLAM
{

template<>
void SlamGraph<SE3,StereoCamera, SE3XYZ_STEREO, 3>
::addPoseToG2o(const SE3 & T_me_from_w,
               int pose_id,
               bool fixed,
               g2o::SparseOptimizer * optimizer)
{
  G2oVertexSE3 * v_se3 = new G2oVertexSE3();

  v_se3->setId(pose_id);
  v_se3->setEstimate(T_me_from_w);
  v_se3->setFixed(fixed);

  optimizer->addVertex(v_se3);
}

template <>
void SlamGraph<SE3,StereoCamera, SE3XYZ_STEREO, 3>
::addObsToG2o(const Vector3d & obs,
              const Matrix3d & Lambda,
              int g2o_point_id,
              int pose_id,
              int anchor_id,
              bool robustify,
              double huber_kernel_width,
              g2o::SparseOptimizer * optimizer)
{
  G2oEdgeProjectPSI2UVU * e = new G2oEdgeProjectPSI2UVU();

  e->resize(3);

  g2o::OptimizableGraph::Vertex * point_vertex
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>
      (GET_MAP_ELEM(g2o_point_id, optimizer->vertices()));
  g2o::OptimizableGraph::Vertex * pose_vertex
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>
      (GET_MAP_ELEM(pose_id, optimizer->vertices()));
  g2o::OptimizableGraph::Vertex * anchor_vertex
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>
      (GET_MAP_ELEM(anchor_id, optimizer->vertices()));

  assert(point_vertex!=NULL);
  assert(point_vertex->dimension()==3);
  e->vertices()[0] = point_vertex;


  assert(pose_vertex!=NULL);
  assert(pose_vertex->dimension()==6);
  e->vertices()[1] = pose_vertex;


  assert(anchor_vertex!=NULL);
  assert(anchor_vertex->dimension()==6);
  e->vertices()[2] = anchor_vertex;
 // cout<<"creating edge from pose "<<pose_id<< " to point "<<g2o_point_id<< " with anchor "<<anchor_id<<endl;
  e->setMeasurement(obs);
  e->information() = Lambda;

  if (robustify)
  {
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
  }

  e->resizeParameters(1);
  bool param_status = e->setParameterId(0, 0);
  assert(param_status);

  optimizer->addEdge(e);
}

template <>
void SlamGraph<SE3,StereoCamera, SE3XYZ_STEREO, 3>
::addLineObsToG2o(const Vector6d & obs, //Here 6->3 rollllllback
              const Matrix6d & Lambda, //Again
              int g2o_line_id,
              int pose_id,
              bool robustify,
              double huber_kernel_width,
              g2o::SparseOptimizer * optimizer
             )// SE3 T_me_f_actkey)
{
  G2oEdgeSE3PlueckerLine * e = new G2oEdgeSE3PlueckerLine();

  e->resize(2);

  g2o::OptimizableGraph::Vertex * line_vertex
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>
      (GET_MAP_ELEM(g2o_line_id, optimizer->vertices()));

  g2o::OptimizableGraph::Vertex * pose_vertex
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>
      (GET_MAP_ELEM(pose_id, optimizer->vertices()));

  //cout<<"creating edge from pose "<<pose_id<< " to line "<<g2o_line_id<<endl;
  //cout<<"obs: "<<obs<<endl;

  assert(pose_vertex!=NULL);
  assert(pose_vertex->dimension()==6);
  e->vertices()[0] = pose_vertex;
  //pose_vertex->


   assert(line_vertex!=NULL);
  assert(line_vertex->dimension()==6);
  e->vertices()[1] = line_vertex;

  e->setMeasurement(obs);
  e->information() = Lambda;

//  SE3 tf;
  //tf=SE3(T_me_f_w);

//  if(true){
//
//	  Matrix<double,4,4> tr = T_me_f_w;//v1->estimate().matrix();
//	  Matrix<double,3,3> roo;
//	  Vector3d t;
//	  for(int i=0;i<3;i++){
//		  for(int j=0;j<3;j++)
//			  roo(i,j)=T_me_f_w(i,j);
//		  t(i)=T_me_f_w(i,3);
//	  }
//	  SE3 s= SE3(roo,t);
//	  Vector3d tra=s.translation();
//	  Quaterniond  ro=s.unit_quaternion();
//	  dumpToFile("Transform from frame to world (approximate) : translation + quat",tra(0),tra(1),tra(2),double(ro.x()),double(ro.y()),double(ro.z()),double(ro.w()),"/home/rmb-am/Slam_datafiles/ProjectionMatrixLines.txt");
//	  Matrix<double,3,6> projMat=computeLineProjectionMatrix(computeProjectionMatrix(g_camera_matrix, tr ));
//	  dumpToFile("projection matrix of line : ", g2o_line_id, pose_id,0,0,0,0,0,"/home/rmb-am/Slam_datafiles/ProjectionMatrixLines.txt");
//	  dumpToFile("line1 : ",projMat(0,0), projMat(0,1), projMat(0,2), projMat(0,3),projMat(0,4),projMat(0,5),pow(10,7),"/home/rmb-am/Slam_datafiles/ProjectionMatrixLines.txt");
//	  dumpToFile("line2 : ",projMat(1,0), projMat(1,1), projMat(1,2), projMat(1,3),projMat(1,4),projMat(1,5),pow(10,7),"/home/rmb-am/Slam_datafiles/ProjectionMatrixLines.txt");
//	  dumpToFile("line3 : ",projMat(2,0), projMat(2,1), projMat(2,2), projMat(2,3),projMat(2,4),projMat(2,5),pow(10,7),"/home/rmb-am/Slam_datafiles/ProjectionMatrixLines.txt");
//	  dumpToFile("projection of line : ", g2o_line_id, 0, 0,0,0,0,0,"/home/rmb-am/Slam_datafiles/ProjectionMatrixLines.txt");
//	  dumpToFile("ax+by+c : ", obs(0),obs(1),obs(2),pow(10,7),pow(10,7),pow(10,7),pow(10,7),"/home/rmb-am/Slam_datafiles/ProjectionMatrixLines.txt");
//	  dumpToFile("", 0,0,0,0,0,0,0,"/home/rmb-am/Slam_datafiles/ProjectionMatrixLines.txt");
//  }

  e->setId(current_id);
  current_id+=1;
  //e->T_cur_f_actkey=T_me_f_actkey;

  if (robustify)
  {
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
  }

  e->resizeParameters(1);
  bool param_status = e->setParameterId(0, 0);
  assert(param_status);

  optimizer->addEdge(e);
}

template <>
void SlamGraph<SE3,StereoCamera, SE3XYZ_STEREO, 3>
::addConstraintToG2o(const SE3 & T_2_from_1,
                     const Matrix6d &
                     Lambda_2_from_1,
                     int pose_id_1,
                     int pose_id_2,
                     g2o::SparseOptimizer * optimizer)
{

  G2oEdgeSE3 * e = new G2oEdgeSE3();

  g2o::HyperGraph::Vertex * pose1_vertex
      = GET_MAP_ELEM(pose_id_1, optimizer->vertices());
  e->vertices()[0]
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>(pose1_vertex);

  g2o::HyperGraph::Vertex * pose2_vertex
      = GET_MAP_ELEM(pose_id_2, optimizer->vertices());
  e->vertices()[1]
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>(pose2_vertex);

  e->setMeasurement( T_2_from_1);
  e->information() = Lambda_2_from_1;


  optimizer->addEdge(e);
}

#ifdef MONO
template<>
void SlamGraph<Sim3,LinearCamera, Sim3XYZ, 2>
::addPoseToG2o(const Sim3 & T_me_from_w,
               int pose_id,
               bool fixed,
               g2o::SparseOptimizer * optimizer)
{
  G2oVertexSim3 * v_sim3 = new G2oVertexSim3();

  v_sim3->setId(pose_id);
  v_sim3->estimate() = T_me_from_w;
  v_sim3->setFixed(fixed);

  v_sim3->focal_length = cam_.focal_length();
  v_sim3->principle_point[0] = cam_.principal_point()[0];
  v_sim3->principle_point[1] = cam_.principal_point()[1];

  optimizer->addVertex(v_sim3);
}


template <>
void SlamGraph<Sim3,LinearCamera, Sim3XYZ, 2>
::addObsToG2o(const Vector2d & obs,
              const Matrix2d & Lambda,
              int g2o_point_id,
              int pose_id,
              int anchor_id,
              bool robustify,
              double huber_kernel_width,
              g2o::SparseOptimizer * optimizer)
{
  G2oEdgeSim3ProjectUVQ * e = new G2oEdgeSim3ProjectUVQ();
  // TODO: implement anchored edges!!
  assert(false);

  e->resize(3);

  g2o::HyperGraph::Vertex * point_vertex
      = GET_MAP_ELEM(g2o_point_id, optimizer->vertices());
  e->vertices()[0]
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>(point_vertex);

  g2o::HyperGraph::Vertex * pose_vertex
      = GET_MAP_ELEM(pose_id, optimizer->vertices());
  e->vertices()[1]
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>(pose_vertex);

  g2o::HyperGraph::Vertex * anchor_vertex
      = GET_MAP_ELEM(anchor_id, optimizer->vertices());
  e->vertices()[2]
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>(anchor_vertex);

  e->measurement() = obs;
  e->inverseMeasurement() = -obs;
  e->information() = Lambda;

  if (robustify)
  {
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
  }

  e->resizeParameters(1);
  bool param_status = e->setParameterId(0, 0);
  assert(param_status);

  optimizer->addEdge(e);
}


template <>
void SlamGraph<Sim3, LinearCamera, Sim3XYZ, 2>
::addConstraintToG2o(const Sim3 & T_2_from_1,
                     const Matrix6d & Lambda_2_from_1,
                     int pose_id_1,
                     int pose_id_2,
                     g2o::SparseOptimizer * optimizer)
{
  G2oEdgeSim3 * e = new G2oEdgeSim3();

  g2o::HyperGraph::Vertex * pose1_vertex
      = GET_MAP_ELEM(pose_id_1, optimizer->vertices());
  e->vertices()[0]
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>(pose1_vertex);

  g2o::HyperGraph::Vertex * pose2_vertex
      = GET_MAP_ELEM(pose_id_2, optimizer->vertices());
  e->vertices()[1]
      = dynamic_cast<g2o::OptimizableGraph::Vertex*>(pose2_vertex);

  e->measurement() = T_2_from_1;
  e->inverseMeasurement() = T_2_from_1.inverse();
  e->information() = Lambda_2_from_1;

  optimizer->addEdge(e);
}


namespace SlamGraphMethods
{
template <>
bool
restorePoseFromG2o<Sim3, LinearCamera, Sim3XYZ, 2>
(const g2o::HyperGraph::Vertex * g2o_vertex,
 SlamGraph<Sim3, LinearCamera, Sim3XYZ, 2>::VertexTable * vertex_map)
{
  const G2oVertexSim3 * v_sim3
      = dynamic_cast<const G2oVertexSim3*>(g2o_vertex);
  if (v_sim3!=NULL)
  {
    int frame_id = v_sim3->id();
    SlamGraph<Sim3, LinearCamera, Sim3XYZ, 2>::Vertex &
        frame_vertex = GET_MAP_ELEM_REF(frame_id, vertex_map);
    frame_vertex.T_me_from_world = v_sim3->estimate();
    return true;
  }
  return false;
}
}
#endif

template class SlamGraph<SE3,StereoCamera,SE3XYZ_STEREO,3>;
#ifdef MONO
template class SlamGraph<Sim3,LinearCamera,Sim3XYZ,2>;
#endif


}

