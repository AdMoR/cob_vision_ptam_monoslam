/*
 * mono_matcher.cpp
 *
 *  Created on: Nov 19, 2013
 *      Author: rmb-am
 */


#include "matcher.hpp"

#include <stdint.h>
#include <tgmath.h>

#include <sophus/se3.h>
#include <fstream>
#include <visiontools/accessor_macros.h>

#include "data_structures.h"
#include "homography.h"
#include "keyframes.h"
#include "transformations.h"

#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;


namespace ScaViSLAM
{

template <class Camera>
uint8_t GuidedMatcher<Camera>::KEY_PATCH[BOX_AREA];
template <class Camera>
uint8_t GuidedMatcher<Camera>::CUR_PATCH[BOX_AREA];


//Only works for 5 obs
void fillTheMatrix( Matrix<double,15,6> A,Matrix<double,15,1> b,vector< pair< Vector3d, Matrix<double,3,6> > > obsList,int nb_obs ){

	for(int i=0;i<nb_obs;i++){
		for(int j=0;j<3;j++){
			for(int k=0;k<6;k++){
				A(i*nb_obs+j,k)=obsList[i].second(j,k);
			}
			b(i*nb_obs+j)=obsList[i].first(j);
		}
	}

}



//	template <class Camera>
//	void GuidedMatcher<Camera>::lineMatcher(std::vector<Line> &linesOnCurrentFrame,
//											tr1::unordered_map<int,Line> &tracked_lines,
//											const SE3 & T_cur_from_actkey,
//											int actkey_id,
//											const ALIGNED<FrontendVertex>::int_hash_map & vertex_map,
//											const Matrix<double,3,3> camera_matrix,
//											cv::Mat *curFrameRGB, const vector<Vector3d> & edges,
//											MonoFrontend *monoFrontednPtr,
//											AddToOptimzerPtr & to_optimizer,
//											std::vector<int> & localIDsofNewLinesToBeAdded,
//											SE3 &T_cur_from_w){
//
//
//			int OptimThres=5;
//
//			bool display=false;
//			bool one_line_track=true;
//			int line_tracked=0;
//			bool color_mode=false;
//			//bool vertical_mode=true;
//
//
//
//			  SE3 T_actkey_from_w    = GET_MAP_ELEM(actkey_id, vertex_map).T_me_from_w;
//			  T_cur_from_w = T_cur_from_actkey*T_actkey_from_w; //Position wo  sich die Kamera in Weltkoordinaten befindet nach dem Optischen FLuß
//
//
//
//			  //Build color map of candidate lines
//			  vector<pair<int,cv::Scalar>> colormap;
//			  float n=0,m=linesOnCurrentFrame.size()-1;
//			  for(auto ptr=linesOnCurrentFrame.begin();ptr!=linesOnCurrentFrame.end();ptr++){
//				  if(n<m/3)
//					  colormap.push_back(make_pair((*ptr).global_id,cv::Scalar(floor(256.*3*n/m),255, 0)));
//				  else{
//					  if(n<(2*m/3))
//						  colormap.push_back(make_pair((*ptr).global_id,cv::Scalar(0,floor(256.*n/m), 128)));
//					  else
//						  colormap.push_back(make_pair((*ptr).global_id,cv::Scalar(0,0,floor(256.*n/m))));
//				  }
//				  n++;}
//
//
//			  //Build match map of candidate lines
//			 map<int,int> matchmap;
//			 for(auto ptr=linesOnCurrentFrame.begin();ptr!=linesOnCurrentFrame.end();ptr++)
//			 	matchmap.insert(make_pair((*ptr).global_id,-1));
//
//
//			  //Projiziere diese ins aktuelle Kamerabild, gucken ob die drin sind, falls nicht nicht weitermachen
//			 //cout << T_cur_from_w.matrix() << endl;
//			 Matrix<double, 3, 4> projectionsMatrix = computeProjectionMatrix(camera_matrix, T_cur_from_w.matrix() );
//			 int line_num=0;
//			  cout<<" size of linesOnCurrentFrame : "<<linesOnCurrentFrame.size()<<endl;
//			  cout<<" size of trackedlines: "<<tracked_lines.size()<<endl;
//
//			  if(linesOnCurrentFrame.size()>0)
//			  {
//
//
//				  set<int> currentLineLocalId;
//				  for (auto tracked_lines_iter = tracked_lines.begin(); tracked_lines_iter != tracked_lines.end(); ++tracked_lines_iter)
//				  {
//
//					  if((*tracked_lines_iter).second.active){
//						  //Do normal stuff
//					  }
//					  else{
//						  double normalizedError;
//						  map<int,Line> candidateLinesOrderedByError;
//						  for(auto ptr=linesOnCurrentFrame.begin();ptr!=linesOnCurrentFrame.end();ptr++){
//							  if(matchTwoLinesSSD((*tracked_lines_iter).second.descriptor,(*ptr).descriptor, normalizedError, false) && GuidedMatcher<Camera>::closeLinearForm((*tracked_lines_iter).second.linearForm,(*ptr).linearForm,10,normalizedError) ){
//								  //Add the current line to the candidates
//								  candidateLinesOrderedByError.insert(make_pair(normalizedError,(*ptr)));
//							  }
//						  }
//						  Line bestMatch = candidateLinesOrderedByError.begin()->second;
//						  //Add the linear form to the list of observation
//						  Matrix<double, 3, 4> projectionsMatrix = computeProjectionMatrix(camera_matrix, T_cur_from_w.matrix() );
//						  //Add the 2d observation to the tracked line to prepare its 3D estimation
//						  (*tracked_lines_iter).second.obsList.push_back(make_pair(bestMatch.linearForm,computeLineProjectionMatrix(projectionsMatrix)));
//
//						  if((*tracked_lines_iter).second.obsList.size()==OptimThres){
//							  //Do the optim
//							  //Build the 2 matrix A : OptimThres*3 x 6, and b= OptimThres *3 x 1
//							  Matrix<double,3*OptimThres,6> A;
//							  Matrix<double,3*OptimThres,1> b;
//							  fillTheMatrix(A,b,(*tracked_lines_iter).second.obsList,OptimThres);
//							  (*tracked_lines_iter).second.optimizedPluckerLines=A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
//							  cout << "estimated absolute Plücker : " << (*tracked_lines_iter).second.optimizedPluckerLines << endl;
//							  (*tracked_lines_iter).second.active=true;
//						  }
//					  }
//
//				  }
//
//
//			  }
//			  else
//				  cout << "no line on frame !" << endl;
//
//		return;
//	}
}
