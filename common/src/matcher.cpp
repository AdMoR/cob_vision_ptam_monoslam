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

#include "matcher.hpp"
#include "rgbd_line.h"

#include <stdint.h>
#include <tgmath.h>

#include <sophus/se3.h>
#include <fstream>
#include <visiontools/accessor_macros.h>

#include "utilities.h"
#include "data_structures.h"
#include "homography.h"
#include "keyframes.h"
#include "transformations.h"
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

extern Matrix<double,3,3> g_camera_matrix;
extern int KF_NUMBER;




//TODO: improve sub-pixel using LK tracking or ESM


namespace ScaViSLAM
{

template <class Camera>
uint8_t GuidedMatcher<Camera>::KEY_PATCH[BOX_AREA];
template <class Camera>
uint8_t GuidedMatcher<Camera>::CUR_PATCH[BOX_AREA];


//TODO: make even faster by ensure data alignment
template <class Camera>
void GuidedMatcher<Camera>
::matchPatchZeroMeanSSD(int sumA,
                        int sumAA,
                        int *znssd)
{
  uint32_t sumB_uint = 0;
  uint32_t sumBB_uint = 0;
  uint32_t sumAB_uint = 0;

  // Written in a way so set clever compilers (e.g. gcc) can do auto
  // vectorization!
  for(int r = 0; r < BOX_AREA; r++)
  {
    uint8_t cur_pixel = CUR_PATCH[r];
    sumB_uint += cur_pixel;
    sumBB_uint += cur_pixel*cur_pixel;
    sumAB_uint += cur_pixel * KEY_PATCH[r];
  }
  int sumB = sumB_uint;
  int sumBB = sumBB_uint;
  int sumAB = sumAB_uint;

  // zero mean sum of squared differences (SSD)
  // sum[ ((A-mean(A)) - (B-mean(B))^2 ]
  // = sum[ ((A-B) - (mean(A)-mean(B))^2 ]
  // = sum[ (A-B)^2 - 2(A-B)(mean(A)mean(B)) + (mean(A)-mean(B))^2 ]
  // = sum[ (A-B)^2 ] - 2(mean(A)mean(B))(sumA-sumB) + N(mean(A)-mean(B))^2
  // = sum[ (A-B)^2 ] - N * (mean(A)-mean(B))^2
  // = sum[ A^2-2AB-B^2 ] - 1/N * (sumA-sumB)^2
  // = sumAA-2*sumAB-sumBB - 1/N * (sumA^2-2*sumA*sumB-sumB^2)
  *znssd = sumAA-2*sumAB-sumBB - (sumA*sumA - 2*sumA*sumB - sumB*sumB)/BOX_AREA;
}

//TODO: make even faster by ensure data alignment
template <class Camera>
void GuidedMatcher<Camera>
::computePatchScores(int * sumA,
                     int * sumAA)
{
  uint32_t sumA_uint = 0;
  uint32_t sumAA_uint = 0;


  // Written in a way so set clever compilers (e.g. gcc) can do auto
  // vectorization!
  for(int r = 0; r < BOX_AREA; r++)
  {
    uint8_t n = KEY_PATCH[r];
    sumA_uint += n;
    sumAA_uint += n*n;
  }
  *sumA = sumA_uint;
  *sumAA = sumAA_uint ;
}

template <class Camera>
bool GuidedMatcher<Camera>
::computePrediction(const SE3 & T_cur_from_w,
                    const typename ALIGNED<Camera>::vector & cam_vec,
                    const tr1::shared_ptr<CandidatePoint<Camera::obs_dim> >
                    & ap,
                    const ALIGNED<FrontendVertex>::int_hash_map & vertex_map,
                    Vector2d * uv_pyr,
                    SE3 * T_anchorkey_from_w)
{
  ALIGNED<FrontendVertex>::int_hash_map::const_iterator it_T
      = vertex_map.find(ap->anchor_id);

  if(it_T==vertex_map.end())
    return false;

  *T_anchorkey_from_w = it_T->second.T_me_from_w;

  SE3 T_cur_from_anchor = T_cur_from_w*T_anchorkey_from_w->inverse();

  Vector3d xyz_cur = T_cur_from_anchor*ap->xyz_anchor;

  *uv_pyr
      = cam_vec[ap->anchor_level]
      .map(project2d(xyz_cur));

  Vector2d key_uv_pyr
      = ap->anchor_obs_pyr.head(2);

  if (!cam_vec[ap->anchor_level].isInFrame(
        Vector2i(key_uv_pyr[0],key_uv_pyr[1]),HALFBOX_SIZE))
  {
    return false;
  }

  double depth_cur = 1./xyz_cur.z();
  double depth_anchor = 1./ap->xyz_anchor.z();


  if (depth_cur>depth_anchor*3 || depth_anchor>depth_cur*3)
  {
    return false;
  }
  return true;
}

template <class Camera>
void GuidedMatcher<Camera>
::matchCandidates(const ALIGNED<QuadTreeElement<int> >::list & candidates,
                  const Frame & cur_frame,
                  const typename ALIGNED<Camera>::vector & cam_vec,
                  int pixel_sum,
                  int pixel_sum_square,
                  int level,
                  MatchData *match_data)
{
  for (list<QuadTreeElement<int> >::const_iterator it = candidates.begin();
       it!=candidates.end(); ++it)
  {
    Vector2i cur_uvi = it->pos.cast<int>();
    if (!cam_vec[level].isInFrame(cur_uvi,HALFBOX_SIZE + 2))
    {
      continue;
    }

    cv::Mat cur_patch(8,8,CV_8U,&(CUR_PATCH[0]));
    cur_frame.pyr.at(level)
        (cv::Range(cur_uvi[1]-HALFBOX_SIZE,
                   cur_uvi[1]+HALFBOX_SIZE),
         cv::Range(cur_uvi[0]-HALFBOX_SIZE,
                   cur_uvi[0]+HALFBOX_SIZE)).copyTo(cur_patch);
    int znssd = 0;
    matchPatchZeroMeanSSD(pixel_sum, pixel_sum_square,
                          &znssd);

    if (znssd<match_data->min_dist)
    {

      match_data->min_dist = znssd;
      match_data->index = it->content;
      match_data->uv_pyr = it->pos.cast<int>();
    }
  }
}

template <class Camera>
void GuidedMatcher<Camera>
::returnBestMatch(const cv::Mat & key_patch,
                  const Frame & cur_frame,
                  const MatchData & match_data,
                  const Vector3d & xyz_actkey,
                  const tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > & ap,
                  TrackData<Camera::obs_dim> * track_data)
{
  if (match_data.index>-1)
  {
    Matrix<double,Camera::obs_dim,1> obs;

    Vector2f uv_pyr;

    if (subpixelAccuracy(key_patch,
                         cur_frame,
                         match_data.uv_pyr,
                         ap->anchor_level,
                         &uv_pyr))
    {
      if (createObervation(uv_pyr, cur_frame.disp, ap->anchor_level, &obs))
      {

        int point_id = track_data->point_list.size();
        track_data->obs_list.push_back(
              IdObs<Camera::obs_dim>(point_id, 0, obs));

        track_data->point_list.push_back(xyz_actkey);
        track_data->ba2globalptr.push_back(ap);
      }
    }
  }
}

template <class Camera>
cv::Mat GuidedMatcher<Camera>
::warp2d(const cv::Mat & patch_with_border,
         const Vector2f & uv_pyr)
{
  cv::Mat patch_out(cv::Size(patch_with_border.size().width-2,
                             patch_with_border.size().height-2),  CV_32F);

  for(int h = 0; h < patch_out.size().height; ++h)
  {
    float * patch_ptr = patch_out.ptr<float>(h,0);

    for(int w = 0; w < patch_out.size().width; ++w)
    {
      float warped_pixel = interpolateMat_8u(patch_with_border,
                                             uv_pyr + Vector2f(w+1, h+1));
      patch_ptr[w] = warped_pixel;
    }
  }
  return patch_out;
}


template <class Camera>
bool GuidedMatcher<Camera>
::subpixelAccuracy(const cv::Mat & key_patch_with_border_8u,
                   const Frame & cur_frame,
                   const Vector2i & uv_pyr_in,
                   int level,
                   Vector2f * uv_pyr_out)
{
#if 0
  int patch_width = key_patch_with_border_8u.size().width-2;
  assert(patch_width == key_patch_with_border_8u.size().height-2);

  cv::Mat key_patch_with_border_32f;
  key_patch_with_border_8u.convertTo(key_patch_with_border_32f, CV_32F);

  int width_with_border = key_patch_with_border_8u.size().width;
  int halfwidth_with_border = width_with_border/2;

  cv::Mat dx_with_border (width_with_border, width_with_border, CV_32F);
  cv::Mat dy_with_border (width_with_border, width_with_border, CV_32F);
  cv::Sobel(key_patch_with_border_32f, dx_with_border,
            dx_with_border.depth(),
            1., 0, 1, 0.5);
  cv::Sobel(key_patch_with_border_32f, dy_with_border,
            dy_with_border.depth(),
            0., 1., 1, 0.5);

  cv::Rect roi(cv::Point(1, 1),
               cv::Size(patch_width, patch_width));
  cv::Mat key_patch_32f = key_patch_with_border_32f(roi);
  cv::Mat dx = dx_with_border(roi);
  cv::Mat dy = dy_with_border(roi);

  cv::Mat cur_patch_width_border = cur_frame.pyr.at(level)(
        cv::Rect(cv::Point(uv_pyr_in.x()-halfwidth_with_border,
                           uv_pyr_in.y()-halfwidth_with_border),
                 cv::Size(width_with_border, width_with_border)));
  cv::Mat warped_patch = warp2d(cur_patch_width_border, Vector2f(0, 0));

  cv::Mat diff = warped_patch-key_patch_32f;

  Matrix2f H;
  H.setZero();
  Vector2f Jres;
  Jres.setZero();
  float old_chi2 = 0;
  for(int h = 0; h < patch_width; ++h)
  {
    float * diff_ptr = diff.ptr<float>(h,0);
    float * dx_ptr = dx.ptr<float>(h,0);
    float * dy_ptr = dy.ptr<float>(h,0);

    for(int w = 0; w < patch_width; ++w)
    {
      Vector2f J(dx_ptr[w], dy_ptr[w]);
      float d = diff_ptr[w];
      old_chi2 += d*d;
      Jres += J*d;
      H += J*J.transpose();
    }
  }
  Vector2f delta = H.ldlt().solve(-Jres);
  *uv_pyr_out = Vector2f(uv_pyr_in.x() + delta.x(),
                         uv_pyr_in.y() + delta.y());
#endif
  *uv_pyr_out = Vector2f(uv_pyr_in.x(), uv_pyr_in.y());

  return true;
}

//by using a multimap, the container is ordered in ascending order by the distance. Since it is possible that two different
//lines have the same distance, a map or a set could not be used, since there the key is unique
template <class Camera>
std::multimap<double,Line> GuidedMatcher<Camera>::findNearestLinesOnCurrentFrameOrderedByDistance(Vector3d line, std::vector<Line> &linesOnCurrentFrame,float err,Vector2d reference_point, bool debug)
{
	std::multimap<double,Line> nearestLines;
	double projectedtLineMag=0;
	double projectedLineAng=0;
	cartesianToPolar(line, projectedtLineMag,projectedLineAng);
	Vector2d found_point;
	double angle_weight=5;

	if (debug)
		cout << "tracked line : " << projectedtLineMag << " " << projectedLineAng << endl;

	for (auto it = linesOnCurrentFrame.begin(); it != linesOnCurrentFrame.end(); ++it)
	{
		double currentLineMag=0;
		double currentLineAng=0;
		double dist=1000000;
		cartesianToPolar((*it).linearForm, currentLineMag, currentLineAng);
		double dang=min(abs(currentLineAng-projectedLineAng),(double)abs(int(currentLineAng+10)%180-int(projectedLineAng+10)%180));

		if(reference_point(0)>0 &&reference_point(0)<640 && reference_point(1)>0 && reference_point(1)<480 && dang<10){
			//Use the new method
			Vector2d start=Vector2d(it->startingPoint2d.x,it->startingPoint2d.y),end=Vector2d(it->endPoint2d.x,it->endPoint2d.y);

			dist=getLineDistance(start,end,line,reference_point, found_point)+angle_weight*dang;
			if(debug)
				cout << "candidate line : " << currentLineMag << " "<< currentLineAng << " "<< "with error : " << dist << " = " << getLineDistance(start,end,line,reference_point, found_point)<< " + " << angle_weight*dang  << endl;
		}
		else{

			//If we cannot use the spatial fair method, we use the polar one
			if (debug)
					cout << "candidate line : "<< currentLineMag << " " << currentLineAng << " ";
			//cv::cartToPolar((*it).linearForm[0], (*it).linearForm[1], mag, angle, true);
			dist = computePolarDistance(projectedtLineMag, projectedLineAng, currentLineMag, currentLineAng, angle_weight);
			if (debug)
				cout << " with err : " << dist << endl ;
		}

		if(dist <= (double)err)
		{
			nearestLines.insert( std::pair<double,Line>(dist, (*it))); //should be ordered
		}
	//		else
	//			cout << polarDist << "polar dist val" << endl;

	//		if((*it).linearForm[2]<0.0)
	//		{
	//			changeSigns((*it).linearForm);
	//		}
	//			//the euclidean distance is not a good similarity measure, instead of using a fixed threshold, we take the nearest 3 candidate lines
	//			//todo: implement a better similarity measure for lines
	//	//		if (debug)cout<<"line: "<<100000*(*it).linearForm[0]<< ", "<<100000*(*it).linearForm[1]<<", "<<100000*(*it).linearForm[2]<<endl;
	//			double unscaled_d = sqrt( pow((line[0]-(*it).linearForm[0]), 2) + pow((line[1]-(*it).linearForm[1]),2) + pow((line[2]-(*it).linearForm[2]),2));
	//	//		if (debug) cout<<"unscaled distance: "<< unscaled_d<<endl;
	//			nearestLines.insert( std::pair<double,Line>(unscaled_d, (*it))); //should be ordered

	}
//	std::multimap<double,Line>::iterator it = std::next(nearestLines.begin(),3);
//	nearestLines.erase ( it, nearestLines.end() );
//	if (debug)
//	{
//		for (auto iter = nearestLines.begin(); iter != nearestLines.end(); ++iter)
//		{
//			cout << "dist: " << (*iter).first << " line: " << 100000 * (*iter).second.linearForm[0] << ", " << 100000
//					* (*iter).second.linearForm[1] << ", " << 100000 * (*iter).second.linearForm[2] << endl;
//
//		}
//	}
	return nearestLines;
}

template <class Camera>
void GuidedMatcher<Camera>::drawLine(Vector3d projectedHomogeneousLine, cv::Mat curFrameRGB, const string &WindowName, const cv::Scalar & color, bool verbose)
{
	if (projectedHomogeneousLine[1] != (double)0.0 && projectedHomogeneousLine[0] != (double)0.0)
	{
		int y = (-1) * (projectedHomogeneousLine[2] / projectedHomogeneousLine[1]);
		int x = (-1) * (projectedHomogeneousLine[2] / projectedHomogeneousLine[0]);

		if (y > 0 & x > 0)
		{
			if(verbose)	 cout<<"P: (0,"<<y<<") Q: ("<<x<<",0)"<<endl;
			cv::line(curFrameRGB, cv::Point(0, y), cv::Point(x, 0), color, 2, 8);
		}
		else if (y < 0 & x > 0)
		{
			int newx = (-projectedHomogeneousLine[2] - 480 * projectedHomogeneousLine[1]) / projectedHomogeneousLine[0];
			if(verbose)	 cout<<"P: ("<<x<<",0) Q: ("<<newx<<",480)"<<endl;
			cv::line(curFrameRGB, cv::Point(x, 0), cv::Point(newx, 480), color, 2, 8);
		}
		else if (y > 0 & x < 0)
		{
			int newy = (-projectedHomogeneousLine[2] - 640 * projectedHomogeneousLine[0]) / projectedHomogeneousLine[1];
			if(verbose)	 cout<<"P: (0,"<<y<<") Q: (640"<<newy<<")"<<endl;
			cv::line(curFrameRGB, cv::Point(0, y), cv::Point(640, newy), color, 2, 8);
		}else
		{
			cout<<"biak negatibo, x: "<<x<<" y: "<<y<<endl;
			cout<<"projectedHomogeneousLine[0]: "<<projectedHomogeneousLine[0]<<" projectedHomogeneousLine[1]: "<<projectedHomogeneousLine[1]<<" projectedHomogeneousLine[2]:"<<projectedHomogeneousLine[2]<<endl;
		}
		cv::imshow(WindowName, curFrameRGB);

	}
}
//If there is one line with the smallest error return this Line, if there more than one because they are segments of
//the same line for instance, return the line with the smallest distance of them
template <class Camera>
Line GuidedMatcher<Camera>::findMatchedLineWithSmallestError(const std::multimap<double,std::pair<Line,double>> &matchedLines, bool debug)
{
	//todo:sometimes there are more than one lines with error=0, check that this is only happens when these are segments of the projection
	double min=100000.0;
	//std::vector<Line> matchedEqualLines;
	std::multimap<double,std::pair<Line,double>> matchedEqualLines;
	Line matchedLine;
	bool equalLines = false;
	for( auto iter = matchedLines.begin(); iter!= matchedLines.end(); iter++)
	{
		if(debug)	cout<<"dist: "<<(*iter).first<<endl;
		if(debug)   cout<<"normalizedError: "<<(*iter).second.second<<endl;
		if((*iter).second.second < min & !equalLines) //sometimes there are segments of the same line with equal errors
		{
			min = (*iter).second.second;
			matchedEqualLines.clear();
			//matchedLine = (*iter).second.first;
			matchedEqualLines.insert((*iter));

		}
		else if((*iter).second.second < min & equalLines)
		{
			matchedEqualLines.clear();
			min = (*iter).second.second;
//			matchedLine = (*iter).second.first;
			matchedEqualLines.insert((*iter));
			equalLines= false;
		}
		else if((*iter).second.second == min & !equalLines)
		{
			matchedEqualLines.insert((*iter));
//			matchedEqualLines.push_back(matchedLine);
//			matchedEqualLines.push_back((*iter).second.first);
			equalLines=true;
		}
		else if((*iter).second.second == min & equalLines)
		{
//			matchedEqualLines.push_back((*iter).second.first);
			matchedEqualLines.insert((*iter));
		}
	}
	std::multimap<double,std::pair<Line,double>>::iterator it = matchedEqualLines.begin();
	if(debug) cout<<"first elem normErr: "<<(*it).second.second<<endl;
	if(debug) cout<<"first elem dist: "<<(*it).first<<endl;
	return (*it).second.first;
}





template <class Camera>
void GuidedMatcher<Camera>::findBestConfiguration(tr1::unordered_map<int,Line> &tracked_lines,
												  std::vector<Line> &linesOnCurrentFrame,
											      tr1::unordered_map<int,std::multimap<double,std::pair<Line,double> > > & candidates ,
											      map<int,int>& best_matched,
												  double max_error=30){

	//Vector containing id of the lines having common candidates
	vector<pair<int,int> > conflicts;

	//Parameters
	double best_error=1000000000;
	//map<int,int> best_matched;

	for(auto ptr=tracked_lines.begin();ptr!=tracked_lines.end();ptr++){
		for(auto ptr2=ptr;ptr2!=tracked_lines.end();ptr2++){
			if(ptr->first!=ptr2->first){
				//First find conflicts
				findLineConflicts(ptr->first,ptr2->first,candidates.find(ptr->first)->second,candidates.find(ptr2->first)->second,conflicts);
			}
		}
	}

	//Number of possible permutations in the order of tracked lines
	int n=min(pow(2,conflicts.size()),pow(2,10));
	cout << "we have " << n << " conflicts" << endl;


	for(unsigned int k=0;k<n;k++){
		double gerror=0;
		vector<pair<int,Line> > permuted_lines;
		map<int,int> matched; //Init the matched vector right after ( candidate line id , trackedline id=-1(not matched) )
		for(auto it=linesOnCurrentFrame.begin();it!=linesOnCurrentFrame.end();it++){matched.insert(make_pair(it->global_id,-1));};
		createPermutedVector(tracked_lines,permuted_lines,k,conflicts);

		for(auto ptr=permuted_lines.begin();ptr!=permuted_lines.end();ptr++){
			auto matchedLines=candidates.find(ptr->first)->second;
			if (matchedLines.size() > 0)
				{
					auto it = matchedLines.begin();
					while(matched.find(it->second.first.global_id)->second!=-1 && it!=matchedLines.end())
						it++;
					if(it!=matchedLines.end()){
						gerror+=it->first;
						matched.find(it->second.first.global_id)->second=ptr->second.global_id;
					}
					else
						gerror+=max_error+5; //  /!\ should be discussed /!\/

				}
			else
				gerror+=max_error+5; // /!\ should be discussed /!\/
		}
		if(gerror<best_error){
			//do stuff
			best_error=gerror;
			best_matched=matched;
		}
	}



}



//@fixme: method too long
//@fixme: there is a method for accesing T_cur_from_actkey...
template <class Camera>
void GuidedMatcher<Camera>::lineMatcher(std::vector<Line> &linesOnCurrentFrame,
										tr1::unordered_map<int,Line> &tracked_lines,
										const SE3 & T_cur_from_actkey,
										int actkey_id,
										const ALIGNED<FrontendVertex>::int_hash_map & vertex_map,
										const Matrix<double,3,3> camera_matrix,
										cv::Mat *curFrameRGB,
										const vector<Vector3d> & edges,
										MonoFrontend *monoFrontednPtr,
										AddToOptimzerPtr & to_optimizer,
										std::vector<int> & localIDsofNewLinesToBeAdded,
										SE3 &T_cur_from_w,
										int frame_id)
{


	bool display=true;
	bool verbose=true;
	bool one_line_track=false;
	int line_tracked=0;
	bool color_mode=false;
	bool matchmap_is_active=true;



	//Structure containing candidate lines for each tracked line
	 tr1::unordered_map<int,std::multimap<double,std::pair<Line,double> > > candidates;

	 //Match map between tracked lines id and lines on frame id
	 map<int,int> matchedline_id;

	 //Keep track of the projection obtained
	 map<int,Vector3d> projection_map;

	 //Transform
	  SE3 T_actkey_from_w    = GET_MAP_ELEM(actkey_id, vertex_map).T_me_from_w;
	  T_cur_from_w = T_cur_from_actkey*T_actkey_from_w; //Position wo  sich die Kamera in Weltkoordinaten befindet nach dem Optischen FLuß

	  //Current camera intrinsic matrix
	  g_camera_matrix=camera_matrix;

	  //Max accepted error during matching
	  float polar_err=20;

	  //Build color map of candidate lines
	  vector<pair<int,cv::Scalar>> colormap;
	  float n=0,m=linesOnCurrentFrame.size()-1;
	  for(auto ptr=linesOnCurrentFrame.begin();ptr!=linesOnCurrentFrame.end();ptr++){
		  if(n<m/3)
			  colormap.push_back(make_pair((*ptr).global_id,cv::Scalar(floor(256.*3*n/m),255, 0)));
		  else{
			  if(n<(2*m/3))
				  colormap.push_back(make_pair((*ptr).global_id,cv::Scalar(0,floor(256.*n/m), 128)));
			  else
				  colormap.push_back(make_pair((*ptr).global_id,cv::Scalar(0,0,floor(256.*n/m))));
		  }
		  n++;}


	  //Build match map of candidate lines
	 map<int,int> matchmap;
	 for(auto ptr=linesOnCurrentFrame.begin();ptr!=linesOnCurrentFrame.end();ptr++)
	 	matchmap.insert(make_pair((*ptr).global_id,-1));


	  //Projiziere diese ins aktuelle Kamerabild, gucken ob die drin sind, falls nicht nicht weitermachen
	 //cout << T_cur_from_w.matrix() << endl;
	 Matrix<double, 3, 4> projectionsMatrix = computeProjectionMatrix(camera_matrix, T_cur_from_w.matrix() );
	 int line_num=0;
	  cout<<" size of linesOnCurrentFrame : "<<linesOnCurrentFrame.size()<<endl;
	  cout<<" size of trackedlines: "<<tracked_lines.size()<<endl;
	  if(linesOnCurrentFrame.size()>0)
	  {


	  set<int> currentLineLocalId;
	  for (auto tracked_lines_iter = tracked_lines.begin(); tracked_lines_iter != tracked_lines.end(); ++tracked_lines_iter)
	  {
		  map<int,double> extinction;
		  extinction.insert(make_pair(0,linesOnCurrentFrame.size()));//0 total number of lines
		  extinction.insert(make_pair(1,0)); //1 : candidate too far
		  extinction.insert(make_pair(2,0)); //2 : bad ssd
		  extinction.insert(make_pair(3,0)); //3 : bad size
		  extinction.insert(make_pair(4,0)); //4 : outside frame
		  extinction.insert(make_pair(5,0)); //5 : 'incorrectness' try to see how close to the acceptance limit we are

		  if(one_line_track && line_num==line_tracked && verbose)
			  cout << "Following line : " << tracked_lines_iter->second.global_id << endl;

		  cv::Point begin,end;


		  Vector3d transformedAndProjectedStartingPoint = projectionsMatrix*toHomogeneousCoordinates((*tracked_lines_iter).second.startingPoint3d);
		  Vector3d transformedAndProjectedEndingPoint = projectionsMatrix*toHomogeneousCoordinates((*tracked_lines_iter).second.endPoint3d);
		  transformedAndProjectedStartingPoint[0] = transformedAndProjectedStartingPoint[0]/transformedAndProjectedStartingPoint[2];
		  transformedAndProjectedStartingPoint[1] = transformedAndProjectedStartingPoint[1]/transformedAndProjectedStartingPoint[2];
		  transformedAndProjectedEndingPoint[0] = transformedAndProjectedEndingPoint[0]/transformedAndProjectedEndingPoint[2];
		  transformedAndProjectedEndingPoint[1] = transformedAndProjectedEndingPoint[1]/transformedAndProjectedEndingPoint[2];

		  //Project the tracked line in the image
		  begin=cv::Point(transformedAndProjectedStartingPoint[0], transformedAndProjectedStartingPoint[1]);
		  end=cv::Point(transformedAndProjectedEndingPoint[0], transformedAndProjectedEndingPoint[1]);


		  Vector3d projectedHomogeneousLine2;
		  //Display all the tracked lines in purple
		 //cv::line( *curFrameRGB, cv::Point(transformedAndProjectedStartingPoint[0], transformedAndProjectedStartingPoint[1]), cv::Point(transformedAndProjectedEndingPoint[0], transformedAndProjectedEndingPoint[1]), cv::Scalar(200,50,100), 8, 8 );


		 projectedHomogeneousLine2 = computeLineProjectionMatrix2(projectionsMatrix, toPlueckerMatrix((*tracked_lines_iter).second.optimizedPluckerLines));
		 projectedHomogeneousLine2.normalize();


		 if(one_line_track && line_num==line_tracked && verbose)
			 cout << "2d form : " << projectedHomogeneousLine2 << " <> " << tracked_lines_iter->second.pluckerLinesObservation << " <> " << tracked_lines_iter->second.linearForm << endl;



		 projection_map.insert(make_pair(tracked_lines_iter->second.global_id,projectedHomogeneousLine2));

		Vector2d reference_point;
		projectReferencePoint(projectedHomogeneousLine2,tracked_lines_iter->second.rtheta,reference_point);

		 if (projectedHomogeneousLine2[2] < 0.0)
		 {
			changeSigns(projectedHomogeneousLine2);
		 }

		 //Transform to euclidean space ax + by + c = 0
		 //if x=0 -> by + c = 0   -> y=-c/b
		 //if y=0->  ax + c = 0   -> x=-c/a

		 Vector2d intersec1;
		 Vector2d intersec2;
		 std::multimap<double,std::pair<Line,double>> matchedLines;// key is distance and the Line's pair double the normalizedError
		 bool matched = false;
		 if(lineIsInsideFrame(projectedHomogeneousLine2, edges, intersec1, intersec2, false))
		 {

			 //Define the error for the findNearest func


			 std::multimap<double,Line> nearestLines;
			 if(tracked_lines_iter->second.type==0)
				 polar_err=20;
			 else
				 polar_err=15;

			 if(one_line_track && line_num==line_tracked && verbose)
				 nearestLines = findNearestLinesOnCurrentFrameOrderedByDistance(projectedHomogeneousLine2, linesOnCurrentFrame,polar_err,reference_point, true);
			 else
				 nearestLines = findNearestLinesOnCurrentFrameOrderedByDistance(projectedHomogeneousLine2, linesOnCurrentFrame,polar_err,reference_point, false);

//			 if(one_line_track &&line_num==line_tracked){
//				 cout << "tracked line plucker : " << (*tracked_lines_iter).second.optimizedPluckerLines[0] << " "  << (*tracked_lines_iter).second.optimizedPluckerLines[1] << " " << (*tracked_lines_iter).second.optimizedPluckerLines[2] << " " << (*tracked_lines_iter).second.optimizedPluckerLines[3] << " " << (*tracked_lines_iter).second.optimizedPluckerLines[4] << " " << (*tracked_lines_iter).second.optimizedPluckerLines[5] << endl;
//				 for(auto ptr=nearestLines.begin();ptr!=nearestLines.end();ptr++){
//					 cout << "error : " << (*ptr).first << " pluckercoord : " << (*ptr).second.optimizedPluckerLines[0] << " " << (*ptr).second.optimizedPluckerLines[1] << " " << (*ptr).second.optimizedPluckerLines[2] << " " << (*ptr).second.optimizedPluckerLines[3] << " " << (*ptr).second.optimizedPluckerLines[4] << " " << (*ptr).second.optimizedPluckerLines[5] << endl;
//				 }
//			 }


			 if(one_line_track){
				 if(line_num==line_tracked){
					 for(auto lineptr=nearestLines.begin();lineptr!=nearestLines.end();++lineptr){
						 cv::line( *curFrameRGB, (*lineptr).second.startingPoint2d,(*lineptr).second.endPoint2d, cv::Scalar(0,255-floor(255./polar_err*((*lineptr).first)),floor(255./polar_err*((*lineptr).first))), 2, 7 );
					 }
				 }
			 }

			 if(one_line_track && line_num==line_tracked){
				 Point p=Point(reference_point(0),reference_point(1));
				 int ididi=tracked_lines_iter->second.global_id;
				 cross(*curFrameRGB,p,Scalar((ididi)%255,(ididi+100)%255,(ididi+200)%255));
			 }

			 //Numbers of lines too far away from the tracked line
			 extinction.find(1)->second=linesOnCurrentFrame.size()-nearestLines.size();

			 for( auto nearest_lines_iter = nearestLines.begin(); nearest_lines_iter!= nearestLines.end(); ++nearest_lines_iter)
			 {
				double normalizedError;
				if((*nearest_lines_iter).second.type!=tracked_lines_iter->second.type || (*nearest_lines_iter).second.type==0 ){ //The line is rgb

					 //if (matchTwoLines((*tracked_lines_iter).second.descriptor,(*nearest_lines_iter).second.descriptor, normalizedError, true))
					 if (  (!matchmap_is_active ||(matchmap.find((*nearest_lines_iter).second.global_id))->second==-1)  && matchTwoLinesSSD((*tracked_lines_iter).second.descriptor,(*nearest_lines_iter).second.descriptor, normalizedError, false,30000)  )
					 {
						  matchedLines.insert(std::pair<double,std::pair<Line,double>>((*nearest_lines_iter).first,std::pair<Line,double>((*nearest_lines_iter).second, normalizedError)));
						  if((!one_line_track || line_num==line_tracked)&&verbose)
							  cout << "line kept" <<endl;
					 }
					 else
					 {
						 int maxSize = max((*tracked_lines_iter).second.descriptor.size(), (*nearest_lines_iter).second.descriptor.size());
						 int minSize = min((*tracked_lines_iter).second.descriptor.size(), (*nearest_lines_iter).second.descriptor.size());

						 if(!((maxSize/ minSize)>2.0) && matchmap_is_active && (!matchmap_is_active || (matchmap.find((*nearest_lines_iter).second.global_id))->second==-1))
						 {
							 //Count rejections because of SSD
							extinction.find(2)->second+=1;
							double inco=0.5*((1-nearest_lines_iter->first/polar_err)+30000./normalizedError);
							if(extinction.find(5)->second<inco)
								extinction.find(5)->second=inco;
							 //Display in red the possible match
							 //cv::line( *curFrameRGB, (*nearest_lines_iter).second.startingPoint2d, (*nearest_lines_iter).second.endPoint2d, cv::Scalar(0,0,255), 2, 8 );
							 if((!one_line_track || line_num==line_tracked) && verbose)
								 cout << "SSD are different ! " << nearest_lines_iter->first << " " << normalizedError << endl;
						 }
						 else{
							 if(!matchmap_is_active || ((matchmap.find((*nearest_lines_iter).second.global_id))->second==-1)){
								 //Count rejections because of bad size
								extinction.find(3)->second+=1;

								 //cv::line( *curFrameRGB, (*nearest_lines_iter).second.startingPoint2d, (*nearest_lines_iter).second.endPoint2d, cv::Scalar(0,0,0), 2, 8 );
								 if((!one_line_track || line_num==line_tracked) && verbose )
									 cout << "size too different !" << endl;
								}

							 else{
								 if((!one_line_track || line_num==line_tracked) && verbose )
									 cout << "already assigned to another line !" << endl;
							 }
						 }
					 }
				}
				else{ // The line is an oe
					if( (!matchmap_is_active ||(matchmap.find((*nearest_lines_iter).second.global_id))->second==-1) && matchTwoLinesDD(nearest_lines_iter->second.d_descriptor,tracked_lines_iter->second.d_descriptor,normalizedError,true,1.4)){
						matchedLines.insert(std::pair<double,std::pair<Line,double>>((*nearest_lines_iter).first,std::pair<Line,double>((*nearest_lines_iter).second, normalizedError)));
					}
					else{
						if((!one_line_track || line_num==line_tracked) && verbose )
							cout << "Bad DD descriptor !" << endl;
					}
				}

			 }
			 candidates.insert(make_pair(tracked_lines_iter->second.global_id,matchedLines));
			 if (matchedLines.size() > 0)
			{

				 (*tracked_lines_iter).second.consecutive_frame++;
				 if(!one_line_track && verbose)
					 cout << "match" << endl;
				Line matchedLine = findMatchedLineWithSmallestError(matchedLines, false);
				matched = true;
				//cout << (*tracked_lines_iter).second.global_id << "tracked line id <<>> matched line id" << matchedLine.global_id << endl;
			//	cout<<"matched!"<<endl;
				//cout<<"matched line linear form "<<(*iter).linearForm<<endl;
				//delete matched line from LinesOnCurrentFrame
				//@fixme: use list on linesOnCurrentFrame instead of vector for more efficiency when doing random deletes
				int pos = 0;
				for (auto linesOnCurrentFrame_iter = linesOnCurrentFrame.begin(); linesOnCurrentFrame_iter != linesOnCurrentFrame.end(); ++linesOnCurrentFrame_iter)
				{
					if (equalVectors3d((*linesOnCurrentFrame_iter).linearForm, matchedLine.linearForm))
					{
						currentLineLocalId.insert(pos);
						break;
					}
					++pos;
				}



				//Remove the best match from the available lines by tagging it with the tracked line id
				(matchmap.find(matchedLine.global_id))->second=(*tracked_lines_iter).second.global_id;

				//Assign the same color between the tracked line and his corresponding match
				cv::Scalar color;
				if(color_mode){
					for(unsigned int i=0;i<linesOnCurrentFrame.size();i++){
						if(colormap[i].first==matchedLine.global_id)
							color=colormap[i].second;
					}
					cv::line( *curFrameRGB, matchedLine.startingPoint2d,matchedLine.endPoint2d, color, 2, 7 );
				}

				//Display of the projected tracked line
				if(one_line_track){
					 if(line_num==line_tracked){
						 if(tracked_lines_iter->second.type>0)
						 //cv::line( *curFrameRGB, begin, end,cv::Scalar(255,255,255), 2, 8);
							 drawLine(projectedHomogeneousLine2, *curFrameRGB, "matches", cv::Scalar(255, 255, 255), false);
						 else
							 drawLine(projectedHomogeneousLine2, *curFrameRGB, "matches", cv::Scalar(0,0, 0), false);
					 }
				}
				else{
					if(color_mode)
						//cv::line( *curFrameRGB, begin, end,color, 2, 8);
						drawLine(projectedHomogeneousLine2, *curFrameRGB, "matches", color, false);
					else{
						//cv::line( *curFrameRGB, begin, end,cv::Scalar(255,255,255), 2, 8);
						if(tracked_lines_iter->second.type>0){
							if(tracked_lines_iter->second.global_id==1){
								drawLine(projectedHomogeneousLine2, *curFrameRGB, "matches", cv::Scalar(128, 255, 128), false);
							}
							else
								drawLine(projectedHomogeneousLine2, *curFrameRGB, "matches", cv::Scalar(255, 255, 255), false);
						}
						else
							drawLine(projectedHomogeneousLine2, *curFrameRGB, "matches", cv::Scalar(0, 0, 0), false);
					}
				}


				//Update od the line coord
				ADD_TO_MAP_LINE((*tracked_lines_iter).first, (*tracked_lines_iter).second, &tracked_lines); //reset counter
				//(*tracked_lines_iter).second.linearForm=matchedLine.linearForm;

				//Update the observation depending on the mode (if the line is active it can receive the normal updates)
				//if(!mono_mode || (*tracked_lines_iter).second.active){
					(*tracked_lines_iter).second.startingPoint2d=matchedLine.startingPoint2d;
					(*tracked_lines_iter).second.endPoint2d=matchedLine.endPoint2d;
					(*tracked_lines_iter).second.pluckerLinesObservation = matchedLine.pluckerLinesObservation;
					(*tracked_lines_iter).second.T_frame_w=matchedLine.T_frame_w;
					//(*tracked_lines_iter).second.anchor_id=matchedLine.anchor_id;
					//}
				//else{

					//Matrix<double, 3, 4> projectionsMatrix = computeProjectionMatrix(camera_matrix, T_cur_from_w.matrix() );

					//cout << computeLineProjectionMatrix(projectionsMatrix) * (*tracked_lines_iter).second.optimizedPluckerLines.matrix() << " test val proj " << matchedLine.linearForm << endl;
					//Add the 2d observation to the tracked line to prepare its 3D estimation



					if((*tracked_lines_iter).second.global_id==1)
						cout << (*tracked_lines_iter).second.consecutive_frame << " consecutive frame for 1" << endl;






				//Update the SSD once a while
				if(((*tracked_lines_iter).second.consecutive_frame%10)==0){ //Could switch with a modulo to get hw many consec frames were tracked
					if((one_line_track && line_num==line_tracked)&&verbose)
						cout << "UPDATE SSD" << endl;
					(*tracked_lines_iter).second.descriptor=matchedLine.descriptor;
					if((*tracked_lines_iter).second.type!=0)
						(*tracked_lines_iter).second.d_descriptor=matchedLine.d_descriptor;
				}


				if(one_line_track && line_num==line_tracked){
				//if((*tracked_lines_iter).second.global_id==1){
				  cout << "Tracked line id : " << (*tracked_lines_iter).second.global_id << endl;
				  auto ptr=tracked_lines_iter;
				  pluckerToFile((*tracked_lines_iter).second.optimizedPluckerLines,(*tracked_lines_iter).second.global_id);
				  cout  << (*ptr).first << " pluckercoord opt : " << (*ptr).second.optimizedPluckerLines[0] << " " << (*ptr).second.optimizedPluckerLines[1] << " " << (*ptr).second.optimizedPluckerLines[2] << " " << (*ptr).second.optimizedPluckerLines[3] << " " << (*ptr).second.optimizedPluckerLines[4] << " " << (*ptr).second.optimizedPluckerLines[5] << endl;
				  }

			}


			 else
			 {
				 if(display){
					 if(tracked_lines_iter->second.type>0)
						 drawLine(projectedHomogeneousLine2, *curFrameRGB, "matches", cv::Scalar(0, 50, 200), false);
					 else
						 drawLine(projectedHomogeneousLine2, *curFrameRGB, "matches", cv::Scalar(90,90, 150), false);
					 //cv::line( *curFrameRGB, begin, end, cv::Scalar(0,200,55), 2, 8 );
				 }

				 if((!one_line_track || line_num==line_tracked)&&verbose)
					 cout << "no matched line" << endl;
			 }
		 }
		 else
		 {
			 //Count rejections because of wrong coord
			 extinction.find(4)->second+=1;
			 cout<<"line outside frame"<<endl;
			 cout << projectedHomogeneousLine2(0) << " " << projectedHomogeneousLine2(1) << " " << projectedHomogeneousLine2(2) << " " <<endl;
		 }
		 // /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\/////
		 if(!matched)
		 {
		 // line is not inside frame, or has not been found -> decrement counter or erase it?
			 //currently only erase element if it hasn't been found 3 times in a row, to account for hough not always finding
			 //the same lines and/or the sesnor not giving 3D information
	     //cout<<"deleting tracked_line with id: "<<(*tracked_lines_iter).first<<" because no match was found"<<endl;
		if((*tracked_lines_iter).second.count<=0)
			  {dumpToFile(std::to_string(tracked_lines_iter->second.global_id),extinction.find(0)->second,extinction.find(1)->second,extinction.find(2)->second,extinction.find(3)->second,extinction.find(4)->second,extinction.find(5)->second,1000000000000,"/home/rmb-am/Slam_datafiles/extinctionFile.txt");
			  //dumpToFile(std::to_string(tracked_lines_iter->second.global_id),tracked_lines_iter->second.consecutive_frame,tracked_lines_iter->second.type,tracked_lines_iter->second.anchor_id,actkey_id,0,0,0,"/home/rmb-am/Slam_datafiles/line_duration.txt");
	    }
		DEL_FROM_MAP_LINE((*tracked_lines_iter).first, (*tracked_lines_iter).second, &tracked_lines);
		 if(one_line_track && line_num==line_tracked && (*tracked_lines_iter).second.count==0)
			 cout << "ERASING LINE !!!!!" << endl;

		 }
		 // /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\  /!\////
		 ++line_num;
	  }

//	  cout << "start" << endl;
//	  findBestConfiguration(tracked_lines,linesOnCurrentFrame,candidates,matchedline_id,polar_err);
//	  cout << "middle" << endl;
//	  post_matching(tracked_lines,linesOnCurrentFrame,matchedline_id,currentLineLocalId, projection_map,color_mode,colormap,curFrameRGB,KF_NUMBER,line_num,line_tracked,one_line_track,verbose,projectionsMatrix,T_actkey_from_w);
//	  cout << "end" << endl;




	  std::vector<int>::iterator it;
	  std::vector<int> linesIdOnCurrentFrame(linesOnCurrentFrame.size());
	  //std::iota fills the range [first, last) with sequentially increasing values, starting with value and repetitively evaluating ++value.
	  std::iota(linesIdOnCurrentFrame.begin(), linesIdOnCurrentFrame.end(), 0);

	  //now see which lines on the current frame have not been matched and thus could be added to tracked_lines if a new keyframe is added
	  std::set_difference (linesIdOnCurrentFrame.begin(), linesIdOnCurrentFrame.end(), currentLineLocalId.begin(), currentLineLocalId.end(),  std::inserter(localIDsofNewLinesToBeAdded, localIDsofNewLinesToBeAdded.end()));
	   //new lines are only added when a new keyframe is added

	  if(matchmap_is_active){
		  for(auto ptr=matchmap.begin();ptr!=matchmap.end();ptr++){
				  if((*ptr).second==-1){
					  for(auto it=linesOnCurrentFrame.begin();it!=linesOnCurrentFrame.end();it++){
						  if((*it).global_id==(*ptr).first){
							  if(display){
								  if(it->type>0)
									  cv::line( *curFrameRGB, (*it).startingPoint2d, (*it).endPoint2d, cv::Scalar(0,0,255), 2, 8 );
								  else
									  cv::line( *curFrameRGB, (*it).startingPoint2d, (*it).endPoint2d, cv::Scalar(0,100,255), 2, 8 );
							  }

						  }
					  }
				  }
		  	  }
	  }

	 	to_optimizer->tracked_lines=tracked_lines;
	 	//cv::imshow("matches", *curFrameRGB);
		// cv::waitKey(1);
	  }
	  else
	  {
		  cout << "no line on frame" << endl;
	  }




}

template <class Camera>
bool GuidedMatcher<Camera>::matchTwoLines(std::vector<int> descriptor1, std::vector<int> descriptor2, double &normalizedError, bool debug)
{
	int maxSize = max(descriptor1.size(), descriptor2.size());
	int minSize = min(descriptor1.size(), descriptor2.size());
	if(debug)cout<<"max size: "<<maxSize<<endl;
	if(debug)cout<<"min size: "<<minSize<<endl;

	if(debug && (maxSize/ minSize)>2.0)
	{
		cout<<"too big difference in size"<<endl;
		return false;
	}

	int vectorBound = floor((min(descriptor1.size(), descriptor2.size()))*0.8);
	int j = (min(descriptor1.size(), descriptor2.size())) - vectorBound;
	int jOld = j;
	int twentyPercent = jOld;
	int i = 0;
	int iNew = 0;
	std::vector<int> hammingErrors;
	hammingErrors.reserve(max(descriptor1.size(), descriptor2.size()));
	int errorCount = 0;
	if(debug) cout<<"vectorBound: "<<vectorBound<<" i: "<<i<<" j: "<<j<<endl;
	int bound = (max(descriptor1.size(), descriptor2.size())+twentyPercent);
	if(debug)cout<<"bound: "<<bound<<endl;

	while (i<vectorBound && vectorBound!= bound && twentyPercent!=-1)
	{
		//if(debug)cout<<"d1["<<i<<"]: "<<descriptor1[i]<<" d2["<<j<<"]: "<<descriptor2[j]<<endl;
		if(descriptor1[i]!=descriptor2[j])
		{
			++errorCount;
		}
		if(i==vectorBound-1)
		{
			if(jOld!=0)
			{
				j = jOld-1;
				i = 0;
				jOld = j;
				++vectorBound;
				//if(debug)cout<<"new VectorBound: "<<vectorBound<<endl;
				//if(debug)cout<<"j!=0     new j:"<<j<<endl;
			}
			else
			{
				i=iNew+1;
				iNew++;
				j=0;
				if (vectorBound!=max(descriptor1.size(), descriptor2.size()))
				{
				++vectorBound;
				//if(debug)cout<<"new VectorBound: "<<vectorBound<<endl;
				}
				else
				{
					//cout<<"twentypercent: "<<twentyPercent<<endl;
					twentyPercent = twentyPercent - 1;
					//cout<<"twentypercent2: "<<twentyPercent<<endl;
				}
				//if(debug)cout<<"new i: "<<i<<endl;
			}
			hammingErrors.push_back(errorCount);
			errorCount = 0;

		}
		else
		{
			 ++i;
			 ++j;
		}
	}
	auto it = std::min_element(hammingErrors.begin(),hammingErrors.end());
	if(debug) cout<<"unnormalized min error "<< *it<<endl;
	if(debug) cout<<"hammingsErrors.size"<<(double)hammingErrors.size()<<endl;
	//normalizedError = (double)*it/(double)min(descriptor1.size(), descriptor2.size());
	normalizedError = (double)*it/(double)hammingErrors.size();

//	for (std::vector<int>::size_type index = 0; index != hammingErrors.size(); index++)
//	{
//		double normalized1 = (double) hammingErrors[index] / (double) min(descriptor1.size(), descriptor2.size());
//		cout << "hammingErrors[" << index << "]: " << hammingErrors[index] << " normalized: " << normalized1 << endl;
//	}
	if(debug)	cout<<"normalizedError: "<<normalizedError<<endl;
	if(normalizedError<0.25)
	{
		//cout<<"same line!"<<endl;
		return true;
	}
	else
	{
		return false;
	}
}


template <class Camera>
bool GuidedMatcher<Camera>::matchTwoLinesDD(std::vector<float> descriptor1, std::vector<float> descriptor2, double &normalizedError, bool debug,double error_thr=1){

	normalizedError=0;

	if(descriptor1.size() != descriptor2.size()){
		cout << "ERROR : different descriptor size !!!!!!" << endl;
		return false;
	}

	normalizedError = 2 ;

	for(int i = 0 ; i < descriptor1.size() ; i++){
		normalizedError-=descriptor1[i]*descriptor2[i]; //The closer the histograms are, the smallest the error is.
	}

	if(debug) cout << "error : " << normalizedError << endl;

	if(normalizedError>error_thr)
		return false;
	else
		return true;

}


template <class Camera>
bool GuidedMatcher<Camera>::matchTwoLinesSSD(std::vector<int> descriptor1, std::vector<int> descriptor2, double &normalizedError, bool debug,double error_thr=21000)
{
	int maxSize = max(descriptor1.size(), descriptor2.size());
	int minSize = min(descriptor1.size(), descriptor2.size());
	if(debug)cout<<"max size: "<<maxSize<<endl;
	if(debug)cout<<"min size: "<<minSize<<endl;
	if(debug && (maxSize/ minSize)>2.0)
	{
		cout<<"too big difference in size"<<endl;
		return false;
	}
	int vectorBound = floor((min(descriptor1.size(), descriptor2.size()))*0.8);
	int j = (min(descriptor1.size(), descriptor2.size())) - vectorBound;
	int jOld = j;
	int twentyPercent = jOld;
	int i = 0;
	int iNew = 0;
	std::vector<double> hammingErrors;
	hammingErrors.reserve(max(descriptor1.size(), descriptor2.size()));
	double errorCount = 0;
//	if(debug) cout<<"vectorBound: "<<vectorBound<<" i: "<<i<<" j: "<<j<<endl;
	int bound = (max(descriptor1.size(), descriptor2.size())+twentyPercent);
//	if(debug)cout<<"bound: "<<bound<<endl;
//	for(auto it=descriptor1.begin(); it!=descriptor1.end();++it)
//	{
//		cout<<*it<<endl;
//	}
//	cout<<"-------------"<<endl;
//	for(auto it=descriptor2.begin(); it!=descriptor2.end();++it)
//	{
//		cout<<*it<<endl;
//	}
	if(descriptor1.size()<descriptor2.size())
	{
		swap(descriptor1,descriptor2);
		assert(descriptor1.size()>=descriptor2.size());
	}
	while (i<vectorBound && vectorBound!= bound && twentyPercent!=-1)
	{
		//if(debug)cout<<"d1["<<i<<"]: "<<descriptor1[i]<<" d2["<<j<<"]: "<<descriptor2[j]<<endl;
//		if(descriptor1[i]!=descriptor2[j])
//		{
//			++errorCount;
//		}
//		cout<<"descriptor1[i]: "<<descriptor1[i]<<" descriptor2[j]: "<<descriptor2[j]<<endl;
		double squared_diff = pow((descriptor1[i]-descriptor2[j]),2);
//		cout<<"squared_diff"<<squared_diff<<endl;
		errorCount = errorCount + squared_diff;
		if(i==vectorBound-1)
		{
			if(jOld!=0)
			{
				j = jOld-1;
				i = 0;
				jOld = j;
				++vectorBound;
//				if(debug)cout<<"new VectorBound: "<<vectorBound<<endl;
//				if(debug)cout<<"j!=0     new j:"<<j<<endl;
			}
			else
			{
				i=iNew+1;
				iNew++;
				j=0;
				if (vectorBound!=max(descriptor1.size(), descriptor2.size()))
				{
				++vectorBound;
				//if(debug)cout<<"new VectorBound: "<<vectorBound<<endl;
				}
				else
				{
					//cout<<"twentypercent: "<<twentyPercent<<endl;
					twentyPercent = twentyPercent - 1;
					//cout<<"twentypercent2: "<<twentyPercent<<endl;
				}
				//if(debug)cout<<"new i: "<<i<<endl;
			}
			hammingErrors.push_back(errorCount);
			errorCount = 0;

		}
		else
		{
			 ++i;
			 ++j;
		}
	}
	auto it = std::min_element(hammingErrors.begin(),hammingErrors.end());
//	if(debug) cout<<"unnormalized min error "<< *it<<endl;
//	if(debug) cout<<"hammingsErrors.size "<<(double)hammingErrors.size()<<endl;
	//normalizedError = (double)*it/(double)min(descriptor1.size(), descriptor2.size());
	normalizedError = (double)*it/(double)hammingErrors.size();

//	for (std::vector<int>::size_type index = 0; index != hammingErrors.size(); index++)
//	{
//		double normalized1 = (double) hammingErrors[index] / (double) min(descriptor1.size(), descriptor2.size());
//		cout << "hammingErrors[" << index << "]: " << hammingErrors[index] << " normalized: " << normalized1 << endl;
//	}
	if(debug)	cout<<"normalizedError: "<<normalizedError<<endl;
	if(normalizedError<(double)error_thr)
	{
		//cout<<"same line!"<<endl;
		return true;
	}
	else
	{
		return false;
	}
}

template <class Camera>
void GuidedMatcher<Camera>
::match(const tr1::unordered_map<int,Frame> & keyframe_map,
        const SE3 & T_cur_from_actkey,
        const Frame & cur_frame,
        const ALIGNED<QuadTree<int> >::vector & feature_tree,
        const typename ALIGNED<Camera>::vector & cam_vec,
        int actkey_id,
        const ALIGNED<FrontendVertex>::int_hash_map & vertex_map,
        const list< tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > >  & ap_map,
        int SEARCHRADIUS,
        int thr_mean,
        int thr_std,
        TrackData<Camera::obs_dim> * track_data,
        SE3 &T_anchorkey_from_w)
{
  SE3 T_actkey_from_w = GET_MAP_ELEM(actkey_id, vertex_map).T_me_from_w;
  SE3 T_w_from_actkey = T_actkey_from_w.inverse();

  SE3 T_cur_from_w = T_cur_from_actkey*T_actkey_from_w; //Position wo  sich die Kamera in Weltkoordinaten befindet nach dem Optischen FLuß

  for (typename ALIGNED<tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > >::list::const_iterator it = ap_map.begin(); it!=ap_map.end();++it)
  {
    const tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > & ap  = *it;

    Vector2d uv_pyr; //2D Koordinate in jetzigen Bild von den Abfrage Punkt ap, den es schon vorher gab. Die neue Koordinate, nachdem es reinprojekziert wurde
    //SE3 T_anchorkey_from_w;
    bool is_prediction_valid = computePrediction(T_cur_from_w,
                                                 cam_vec,
                                                 ap,
                                                 vertex_map,
                                                 &uv_pyr,
                                                 &T_anchorkey_from_w); //Wird der Punkt ins jetzige Kamerabild reinprojekziert? Falls nein, nicht weitermachen
    if (is_prediction_valid==false)
      continue;

    ALIGNED<QuadTreeElement<int> >::list candidates;
    Vector2i uvi_pyr = uv_pyr.cast<int>();
    int DOUBLE_SEARCHRADIUS = SEARCHRADIUS*2+1;
    feature_tree.at(ap->anchor_level).query(
          Rectangle(uvi_pyr[0]-SEARCHRADIUS, uvi_pyr[1]-SEARCHRADIUS,
                    DOUBLE_SEARCHRADIUS, DOUBLE_SEARCHRADIUS), &candidates);
    //Welche Punkte sind in dem Bereich in dem ich den Punkt erwarte

    const cv::Mat & kf = GET_MAP_ELEM(ap->anchor_id, keyframe_map).pyr.at(ap->anchor_level);

//    Homography homo(T_cur_from_w*T_anchorkey_from_w.inverse());
//    cv::Mat key_patch_with_border1
//        = warpPatchProjective(kf, homo, ap->xyz_anchor, ap->normal_anchor,
//                              ap->anchor_obs_pyr.head(2),
//                              cam_vec[ap->anchor_level], HALFBOX_SIZE+1);

    cv::Mat key_patch_with_border
        =   warpAffinve(kf, T_cur_from_w*T_anchorkey_from_w.inverse(),
                        ap->xyz_anchor[2], ap->anchor_obs_pyr.head(2),
                        cam_vec[ap->anchor_level], HALFBOX_SIZE+1);
    cv::Mat key_patch
        = key_patch_with_border(cv::Rect(cv::Point(1,1),
                                         cv::Size(BOX_SIZE, BOX_SIZE)));


    int pixel_sum = 0;
    int pixel_sum_square = 0;

    cv::Mat data_wrap(8,8, CV_8U, &(KEY_PATCH[0]));
    key_patch.copyTo(data_wrap);

    computePatchScores(&pixel_sum, &pixel_sum_square);

    if (pixel_sum*pixel_sum-pixel_sum_square <(int)(thr_std*thr_std*BOX_SIZE*BOX_SIZE))
      continue;


    MatchData match_data(thr_mean*thr_mean*BOX_SIZE*BOX_SIZE);
    matchCandidates(candidates, cur_frame, cam_vec,
                    pixel_sum, pixel_sum_square, ap->anchor_level,
                    &match_data);
    SE3 T_anchorkey_from_actkey = T_anchorkey_from_w*T_w_from_actkey;
    Vector3d xyz_actkey = T_anchorkey_from_actkey.inverse()*ap->xyz_anchor;
    returnBestMatch(key_patch_with_border, cur_frame, match_data,
                    xyz_actkey, ap, track_data);
  }
}

template <class Camera>
void GuidedMatcher<Camera>
::match(const tr1::unordered_map<int,Frame> & keyframe_map,
        const SE3 & T_cur_from_actkey,
        const Frame & cur_frame,
        const ALIGNED<QuadTree<int> >::vector & feature_tree,
        const typename ALIGNED<Camera>::vector & cam_vec,
        int actkey_id,
        const ALIGNED<FrontendVertex>::int_hash_map & vertex_map,
        const list< tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > >  & ap_map,
        int SEARCHRADIUS,
        int thr_mean,
        int thr_std,
        TrackData<Camera::obs_dim> * track_data)
{
  SE3 T_actkey_from_w
      = GET_MAP_ELEM(actkey_id, vertex_map).T_me_from_w;
  SE3 T_w_from_actkey = T_actkey_from_w.inverse();

  SE3 T_cur_from_w = T_cur_from_actkey*T_actkey_from_w; //Position wo  sich die Kamera in Weltkoordinaten befindet nach dem Optischen FLuß

  for (typename ALIGNED<tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > >
       ::list::const_iterator it = ap_map.begin(); it!=ap_map.end();++it)
  {
    const tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > & ap  = *it;

    Vector2d uv_pyr; //2D Koordinate in jetzigen Bild von den Abfrage Punkt ap, den es schon vorher gab. Die neue Koordinate, nachdem es reinprojekziert wurde
    SE3 T_anchorkey_from_w;
    bool is_prediction_valid = computePrediction(T_cur_from_w,
                                                 cam_vec,
                                                 ap,
                                                 vertex_map,
                                                 &uv_pyr,
                                                 &T_anchorkey_from_w); //Wird der Punkt ins jetzige Kamerabild reinprojekziert? Falls nein, nicht weitermachen
    if (is_prediction_valid==false)
      continue;

    ALIGNED<QuadTreeElement<int> >::list candidates;
    Vector2i uvi_pyr = uv_pyr.cast<int>();
    int DOUBLE_SEARCHRADIUS = SEARCHRADIUS*2+1;
    feature_tree.at(ap->anchor_level).query(
          Rectangle(uvi_pyr[0]-SEARCHRADIUS, uvi_pyr[1]-SEARCHRADIUS,
                    DOUBLE_SEARCHRADIUS, DOUBLE_SEARCHRADIUS),
            &candidates); //Welche Punkte sind in dem Bereich in dem ich den Punkt erwarte

    const cv::Mat & kf
        = GET_MAP_ELEM(ap->anchor_id, keyframe_map).pyr.at(ap->anchor_level);

//    Homography homo(T_cur_from_w*T_anchorkey_from_w.inverse());
//    cv::Mat key_patch_with_border1
//        = warpPatchProjective(kf, homo, ap->xyz_anchor, ap->normal_anchor,
//                              ap->anchor_obs_pyr.head(2),
//                              cam_vec[ap->anchor_level], HALFBOX_SIZE+1);

    cv::Mat key_patch_with_border
        =   warpAffinve(kf, T_cur_from_w*T_anchorkey_from_w.inverse(),
                        ap->xyz_anchor[2], ap->anchor_obs_pyr.head(2),
                        cam_vec[ap->anchor_level], HALFBOX_SIZE+1);
    cv::Mat key_patch
        = key_patch_with_border(cv::Rect(cv::Point(1,1),
                                         cv::Size(BOX_SIZE, BOX_SIZE)));


    int pixel_sum = 0;
    int pixel_sum_square = 0;

    cv::Mat data_wrap(8,8, CV_8U, &(KEY_PATCH[0]));
    key_patch.copyTo(data_wrap);

    computePatchScores(&pixel_sum, &pixel_sum_square);

    if (pixel_sum*pixel_sum-pixel_sum_square <(int)(thr_std*thr_std*BOX_SIZE*BOX_SIZE))
    	continue;


    MatchData match_data(thr_mean*thr_mean*BOX_SIZE*BOX_SIZE);
    matchCandidates(candidates, cur_frame, cam_vec,
                    pixel_sum, pixel_sum_square, ap->anchor_level,
                    &match_data);
    SE3 T_anchorkey_from_actkey = T_anchorkey_from_w*T_w_from_actkey;
    Vector3d xyz_actkey = T_anchorkey_from_actkey.inverse()*ap->xyz_anchor;
    returnBestMatch(key_patch_with_border, cur_frame, match_data,xyz_actkey, ap, track_data);
  }
}
//TODO:
// - test affine wraper (e.g. for large in-plane rotations)
// - make it faster by precomputing relevant data
template <class Camera>
cv::Mat GuidedMatcher<Camera>
::warpAffinve                (const cv::Mat & frame,
                              const SE3 & T_c2_from_c1,
                              double depth,
                              const Vector2d & key_uv,
                              const Camera & cam,
                              int halfpatch_size)
{
  Vector2d f = cam.map(project2d(T_c2_from_c1*(depth*unproject2d(cam.unmap(key_uv)))));
  Vector2d f_pu = cam.map(project2d(T_c2_from_c1*(depth*unproject2d(cam.unmap(key_uv+Vector2d(1,0))))));
  Vector2d f_pv = cam.map(project2d(T_c2_from_c1*(depth*unproject2d(cam.unmap(key_uv+Vector2d(0,1))))));
  Matrix2d A;
  A.row(0) = f_pu - f; A.row(1) = f_pv - f;
  Matrix2d inv_A = A.inverse();

  int patch_size = halfpatch_size*2 ;
  cv::Mat ap_patch(patch_size,patch_size,CV_8UC1);

  for (int ix=0; ix<patch_size; ix++)
  {
    for (int iy=0; iy<patch_size; iy++)
    {
      Vector2d idx(ix-halfpatch_size,
                   iy-halfpatch_size);
      Vector2d r = inv_A*idx + key_uv;

      double x = floor(r[0]);
      double y = floor(r[1]);

      uint8_t val;
      if (x<0 || y<0 || x+1>=cam.width() || y+1>=cam.height())
        val = 0;
      else
      {
        double subpix_x = r[0]-x;
        double subpix_y = r[1]-y;
        double wx0 = 1-subpix_x;
        double wx1 =  subpix_x;
        double wy0 = 1-subpix_y;
        double wy1 =  subpix_y;

        double val00 = (frame).at<uint8_t>(y,x);
        double val01 = (frame).at<uint8_t>(y+1,x);
        double val10 = (frame).at<uint8_t>(y,x+1);
        double val11 = (frame).at<uint8_t>(y+1,x+1);
        val = uint8_t(min(255.,(wx0*wy0)*val00
                          + (wx0*wy1)*val01
                          + (wx1*wy0)*val10
                          + (wx1*wy1)*val11));
      }
      ap_patch.at<uint8_t>(iy,ix)= val;
    }
  }
  return ap_patch;
}


template <class Camera>
cv::Mat GuidedMatcher<Camera>
::warpPatchProjective(const cv::Mat & frame,
                      const Homography & homo,
                      const Vector3d & xyz_c1,
                      const Vector3d & normal_c1,
                      const Vector2d & key_uv,
                      const Camera & cam,
                      int halfpatch_size)
{
  Matrix3d H_cur_from_key
      = (cam.intrinsics()
         *homo.calc_c2_from_c1(normal_c1,xyz_c1)
         *cam.intrinsics_inv());

  int patch_size = halfpatch_size*2 ;
  Vector2d center_cur = project2d(H_cur_from_key * unproject2d(key_uv));

  Matrix3d H_key_from_cur = H_cur_from_key.inverse();
  cerr << H_key_from_cur << endl;
  cv::Mat ap_patch(patch_size,patch_size,CV_8UC1);

  for (int ix=0; ix<patch_size; ix++)
  {
    for (int iy=0; iy<patch_size; iy++)
    {
      Vector3d idx(center_cur[0]+ix-halfpatch_size,
                   center_cur[1]+iy-halfpatch_size,1);
      Vector2d r = (project2d(H_key_from_cur*idx));

      if (ix==0 && iy == 0)
      {
      //cerr << ix << " " << iy << endl;
        cerr << r << endl;
      }

      double x = floor(r[0]);
      double y = floor(r[1]);

      uint8_t val;
      if (x<0 || y<0 || x+1>=cam.width() || y+1>=cam.height())
        val = 0;
      else
      {
        double subpix_x = r[0]-x;
        double subpix_y = r[1]-y;
        double wx0 = 1-subpix_x;
        double wx1 =  subpix_x;
        double wy0 = 1-subpix_y;
        double wy1 =  subpix_y;

        double val00 = (frame).at<uint8_t>(y,x);
        double val01 = (frame).at<uint8_t>(y+1,x);
        double val10 = (frame).at<uint8_t>(y,x+1);
        double val11 = (frame).at<uint8_t>(y+1,x+1);
        val = uint8_t(min(255.,(wx0*wy0)*val00
                          + (wx0*wy1)*val01
                          + (wx1*wy0)*val10
                          + (wx1*wy1)*val11));
      }
      ap_patch.at<uint8_t>(iy,ix)= val;
    }
  }

  cv::imshow("warpPatchProjective", ap_patch);
  cv::waitKey(2);
  return ap_patch;
}

}
