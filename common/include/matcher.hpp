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

#ifndef SCAVISLAM_MATCHER_HPP
#define SCAVISLAM_MATCHER_HPP

#include <stdint.h>
#include <tr1/memory>

#include <opencv2/opencv.hpp>

#include "data_structures.h"
#include "global.h"
#include "quadtree.h"
#include "mono_frontend.h"

namespace Sophus
{
class SE3;
}

namespace ScaViSLAM
{

using namespace Sophus;
template<int>
class IdObs;
template<int>
class CandidatePoint;
class Frame;
class Homography;
class SE3XYZ;

template <int obs_dim>
struct TrackData
{
  void clear()
  {
    obs_list.clear();
    point_list.clear();
    ba2globalptr.clear();
  }

  typename ALIGNED<IdObs<obs_dim> >::list obs_list; //observations list
  vector<Vector3d> point_list;
  vector<tr1::shared_ptr<CandidatePoint<obs_dim> > > ba2globalptr;
};

template <class Camera>
class GuidedMatcher
{
public:

  static void
  match                      (const tr1::unordered_map<int,Frame> &
                              keyframe_map,
                              const SE3 & T_cur_from_actkey,
                              const Frame & cur_frame,
                              const ALIGNED<QuadTree<int> >::vector &
                              feature_tree,
                              const typename ALIGNED<Camera>::vector & cam_vec,
                              int actkey_id,
                              const ALIGNED<FrontendVertex>::int_hash_map
                              & vertex_map,
                              const list< tr1::shared_ptr<
                              CandidatePoint<Camera::obs_dim> > > & ap_map,
                              int SEARCHRADIUS,
                              int thr_mean,
                              int thr_std,
                              TrackData<Camera::obs_dim> * track_data,
                              SE3 &T_anchorkey_from_w);
  static void
  match                      (const tr1::unordered_map<int,Frame> &
                              keyframe_map,
                              const SE3 & T_cur_from_actkey,
                              const Frame & cur_frame,
                              const ALIGNED<QuadTree<int> >::vector &
                              feature_tree,
                              const typename ALIGNED<Camera>::vector & cam_vec,
                              int actkey_id,
                              const ALIGNED<FrontendVertex>::int_hash_map
                              & vertex_map,
                              const list< tr1::shared_ptr<
                              CandidatePoint<Camera::obs_dim> > > & ap_map,
                              int SEARCHRADIUS,
                              int thr_mean,
                              int thr_std,
                              TrackData<Camera::obs_dim> * track_data);

 static void
 post_matching			    (tr1::unordered_map<int,Line> &tracked_lines,
  							std::vector<Line> linesOnCurrentFrame,
  							map<int,int>& matchedline_id ,
  							set<int>& currentLineLocalId,
  							map<int,Vector3d>& projection_map,
  							bool color_mode,
  							vector<pair<int,cv::Scalar> >& colormap,
  							cv::Mat *curFrameRGB,
  							int KF_NUMBER,
  							int line_num,
  							int line_tracked,
  							bool one_line_track,
  							bool verbose,
  							Matrix<double,3,4> projectionsMatrix,
  							SE3 T_actkey_from_w
  							);


  static void
  findBestConfiguration       (tr1::unordered_map<int,Line> &tracked_lines,
  	  	  	  	  	  	  	  std::vector<Line> &linesOnCurrentFrame,
  							  tr1::unordered_map<int,std::multimap<double,std::pair<Line,double> > > & candidates ,
  							  map<int,int> & best_matched,
  							  double max_error);


  static void
  lineMatcher 				(std::vector<Line> &linesOnCurrentFrame,
		  	  	  	  	    tr1::unordered_map<int,Line> &tracked_lines,
		  	  	  	  	  	const SE3 & T_cur_from_actkey,
		  	  	  	  	  	int actkey_id,
		  	  	  	  	  	const ALIGNED<FrontendVertex>::int_hash_map & vertex_map,
		  	  	  	  	  	const Matrix<double,3,3> camera_matrix,
		  	  	  	  	  	cv::Mat *curFrameRGB,
		  	  	  	  	  	const vector<Vector3d> &edges,
		  	  	  	  	  	MonoFrontend *monoFrontednPtr,
		  	  	  	  	  	AddToOptimzerPtr & to_optimizer,
		  	  	  	  	  	std::vector<int> & localIDsofNewLinesToBeAdded,
		  	  	  	  	  	SE3 &T_cur_from_w,
		  	  	  	  	  	int frame_id);

  static cv::Mat
  warpPatchProjective        (const cv::Mat & frame,
                              const Homography & homo,
                              const Vector3d & xyz_c1,
                              const Vector3d & normal,
                              const Vector2d & key_uv,
                              const Camera & cam,
                              int halfpatch_size);


  static cv::Mat
  warpAffinve                (const cv::Mat & frame,
                              const SE3 & T_c2_from_c1,
                              double depth,
                              const Vector2d & key_uv,
                              const Camera & cam,
                              int halfpatch_size);

  static void
     drawLine(Vector3d projectedHomogeneousLine, cv::Mat curFrameRGB, const string &WindowName, const cv::Scalar & color,
     		  bool verbose);
private:

  static Line
  findMatchedLineWithSmallestError(const std::multimap<double,std::pair<Line,double>> &matchedLines, bool debug);

  static std::multimap<double,Line>
  findNearestLinesOnCurrentFrameOrderedByDistance(Vector3d line,  std::vector<Line> &linesOnCurrentFrame,float err,Vector2d reference_point, bool debug);

   static bool
   matchTwoLines(std::vector<int> descriptor1, std::vector<int> descriptor2,
   		  	  	  	  	  	  double &normalizedError, bool debug);
   static bool
      matchTwoLinesSSD(std::vector<int> descriptor1, std::vector<int> descriptor2,
      		  	  	  	  	  	  double &normalizedError, bool debug, double error_thres);

 static bool
 	 matchTwoLinesDD(std::vector<float> descriptor1, std::vector<float> descriptor2, double &normalizedError, bool debug,double error_thr);


  typedef uint8_t aligned_uint8_t __attribute__ ((__aligned__(16)));

  struct MatchData
  {
    explicit MatchData(int min_dist)
      : min_dist(min_dist),
        index(-1),
        uv_pyr(0,0)
    {}
    int min_dist;
    int index;
    Vector2i uv_pyr;
  };

  static void
  matchPatchZeroMeanSSD      (//const aligned_uint8_t * data_cur,
                              //const aligned_uint8_t * data_key,
                              int key_pixel_sum,
                              int key_pixel_sum_sq,
                              int * znssd);
  static void
  computePatchScores         (//const aligned_uint8_t * data,
                              int * pixel_sum,
                              int * pixel_sum_square);


  static cv::Mat
  warp2d                     (const cv::Mat & patch_in,
                              const Vector2f & uv_pyr);


  static void
  returnBestMatch            (const cv::Mat & ap_patch,
                              const Frame & cur_frame,
                              const MatchData & match_data,
                              const Vector3d & xyz_actkey,
                              const tr1::shared_ptr<
                              CandidatePoint<Camera::obs_dim> > & ap,
                              TrackData<Camera::obs_dim> * track_data);

  static void
  matchCandidates            (const ALIGNED<QuadTreeElement<int> >::list &
                              candidates,
                              const Frame & cur_frame,
                              const typename ALIGNED<Camera>::vector & cam_vec,
                              int pixel_sum,
                              int pixel_sum_square,
                              int level,
                              MatchData * match_data);

  static bool
  computePrediction          (const SE3 & T_w_from_cur,
                              const typename ALIGNED<Camera>::vector & cam_vec,
                              const tr1::shared_ptr<
                              CandidatePoint<Camera::obs_dim> > & ap,
                              const ALIGNED<FrontendVertex>::int_hash_map
                              & verte_map,
                              Vector2d * uv_pyr,
                              SE3 * T_anchorkey_from_w);

  static bool
  createObervation           (const Vector2f & new_uv_pyr,
                              const cv::Mat & disp,
                              int level,
                              Matrix<double,Camera::obs_dim,1> * obs);

  static bool
  subpixelAccuracy           (const cv::Mat & key_patch_8u,
                              const Frame & cur_frame,
                              const Vector2i & uv_pyr_in,
                              int level,
                              Vector2f * uv_pyr_pout);
  bool firstTime;
  static const int HALFBOX_SIZE = 4;
  static const int BOX_SIZE = HALFBOX_SIZE*2;
  static const int BOX_AREA = BOX_SIZE*BOX_SIZE;

  static uint8_t KEY_PATCH[BOX_AREA];
  static uint8_t CUR_PATCH[BOX_AREA];

  DISALLOW_COPY_AND_ASSIGN(GuidedMatcher)
};
}

#endif
