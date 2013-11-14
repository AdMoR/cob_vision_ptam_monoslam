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

#ifndef SCAVISLAM_DATA_STRUCTURES_H
#define SCAVISLAM_DATA_STRUCTURES_H

#include <list>

#include <sophus/se3.h>
#include "global.h"
#include "keyframes.h"



namespace ScaViSLAM
{

using namespace std;
using namespace Sophus;
using namespace Eigen;

typedef tr1::shared_ptr<Line> LinePtr;
typedef tr1::unordered_map<int,LinePtr>  LineTable;

template <int Dim>
class CandidatePoint
{
public:
  CandidatePoint(int point_id,
              const Vector3d & xyz_anchor,

              int anchor_id,
              //const ImageFeature<ObsDim> & keyfeat,
              const typename VECTOR<Dim>::col & anchor_obs_pyr,
              int anchor_level,
              const Vector3d& normal_anchor)
    : point_id(point_id),
      xyz_anchor(xyz_anchor),
      anchor_id(anchor_id),
      anchor_obs_pyr(anchor_obs_pyr),
      anchor_level(anchor_level),
      normal_anchor(normal_anchor)

  {
  }

  int point_id;

  Vector3d xyz_anchor;
  int anchor_id;
  typename VECTOR<Dim>::col anchor_obs_pyr;
  int anchor_level;
  Vector3d normal_anchor;


  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef tr1::shared_ptr<CandidatePoint<2> > ActivePoint2Ptr;
typedef tr1::shared_ptr<CandidatePoint<3> > CandidatePoint3Ptr;


template <int Dim>
struct ImageFeature
{
  ImageFeature(const Matrix<double,Dim,1>& center,
               int level)
    : center(center),
      level(level)
  {
  }

  Matrix<double,Dim,1> center;	// image coordinates of point in 2d or 3d
  int level;	// pyramid level of observation (its the same as anchor_level)

  typedef typename ALIGNED<ImageFeature<Dim> >::int_hash_map Table; //feature table

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



template <int Dim>
class TrackPoint
{
public:
  TrackPoint(int global_id,
             const ImageFeature<Dim> & feat)
    : global_id(global_id), feat(feat)

  {
  }

  int global_id;
  ImageFeature<Dim> feat;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

typedef tr1::shared_ptr<TrackPoint<2> > TrackPoint2Ptr;
typedef tr1::shared_ptr<TrackPoint<3> > TrackPoint3Ptr;


template <int Dim>
struct NewTwoViewPoint
{
  NewTwoViewPoint(int point_id,
                  int anchor_id,
                  const Vector3d & xyz_anchor,
                  const typename VECTOR<Dim>::col & anchor_obs_pyr,
                  int anchor_level,
                  const Vector3d & normal_anchor,
                  const ImageFeature<Dim> & feat_newkey) //Dim = dimension of observation, 2 for monocular
    : point_id(point_id),
      anchor_id(anchor_id),
      xyz_anchor(xyz_anchor),
      anchor_obs_pyr(anchor_obs_pyr),
      anchor_level(anchor_level),
      normal_anchor(normal_anchor),
      feat_newkey(feat_newkey)

  {}

  int point_id;		// unique id
  int anchor_id;	// unique id of frame of first detection of anchor point
  Vector3d xyz_anchor; // actual 3d position
  const typename VECTOR<Dim>::col anchor_obs_pyr; // initial observation in anchor frame
  int anchor_level;		// pyramid level of observation
  Vector3d  normal_anchor; // estimate of surface normal
  ImageFeature<Dim> feat_newkey;	// image coordinates (3d with homogeneous coordinates) or 3d coordinates of point???


  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef tr1::shared_ptr<NewTwoViewPoint<2> > NewTwoViewPoint2Ptr;
typedef tr1::shared_ptr<NewTwoViewPoint<3> > NewTwoViewPoint3Ptr;


struct AddToOptimzer
{
  AddToOptimzer(bool first_frame=false) : first_frame(first_frame)
  {
  }

  SE3 T_newkey_from_oldkey;		// transformation between old key frame and new key frame
  list<NewTwoViewPoint3Ptr> new_point_list;		// new points to been added to the tracking list (and the map)
  list<TrackPoint3Ptr> track_point_list;		// old points that have been tracked (and found)
  tr1::unordered_map<int, Line> tracked_lines;

  int oldkey_id;	// unique id of the last key frame
  int newkey_id;	// unique id of the new key frame

  Frame kf;		// the backend uses it to pass it to placerecognizer, loopclosure etc. but not to the Graph/g2o!

  bool first_frame;		// is this frame the very first key frame entered to the system

};
typedef tr1::shared_ptr<AddToOptimzer> AddToOptimzerPtr;

struct FrontendVertex {
	SE3 T_me_from_w;
	multimap<int, int> strength_to_neighbors; /*  From SlamGraph::ComputeStrength:
												we don't add link to very distant keyframe
											 Thus, only add a link to keyframe if
											  - there are enough covisible points
											  - AND
											    * if new features are to be initialized in the keyframe
											      (which implies the keyframe is recent in time)
											    * or if features are matched in all parts of the image.*/
	ImageFeature<3>::Table feat_map;	//feature table
	ALIGNED<Line>::int_hash_map line_map; //line feature map
};

struct Neighborhood
{
  list<CandidatePoint3Ptr> point_list; /*candidate 3d points which are visible from the topological neighbourhood around
  	  	  	  	  	  	  	  	  	  	  the reference keyframe Vref. Advantage over PTAM’s approach: it takes care of
  	  	  	  	  	  	  	  	  	  	  occlusion, because those points are not visible from other frames. As in PTAM,
  	  	  	  	  	  	  	  	  	  	  a point is only actively searched for if its reprojection lies within
  	  	  	  	  	  	  	  	  	  	  the current image boundaries.
  	  	  	  	  	  	  	  	  	  	  */
  ALIGNED<FrontendVertex>::int_hash_map vertex_map; //??? neighborhood consists of all keyframes Vi connected to Vref including itself ??

   LineTable tracked_lines_result;
};

typedef tr1::shared_ptr<Neighborhood> NeighborhoodPtr;

}

#endif
