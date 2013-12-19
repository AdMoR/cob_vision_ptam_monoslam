#ifndef SCAVISLAM_MONO_FRONTEND_H
#define SCAVISLAM_MONO_FRONTEND_H

#include <visiontools/performance_monitor.h>

#include "global.h"
#include "draw_items.h"
#include "frame_grabber.hpp"
#include "data_structures.h"
#include "quadtree.h"
#include "fast_grid.h"
#include "stereo_camera.h"
#include "transformations.h"
#include "dense_tracking.h"

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <msgpkg/point_cloud_server.h>
//#include <unordered_map>
#include <tr1/unordered_map>
#include <unordered_set>
#include "timer.h"

#include <ptam/TrackerData.h>
#include <ptam/Tracker.h>
#include <ptam/MapPoint.h>
#include <ptam/MapMaker.h>
#include <ptam/Params.h>
//#include <ptam/OpenGL.h>
#include <ptam/GLWindow2.h>
#include <ptam/MapViewer.h>


#include <gvars3/instances.h>
#include <cvd/image.h>
#include <cvd/rgb.h>
#include <cvd/byte.h>

namespace ScaViSLAM
{

template<int obs_dim>
struct TrackData;

struct MonoFrontendDrawData
{

  MonoFrontendDrawData() :
    tracked_points2d(NUM_PYR_LEVELS),
    tracked_points3d(NUM_PYR_LEVELS),
    newtracked_points2d(NUM_PYR_LEVELS),
    newtracked_points3d(NUM_PYR_LEVELS),
    tracked_anchorpoints2d(NUM_PYR_LEVELS),
    fast_points2d(NUM_PYR_LEVELS),
    new_points2d(NUM_PYR_LEVELS),
    new_points3d(NUM_PYR_LEVELS)
  {
  }

  void clear()
  {
    for (int l = 0; l<NUM_PYR_LEVELS; ++l)
    {
      new_points2d.at(l).clear();
      new_points3d.at(l).clear();
      fast_points2d.at(l).clear();
      tracked_points2d.at(l).clear();
      tracked_points3d.at(l).clear();
      newtracked_points2d.at(l).clear();
      newtracked_points3d.at(l).clear();
      tracked_anchorpoints2d.at(l).clear();
    }
    blobs2d.clear();
  }

  ALIGNED<DrawItems::Line2dList>::vector tracked_points2d;
  ALIGNED<DrawItems::Point3dVec>::vector tracked_points3d;
  ALIGNED<DrawItems::Line2dList>::vector newtracked_points2d;
  ALIGNED<DrawItems::Point3dVec>::vector newtracked_points3d;
  ALIGNED<ALIGNED<DrawItems::Point2dVec>::int_hash_map>::vector  tracked_anchorpoints2d;
  ALIGNED<DrawItems::Point2dVec>::vector fast_points2d;
  ALIGNED<DrawItems::Point2dVec>::vector new_points2d;
  ALIGNED<DrawItems::Point3dVec>::vector new_points3d;
  DrawItems::CircleList blobs2d;
};


class MonoFrontend
{
public:
  MonoFrontend             (FrameData<StereoCamera> * frame_data_,
                              PerformanceMonitor * per_mon_);
  void
  processFirstFrame          ();
  bool
  processFrame               (bool * is_frame_dropped);
  void
  initialize                 (ros::NodeHandle & nh);


  // getter and setter
  NeighborhoodPtr & neighborhood()
  {
    return neighborhood_;
  }
  static vector<pair<int,int>>
  lineBresenham(int p1x, int p1y, int p2x, int p2y);

  int
  getNewUniquePointId           ();

  vector<int>
  computeLineDescriptor(int x1,int y1, int x2, int y2, vector<pair<int,int>> & pixelsOnLine);

  vector<int>
  computeLineDescriptorSSD(int x1,int y1, int x2, int y2, vector<pair<int,int>> & pixelsOnLine);

  const NeighborhoodPtr & neighborhood() const
  {
    return neighborhood_;
  }

  const MonoFrontendDrawData & draw_data() const
  {
    return draw_data_;
  }

  const SE3& T_cur_from_actkey() const
  {
    return T_cur_from_actkey_;
  }

  const DenseTracker & tracker() const
  {
    return tracker_;
  }

  stack<AddToOptimzerPtr> to_optimizer_stack;
  tr1::unordered_map<int,Frame>  keyframe_map;
  tr1::unordered_map<int,list<CandidatePoint3Ptr > >  newpoint_map;
  vector<int>  keyframe_num2id;
  IntTable keyframe_id2num;
  int actkey_id;

private:

  struct RemoveCondition
  {
    RemoveCondition(const tr1::unordered_set<CandidatePoint3Ptr > &
                    matched_new_feat)
      :matched_new_feat(matched_new_feat)
    {
    }

    const tr1::unordered_set<CandidatePoint3Ptr > & matched_new_feat;

    bool operator()(const CandidatePoint3Ptr& ptr)
    {
      return matched_new_feat.find(ptr)!=matched_new_feat.end();
    }
  };

  struct Params
  {
    int  newpoint_clearance;
    int  covis_thr;
    int  num_frames_metric_loop_check;
    int  new_keyframe_pixel_thr;
    int  new_keyframe_featuerless_corners_thr;
    int  graph_inner_window;
    int  graph_outer_window;
    bool  save_dense_cloud;
    bool livestream;
  };

  struct PointStatistics
  {
    PointStatistics(int USE_N_LEVELS_FOR_MATCHING)
      : num_matched_points(USE_N_LEVELS_FOR_MATCHING)
    {
      num_points_grid2x2.setZero();
      num_points_grid3x3.setZero();

      for (int l=0; l<USE_N_LEVELS_FOR_MATCHING; ++l)
      {
        num_matched_points[l]=0;
      }
    }

    vector<int> num_matched_points;
    Matrix2i num_points_grid2x2;
    Matrix3i num_points_grid3x3;
  };

  bool
  shallWeDropNewKeyframe     (const PointStatistics & point_stats);

  bool
  shallWeSwitchKeyframe      (const list<TrackPoint3Ptr> & trackpoint_list,
                              int * other_id,
                              tr1::unordered_map<int, Line> tracked_lines,
                              SE3 * T_cur_from_other,
                              ALIGNED<QuadTree<int> >::vector
                              * other_point_tree,
                              PointStatistics * other_stat);

  void
  addNewKeyframe             (const ALIGNED<QuadTree<int> >::vector &feature_tree,
                              const AddToOptimzerPtr & to_optimizer,
                              tr1::unordered_set<CandidatePoint3Ptr > *matched_new_feat,
                              vector<Line> newLinesOnFrame,
                              ALIGNED<QuadTree<int> >::vector * point_tree,
                              PointStatistics * point_stats);
  AddToOptimzerPtr
  processMatchedPoints       (const TrackData<3> & track_data,
                              int num_new_feat_matched,
                              ALIGNED<QuadTree<int> >::vector * point_tree,
                              tr1::unordered_set<CandidatePoint3Ptr > *
                              matched_new_feat,
                              PointStatistics * stats);

  void
  computeLines(std::vector<Line>  &linesOnCurrentFrame, SE3 & T_cur_from_w, bool firstFrame, int frame_id);

  int computeMeanIntensityOfNeighborhood(const int x, const int y, const int sizeOfNeighborhood);

  void
  updateOptimizedPluckerParameters(tr1::unordered_map<int,Line> &tracked_lines, LineTable &tracked_lines_result);

  bool
  request3DCoordsAndComputePlueckerParams(Vector6d &pluckerLines,Vector6d &localPluckerLines, Vector3d &startingPoint, Vector3d &endPoint, int x1, int y1, int x2, int y2);

  bool
  request3DCoords(int x, int y, Vector3d *output);


  bool
  matchAndTrack              (const ALIGNED<QuadTree<int> >::vector &
                              feature_tree,
                              TrackData<3> * track_data,
                              int * num_new_feat_matched,SE3 &T_anchorkey_from_w);

//  bool
//  matchAndTrack              (const ALIGNED<QuadTree<int> >::vector &
//                              feature_tree,
//                              TrackData<3> * track_data,  int nb_line,
//                              int * num_new_feat_matched,SE3 &T_anchorkey_from_w);

#ifdef SCAVISLAM_CUDA_SUPPORT
  void
  calcDisparityGpu           ();

#else
  void
  calcDisparityCpu           ();

#endif
  void
  computeFastCorners         (int trials,
                              ALIGNED<QuadTree<int> >::vector * feature_tree,
                              vector<CellGrid2d> * cell_grid_2d);
  void
  recomputeFastCorners       (const Frame & frame,
                              ALIGNED<QuadTree<int> >::vector * feature_tree);
  void
  addNewPoints               (int new_keyframe_id,
                              const ALIGNED<QuadTree<int> >::vector &
                              feature_tree);
  void
  addMorePoints              (int new_keyframe_id,
                              const ALIGNED<QuadTree<int> >::vector &
                              feature_tree,
                              const Matrix3i & add_flags,
                              ALIGNED<QuadTree<int> >::vector * new_qt,
                              vector<int> * num_points);

  void
  addMorePointsToOtherFrame  (int new_keyframe_id,
                              const SE3 & T_newkey_from_cur,
                              const ALIGNED<QuadTree<int> >::vector &
                              feature_tree,
                              const Matrix3i & add_flags,
                              const cv::Mat & disp,
                              ALIGNED<QuadTree<int> >::vector * new_qt,
                              vector<int> * num_points);

  void
  addNewLinesToKeyFrame(std::vector<Line> &linesOnCurrentFrame, AddToOptimzerPtr & to_optimizer,
			std::vector<int> &localIDsofNewLinesToBeAdded, const SE3 &T_cur_from_w);

  void initTracker(const CVD::ImageRef & size);
  static void GUICommandCallBack(void* ptr, std::string sCommand, std::string sParams);

  void camera_info_cb(const sensor_msgs::CameraInfoConstPtr& rgbd_camera_info);

  void computeSVDPluecker(vector<pair<int,int>> pixelsOnLine);


  ATANCamera* mpCamera;
  ptam::Map *mpMap;
  MapMaker *mpMapMaker;
  Tracker *mpTracker;
  GLWindow2 *mGLWindow;
  MapViewer *mpMapViewer;
  CVD::Image<CVD::byte > img_bw_;
  CVD::Image<CVD::Rgb<CVD::byte> > img_rgb_;
  bool first_frame_;
  //ros::NodeHandle nh;
  ros::ServiceClient client_3D_data;
  ros::Subscriber rgbd_camera_info_;
  //std::unordered_map<int,Line> tracked_lines;
  tr1::unordered_map<int,Line> tracked_lines;
  Matrix<double,3,3> camera_matrix;
  double timercount1;
  double timercount2;
  int frameCounter;

  Frame cur_frame_;
  vector<Vector3d> edges;
  //cv::Mat prev_frame_gray;
  //cv::Mat prev_frame_rgb;

  FrameData<StereoCamera> * frame_data_;
  PerformanceMonitor * per_mon_;

  NeighborhoodPtr neighborhood_;
  SE3 T_cur_from_actkey_; //SE3 transformation between old key frame and new key frame

  vector<FastGrid> fast_grid_;
  SE3XYZ_STEREO se3xyz_stereo_;

  int USE_N_LEVELS_FOR_MATCHING;
  int unique_point_id_counter_;

  Params params_;
  MonoFrontendDrawData draw_data_;
  DenseTracker tracker_;

  double av_track_length_;

private:
  DISALLOW_COPY_AND_ASSIGN(MonoFrontend)
};

}

#endif

