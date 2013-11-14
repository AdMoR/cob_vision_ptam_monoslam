#ifndef RGBD_GRABBER_H
#define RGBD_GRABBER_H


#include <iostream>
#include <stdlib.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
//#include <pcl/io/openni_grabber.h>
//#include <pcl/io/grabber.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
//#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <stereo_msgs/DisparityImage.h>

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

#include "global.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>



namespace ScaViSLAM
{

class RgbdGrabber
{
public:

  void
  initialize                 ();

  bool
  getFrame                   (cv::Mat * rgb,
                              cv::Mat * depth);

  void
  operator                   ()();

private:

  ros::NodeHandle node_handle;	
  //image_transport::ImageTransport *img_transport;
  message_filters::Subscriber<sensor_msgs::Image> kinect_rgb_sub;
  message_filters::Subscriber<sensor_msgs::Image> kinect_depth_sub;
  message_filters::Subscriber<stereo_msgs::DisparityImage> kinect_disp_sub;
  
  ATANCamera* mpCamera;
  ptam::Map *mpMap;
  MapMaker *mpMapMaker;
  Tracker *mpTracker;
  GLWindow2 *mGLWindow;
  MapViewer *mpMapViewer;
  bool first_frame_;
  CVD::Image<CVD::byte > img_bw_;
  CVD::Image<CVD::Rgb<CVD::byte> > img_rgb_;
  void kinect_callback(const sensor_msgs::ImageConstPtr& image_rgb, const stereo_msgs::DisparityImageConstPtr& image_depth);
//  void kinect_callback(const sensor_msgs::ImageConstPtr& image_rgb,const sensor_msgs::ImageConstPtr& image_depth);
  void initTracker(const CVD::ImageRef & size);
  static void GUICommandCallBack(void* ptr, std::string sCommand, std::string sParams);





};

}

#endif // RGBD_GRABBER_H
