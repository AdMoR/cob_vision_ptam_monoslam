/*
 * dataset_builder.h
 *
 *  Created on: Dec 10, 2013
 *      Author: rmb-am
 */
#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <ios>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/gpu/gpu.hpp"
#include <fstream>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf/tf.h>
#include <ros/ros.h>
#include "sophus/se3.h"
#include <tf/transform_listener.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>

namespace enc = sensor_msgs::image_encodings;
using namespace cv;
using namespace Eigen;
using namespace sensor_msgs;
using namespace ros;
using namespace message_filters;


#ifndef DATASET_BUILDER_H_
#define DATASET_BUILDER_H_

typedef sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::Image> MySyncPolicy;

class DatasetBuilder {

	public:
	vector<Mat> rgb_img,d_img;
	vector<Sophus::SE3> pose_vector;
	int one_image_every_n;
	int frame_counter;

	bool use_Lab;

	tf::TransformListener tflistener;
	tf::StampedTransform current_transform;
	message_filters::Subscriber<sensor_msgs::Image> kinect_rgb_sub;
	message_filters::Subscriber<sensor_msgs::Image> kinect_depth_sub;
	ros::NodeHandle node_handle;


	DatasetBuilder(int n, bool use_lab):one_image_every_n(n),use_Lab(use_lab){};
	DatasetBuilder(int n):one_image_every_n(n),use_Lab(false){};
	void init(string pose_camera, string pose_w,string topic_rgb, string topic_d);
	void remember_training(string prefix);
	void load_from_training(string directory_path,string labels_file);

	private:
	void img_callback(const sensor_msgs::ImageConstPtr& image_rgb, const sensor_msgs::ImageConstPtr& image_d);

};



#endif /* DATASET_BUILDER_H_ */
