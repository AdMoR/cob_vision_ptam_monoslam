#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ros/conversions.h>
#include <iostream>
#include <msgpkg/point_cloud_server.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr globalPointCloud;

bool get_reg_3D_coord(msgpkg::point_cloud_server::Request  &req,
		msgpkg::point_cloud_server::Response &res )


{
	try {
		if (globalPointCloud == NULL) {
			std::cout << "pointcloud is null!" << std::endl;
		} else {
			res.x = globalPointCloud->at(req.x, req.y).x;
			res.y = globalPointCloud->at(req.x, req.y).y;
			res.z = globalPointCloud->at(req.x, req.y).z;
			std::cout<<"request u,v: "<<req.x<<", "<<req.y<<std::endl;
			std::cout<<"response x,y,z "<<res.x<<", "<<res.y<<", "<<res.z<<std::endl;
			return true;
		}
	} catch (...) {
		std::cout << "exception in point_cloud_server" << std::endl;
		return false;
	}

}

void kinect_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{

	//pcl::PointCloud<pcl::PointXYZ> LocalPointCloud;
    pcl::fromROSMsg<pcl::PointXYZ>(*msg, *globalPointCloud);
    //globalPointCloud = *LocalPointCloud;
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "get_kinect_reg_3D_coord");
  ros::NodeHandle n;
  ros::Subscriber cloud = n.subscribe("/camera/depth_registered/points", 1, kinect_callback);

  globalPointCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
  ros::ServiceServer service = n.advertiseService("get_kinect_reg_3D_coord", get_reg_3D_coord);
  ROS_INFO("Get x,y,z registered world coordinates from kinect.");
  ros::spin();

  return 0;
}
