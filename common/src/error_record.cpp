/*
 * error_record.cpp
 *
 *  Created on: Oct 17, 2013
 *      Author: rmb-am
 */

#include <errorf.h>
#include "utilities.h"
#include <ros/ros.h>
#include <msgpkg/errmsg.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tf/transform_listener.h>


using namespace std;
using namespace cv;

string last_frame="!";



void handleTfMessage(){

		  ros::NodeHandle node;

		  cout << "computing" << endl;
		  tf::TransformListener listener;
		  tf::StampedTransform prev=tf::StampedTransform();


		  ros::Rate rate(50.0);
		  while (node.ok()){

			  tf::StampedTransform transform;
			  try{
				  //listener.waitForTransform( "/odom_combined","/base_footprint", ros::Time(0), ros::Duration(2.0));
				  listener.lookupTransform("/odom_combined","/head_cam3d_frame", ros::Time(0), transform);
			  }
			  catch (tf::TransformException ex){
				  ROS_ERROR("%s",ex.what());
			  }

			  string str =  std::to_string(ros::Time::now().sec)+std::to_string(ros::Time::now().nsec);

			  if(transform.getOrigin()!=prev.getOrigin() || transform.getRotation()!=prev.getRotation() )
				  dumpToFile(str,transform.getOrigin().getX(),transform.getOrigin().getY(),transform.getOrigin().getZ(),transform.getRotation().getX(),transform.getRotation().getY(),transform.getRotation().getZ(),transform.getRotation().getW(),"/home/rmb-am/Slam_datafiles/GT_translation.txt");
			  prev=transform;
			  rate.sleep();
		  }


}


void errCallback(msgpkg::errmsg msg)
{
	if(strcmp(msg.frame_nb.c_str(),"No value")!=0){
		cout << msg.frame_nb << " <> " << last_frame.c_str() << " <<>> "  << strcmp(msg.frame_nb.c_str(),last_frame.c_str()) << endl;
		if(strcmp(msg.frame_nb.c_str(),last_frame.c_str())!=0){
			last_frame=msg.frame_nb;
			dumpToFile(msg.frame_nb,msg.x,msg.y,msg.z,msg.rx,msg.ry,msg.rz,msg.rw);
		}
	}
	else{
		dumpToFile(std::to_string(ros::Time::now().sec)+std::to_string(ros::Time::now().nsec),msg.x,msg.y,msg.z,msg.rx,msg.ry,msg.rz,msg.rw);
	}

 }



int main(int argc, char* argv[])
{
	ros::init(argc,argv,"Error calculator");

	cout << argc << " " << argv[0] << endl;

	ros::NodeHandle nh_err;

	//ros::Subscriber err_sub = nh_err.subscribe("err_calc", 1000, errCallback);

	//Used only to record the ground truth from the bagfile
	handleTfMessage();

	ros::spin();

}
