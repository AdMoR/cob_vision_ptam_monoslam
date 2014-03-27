/*
 * dataset_builder.cpp
 *
 *  Created on: Dec 10, 2013
 *      Author: rmb-am
 */

#include "dataset_builder.h"
#include <cv_bridge/cv_bridge.h>
#include <boost/thread.hpp>
#include "utilities.h"
#include <string>
#include <iostream>
#include <boost/regex.hpp>

boost::mutex global_mutex;


//Structure useful to stock images from a bagfile.
//The used images and their labels are stored to file ready for reuse



void DatasetBuilder::remember_training(string prefix){
	int i=0;
	for(auto ptr=pose_vector.begin();ptr!=pose_vector.end();ptr++){
		stringstream ssrgb,ssd;
		Mat show,save;
		ssrgb<<prefix<<"rgb_"<<i<<".png";
		ssd<<prefix<<"d_"<<i<<".yml";
		cout << ssrgb.str().c_str() << endl;
		SE3 se=(*ptr);
		dumpToFile(std::to_string(i),se.translation()(0),se.translation()(1),se.translation()(2),se.unit_quaternion().x(),se.unit_quaternion().y(),se.unit_quaternion().z(),se.unit_quaternion().w(),"/home/rmb-am/Slam_datafiles/frame_labels.txt");
		if(use_Lab)
			cvtColor(rgb_img[i],save,CV_Lab2RGB);
		else
			save=rgb_img[i];
		cv::imwrite(ssrgb.str().c_str(),save);
		FileStorage fs(ssd.str().c_str(), FileStorage::WRITE);
      	Mat matissss=d_img[i];
      	fs << "d" << matissss ;

		i++;
	}

}

void DatasetBuilder::load_from_training(string directory_path, string labels_file){

	//Extract the names of the files
	DIR* dir;
	struct dirent *ent;
	vector<string> file_names;
	map<int,string> d_full_file_name,rgb_full_file_name;
	 if ((dir = opendir (directory_path.c_str())) != NULL) {
	            /* print all the files and directories within directory */
	            while ((ent = readdir (dir)) != NULL) {
	                if ( strcmp(ent->d_name,"d_")>=1 || strcmp(ent->d_name,"rgb_")>=1){
	                   // cout << ent->d_name << endl;
	                   file_names.push_back(ent->d_name);}
	            }
	            closedir (dir);
	        }
	 else {
	            /* could not open directory */
	            perror ("ahhhhhhhhhhhhhhhhh");
	            return;
	       }
	 cout << "directory files found" << endl;

	 //Append the prefix and classify the names
	 for(auto it=file_names.begin();it!=file_names.end();it++){
		 stringstream ss;
		 ss << directory_path << (*it) ;

		 vector<string> fields,fields2;

		boost::regex e("d_(.*)\.(.*)");
		boost::match_results<std::string::const_iterator> what;

	   if(boost::regex_match((*it), what, e, boost::match_default | boost::match_partial)){
				 d_full_file_name.insert(make_pair(atoi(what[1].str().c_str()),ss.str()));
				 //cout << "d is " << (*it) << endl;
	   }
	   else{
			boost::regex e("rgb_(.*)\.(.*)");
		   if(boost::regex_match((*it), what, e, boost::match_default | boost::match_partial)){
				 rgb_full_file_name.insert(make_pair(atoi(what[1].str().c_str()),ss.str()));
				 //cout << "rgb is " << (*it) << endl;
				 }
		   else{
			   cout << "ERROR : FILE NAME ERROR : " << it->c_str() << endl;
			   exit(1);
		   }
	   }
		 }
	 cout << "full name obtained" << endl;

	 //Load the labels
	 ifstream fi;
	 vector<SE3> pose_v;
	 fi.open(labels_file);
	 if(fi.is_open()){

		 while (!fi.eof()){
				 string buffer;
				 vector <string> fields;
				 std::getline(fi, buffer);

				 boost::algorithm::split_regex( fields, buffer,  boost::regex(" ")  );

				 if(fields.size()<4)
				 {cout << "end of file" << endl;
				 break;
				 }

				 int num=atoi(fields[0].c_str());
				 Vector3d v = Vector3d(atof(fields[1].c_str()),atof(fields[2].c_str()),atof(fields[3].c_str()));
				 Quaterniond q = Quaterniond(atof(fields[7].c_str()),atof(fields[4].c_str()),atof(fields[5].c_str()),atof(fields[6].c_str()));
				 SE3 sss = SE3(q,v);
				pose_v.push_back(sss);

		 }
	 }
	 else{
		 cout << "ERROR COULD NOT OPEN LABEL FILE" << endl;
		 exit(1);
	 }
	 cout << "labels acquired" << endl;


	 //Load the image files
	 for(int i = 0; i< d_full_file_name.size();i++){
		 Mat rgb,d,fd,load;

		 string d_name,rgb_name;

		 d_name=d_full_file_name.find(i)->second;
		 rgb_name=rgb_full_file_name.find(i)->second;

		//d=imread(d_name);
		 rgb=imread(rgb_name);
		 if(use_Lab)
			 cvtColor(rgb,load,CV_RGB2Lab);
		 else
			 load=rgb;
		 d=Mat::zeros(rgb.size(),CV_32FC1);
		 getMatrixFromFile(d,d_name);
 		 //cv::convertScaleAbs( d, fd ,7000./256.,0);
		 d_img.push_back(d);
		 rgb_img.push_back(load);
		 pose_vector.push_back(pose_v[i]);

		 //cout << d_name << " " << rgb_name << " " << pos[i] << endl;

	 }

}


void DatasetBuilder::img_callback(const sensor_msgs::ImageConstPtr& image_rgb, const sensor_msgs::ImageConstPtr& image_d){
	boost::mutex::scoped_lock lock(global_mutex);

	try {
		if((frame_counter%one_image_every_n)==0){
			Mat save;
			if(use_Lab)
				cvtColor(cv_bridge::toCvCopy(image_rgb, enc::BGR8)->image,save,CV_RGB2Lab);
			else
				save=cv_bridge::toCvCopy(image_rgb, enc::BGR8)->image;
			rgb_img.push_back( save);
			d_img.push_back(cv_bridge::toCvCopy(image_d, enc::TYPE_32FC1)->image );
			tflistener.lookupTransform("/map","/head_cam3d_frame", ros::Time(0), current_transform);
			Vector3d v=Vector3d(current_transform.getOrigin().z(),current_transform.getOrigin().y(),current_transform.getOrigin().x());
			Quaterniond q = Quaterniond(current_transform.getRotation().x(),current_transform.getRotation().y(),current_transform.getRotation().z(),current_transform.getRotation().w());
			SE3 se=SE3(q,v);
			pose_vector.push_back(se);
			std::cout << "new image !" << std::endl;
		}
		frame_counter=(frame_counter+1)%one_image_every_n;

	}
	catch (cv_bridge::Exception& e) {
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}

}


void DatasetBuilder::init( string pose_camera, string pose_w,string topic_rgb="/camera/rgb/image_color", string topic_d="/camera/depth_registered/image_raw"){


	tf::Transform transform;
	kinect_rgb_sub.subscribe(node_handle, topic_rgb , 1);
	kinect_depth_sub.subscribe(node_handle,topic_d, 1);
	Synchronizer<MySyncPolicy> sync(MySyncPolicy(3), kinect_rgb_sub,kinect_depth_sub);
	std::cout << "ready to get images" << std::endl;
	sync.registerCallback(boost::bind( &DatasetBuilder::img_callback, this, _1, _2));
	ros::spin();

}
