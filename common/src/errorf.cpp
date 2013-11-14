/*
 * errorf.cpp
 *
 *  Created on: Oct 15, 2013
 *      Author: rmb-am
 */

#include <errorf.h>
#include <ros/ros.h>
#include <msgpkg/errmsg.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <tf/tf.h>
#include <data_structures.h>
#include <global.h>
#include "utilities.h"

using namespace std;
using namespace cv;


const int MAX_CHARS_PER_LINE = 512;
const int MAX_TOKENS_PER_LINE = 20;
const char* const DELIMITER = " ";








void parseFile(vector<tf::Transform> & tfv, vector<tf::Transform> & tfgtv, bool use_stereo=false,std::string pathToFile="/home/rmb-am/newcollege/measurements.txt"){

	  ifstream fin,fgt;
	  fin.open(pathToFile); // open a file
	  if (!fin.good())
	    return ; // exit if file not found
	  if(!use_stereo)
		  fgt.open("/home/rmb-am/newcollege/clean.txt"); // open a file
	  else
		  fgt.open("/home/rmb-am/newcollege/measurements_stereo.txt");
	  if (!fgt.good())
	    return ; // exit if file not found

	  cout << "reading file " << endl;

	  int k=0;
	  // read each line of the file
	  while (!fin.eof())
	  {
			// read an entire line into memory
			string buffer;
			vector <string> fields;
			std::getline(fin, buffer);
			//cout << "Original = " << buffer << "\"\n\n";
			boost::algorithm::split_regex( fields, buffer,  boost::regex(" ")  );
			if(fields.size()<2)
				{cout << "end of file" << endl;
				return;
				}
			string name=fields[0];
			float x=atof(fields[1].c_str()),y=atof(fields[2].c_str()),z=atof(fields[3].c_str()),rx=atof(fields[4].c_str()),ry=atof(fields[5].c_str()),rz=atof(fields[6].c_str()),rw=atof(fields[7].c_str());

			tf::Transform tr ;
			tf::Vector3 v=tf::Vector3(x,y,z);
			tf::Quaternion q(rx,ry,rz,rw);
			tr.setOrigin( v );
			tr.setRotation( q);

			tfv.push_back(tr);


			string frame="";


			do{
				std::getline(fgt, buffer);
				boost::algorithm::split_regex( fields, buffer,  boost::regex(" ")  );
				if(!use_stereo)
					frame=fields[1];
				else
					frame=fields[0];
				//cout << name << " != " << frame << " ? " << endl;
			}while(strcmp(name.c_str(),frame.c_str())!=0);

			tf::Transform trgt ;
			tf::Vector3 vgt;
			tf::Quaternion qgt;

			if(!use_stereo){
				vgt=tf::Vector3(atof(fields[9].c_str()),atof(fields[10].c_str()),atof(fields[11].c_str()));
				qgt.setRPY(atof(fields[12].c_str()),atof(fields[13].c_str()),atof(fields[14].c_str()));}
			else{
				vgt=tf::Vector3(atof(fields[1].c_str()),atof(fields[2].c_str()),atof(fields[3].c_str()));
				qgt.setValue(atof(fields[4].c_str()),atof(fields[5].c_str()),atof(fields[6].c_str()),atof(fields[7].c_str()));
			}
			trgt.setOrigin( vgt );
			trgt.setRotation( qgt );

			tfgtv.push_back(trgt);

			if(k<10)
				cout << tfgtv[tfgtv.size()-1].getOrigin().x() << " " << tfv[tfv.size()-1].getOrigin().x() << endl;
			k++;

	  }



}

void lesserFramePoseConverter(string pathToDataFile="/home/rmb-am/Slam_datafiles/map/aux_map.txt", string pathToKfFile="/home/rmb-am/Slam_datafiles/map/map.txt" ,string pathToOutputFile="/home/rmb-am/Slam_datafiles/map/rect_map.txt"){

	ifstream flf,fkf,fout;
	vector< pair<int,tf::Transform> > kfTf,lfTf;




	flf.open(pathToDataFile); // open a file
	if (!flf.good())
		return ; // exit if file not found
	fkf.open(pathToKfFile); // open a file
	if (!fkf.good())
		return ; // exit if file not found
//	fout.open(pathToOutputFile);
//	if (!fout.good())
//		return ; // exit if file not found

	cout << "reading file " << endl;

	 while (!fkf.eof()){
		 string buffer;
		 vector <string> fields;
		 std::getline(fkf, buffer);

		 boost::algorithm::split_regex( fields, buffer,  boost::regex(" ")  );
		 if(fields.size()<2)
		 {cout << "end of file" << endl;
		 break;
		 }

		 string name=fields[0];

		 float x=atof(fields[1].c_str()),y=atof(fields[2].c_str()),z=atof(fields[3].c_str()),rx=atof(fields[4].c_str()),ry=atof(fields[5].c_str()),rz=atof(fields[6].c_str()),rw=atof(fields[7].c_str());

		 tf::Transform tr ;
		 tf::Vector3 v=tf::Vector3(x,y,z);
		 tf::Quaternion q(rx,ry,rz,rw);
		 tr.setOrigin( v );
		 tr.setRotation( q);

		 kfTf.push_back(make_pair(atoi(name.c_str()),tr));
		 dumpToFile(name,x,y,z,rx,ry,rz,rw,pathToOutputFile);
	 }

	 while (!flf.eof()){
			 string buffer;
			 vector <string> fields,nb;
			 std::getline(flf, buffer);

			 boost::algorithm::split_regex( fields, buffer,  boost::regex(" ")  );
			 if(fields.size()<2)
			 {cout << "end of file" << endl;
			 break;
			 }

			 string name=fields[0];

			 boost::algorithm::split_regex( nb,name,  boost::regex("-")  );

			 int kf_nb=atoi(nb[0].c_str());
			 string lf_nb=nb[1];

			 float x=atof(fields[1].c_str()),y=atof(fields[2].c_str()),z=atof(fields[3].c_str()),rx=atof(fields[4].c_str()),ry=atof(fields[5].c_str()),rz=atof(fields[6].c_str()),rw=atof(fields[7].c_str());

			 tf::Transform tr ;
			 tf::Vector3 v=tf::Vector3(x,y,z);
			 tf::Quaternion q(rx,ry,rz,rw);
			 tr.setOrigin( v );
			 tr.setRotation( q);

			 tf::Transform T_Kf_from_w;
			 for(auto ptr=kfTf.begin();ptr!=kfTf.end();ptr++){
				 if(kf_nb==(*ptr).first){
					 T_Kf_from_w=(*ptr).second;
					 break;
				 }
			 }
			 tf::Transform T_lf_from_w=tr*T_Kf_from_w;

			 lfTf.push_back(make_pair(atoi(lf_nb.c_str()),T_lf_from_w));
			 tf::Vector3 v2=T_lf_from_w.getOrigin();
			 tf::Quaternion q2(T_lf_from_w.getRotation().x(),T_lf_from_w.getRotation().y(),T_lf_from_w.getRotation().z(),T_lf_from_w.getRotation().w());

			 dumpToFile(lf_nb,v2.x(),v2.y(),v2.z(),q2.x(),q2.y(),q2.z(),q2.w(),pathToOutputFile);
		 }

	 return;

}


double estimatePose3DAndError(vector<tf::Transform> tfv, vector<tf::Transform> tfgtv, cv::Mat & M ){

	int NB_POINT_POSE_ESTIMATION=4000;

	cv::Mat P,Pgt,Inliers;

	M=cv::Mat::zeros(3,4,CV_32F);

	P=cv::Mat::ones(3,tfv.size(),CV_32F);
	Pgt=cv::Mat::ones(3,tfv.size(),CV_32F);

	cout << "Initialization" << endl;

	for(int i=0; i<tfv.size(); i++){
		P.at<float>(0,i)=tfv[i].getOrigin().x();
		P.at<float>(1,i)=tfv[i].getOrigin().y();
		P.at<float>(2,i)=tfv[i].getOrigin().z();

		Pgt.at<float>(0,i)=tfgtv[i].getOrigin().x();
		Pgt.at<float>(1,i)=tfgtv[i].getOrigin().y();
		Pgt.at<float>(2,i)=tfgtv[i].getOrigin().z();


	}

	cout << "pose estimation" << P.size() << " " << Pgt.size() << " " <<  endl;


	cv::Mat src(1, tfv.size(), CV_32FC3);
	cv::Mat dest(1, tfv.size(), CV_32FC3);
	cv::Mat est;
	vector<uchar> outl;


	int OFFSET=100;
	for (int i = 0; i < NB_POINT_POSE_ESTIMATION; i++) {
		src.ptr<Point3f> ()[i] = Point3f(float(tfv[OFFSET+i].getOrigin().x()),float(tfv[OFFSET+i].getOrigin().y()),float(tfv[OFFSET+i].getOrigin().z()));
		dest.ptr<Point3f> ()[i] = Point3f(tfgtv[OFFSET+i].getOrigin().x(),tfgtv[OFFSET+i].getOrigin().y(),tfgtv[OFFSET+i].getOrigin().z());
	}

	cout << src.colRange(1,5)  << endl;
	cout << dest.colRange(1,5)  << endl;

	cv::estimateAffine3D(src, dest, est, outl,3,0.95);

	cout << "pose estimated "<< src.size() << " " << dest.size()  << endl;
	cout << est << endl;

	Mat P2=cv::Mat::ones(4,tfv.size(),CV_32F);
	for(int i=0; i<tfv.size(); i++){
			P2.at<float>(0,i)=tfv[i].getOrigin().x();
			P2.at<float>(1,i)=tfv[i].getOrigin().y();
			P2.at<float>(2,i)=tfv[i].getOrigin().z();
	}

	Mat Pprime=M*P2;

	cout << Pprime.colRange(1,5)  << endl;
	cout << Pgt.colRange(1,5)  << endl;


	cout << "Prep for err calc" << endl;

	Mat E=(Pprime-Pgt).mul((Pprime-Pgt));

	cout << E.colRange(1,5)  << endl;

	double err=0;

	for(int i=0;i<E.cols;i++){
		err+=E.at<float>(0,i)+E.at<float>(1,i)+E.at<float>(2,i);
	}

	return err;

}



int main(int argc, char* argv[])
{
	ros::init(argc,argv,"Error calculator");

	cout << argc << " " << argv[0] << endl;

	ros::NodeHandle nh_err;

	vector<tf::Transform> tfv,tfgtv;

	//parseFile(tfv,tfgtv);
	lesserFramePoseConverter();

//	cout << "Parsed" << tfv.size() << " " << tfgtv.size() << endl;
//
//	cv::Mat M;
//
//	cout.precision(15);
//
//	cout << "Err per point is " << estimatePose3DAndError(tfv, tfgtv, M ) << endl;

	//ros::Subscriber err_sub = nh_err.subscribe("err_calc", 1000, errCallback);

	//ros::spin();

}
