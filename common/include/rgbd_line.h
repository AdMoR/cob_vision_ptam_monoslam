/*
 * rgbd_line.h
 *
 *  Created on: Jan 7, 2014
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
#include <tf/tf.h>
#include <ros/ros.h>

using namespace std;




inline int isIncorrect(cv::Mat img, int i, int j,int kernelSize=3){

	assert(i>0 && j>0 && i<img.rows-1 && j<img.cols-1);

	//We use a lbp to know where the invalid pixels are
	int lbp=0;

	assert(kernelSize%2==1);

	int u=0;
	for(int k=-floor(kernelSize/2);k<=floor(kernelSize/2);k++){
		for(int l=-floor(kernelSize/2);l<=floor(kernelSize/2);l++){
			if(!(k==0 && l==0) && img.at<float>(i+k,j+l)==0){
				lbp |= 1 << (u);

			}

			u++;
		}
	}

	//cout << lbp << " lbp" << endl;
	return lbp;
}

inline void validPixelMap(cv::Mat& depth_img, cv::Mat& map,float lower_threshold=0.1){

	//Check the pixels with invalid neighbours (depth = 0 )

	map=cv::Mat::zeros(depth_img.rows,depth_img.cols,CV_32S);

	for(int i=0;i<depth_img.rows;i++){
		for(int j=0;j<depth_img.cols;j++){
			if(i==0 || j==0 || i==depth_img.rows-1 || j==depth_img.cols-1){
				map.at<int>(i,j)=-1;
			}
			else{
				if(depth_img.at<float>(i,j)<=lower_threshold)
					map.at<int>(i,j)=-1;
				else
					map.at<int>(i,j)=isIncorrect(depth_img,i,j);
			}
		}
	}


}

inline bool getNextPixel(cv::Mat& img,cv::Mat& val, int i,int j,int x_direction,int y_direction,float& ret,int max_search,int search_depth=0){
	if(i+x_direction>=0 && i+x_direction<img.rows && j+y_direction>=0 && j+y_direction<img.cols && search_depth<max_search ){
		if(val.at<int>(i+x_direction,j+y_direction)!=-1){ //Next neighbor is a correct => return this value
			ret=img.at<float>(i+x_direction,j+y_direction);
			return true;
		}
		else //We continue in the same direction
			return getNextPixel(img,val, i+x_direction,j+y_direction,x_direction,y_direction, ret,max_search,search_depth+1);
	}
	else
		return false;
}


inline void stats(int i, int j,int val_oc,cv::Mat& ROI,cv::Mat& validity,cv::Mat& depth){

	assert(ROI.cols==3 && validity.cols==3 && depth.cols==3);


	ofstream fs;
	fs.open("/home/rmb-am/Slam_datafiles/line_rgbd.txt",ios::app);
	assert(fs.is_open());
	fs << val_oc << "located in " << i << " " << j << std::endl;
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			fs << ROI.at<float>(i,j) << " ";
		}
	}
	fs << std::endl;
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			fs << depth.at<float>(i,j) << " ";
		}
	}
	fs << std::endl;
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			fs << validity.at<int>(i,j) << " ";
		}
	}
	fs << std::endl;
	fs.close();
}

//Occluding Edge score
inline float getOEScore(cv::Mat& ROI){

	if(!(ROI.cols==3))
		return false;

	double maxi=0;

	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			if(!(i==1 && j==1) && abs(ROI.at<float>(1,1)-ROI.at<float>(i,j))>abs(maxi))
				maxi=(ROI.at<float>(1,1)-ROI.at<float>(i,j));
		}
	}
	//std::cout << maxi << " maxi <<>> thres" << threshold*ROI.at<float>(1,1) << endl;

	return maxi;
}

inline void getOccluding(cv::Mat& depth_img, cv::Mat& validity_map, cv::Mat&o_out, cv::Mat& nx_out,cv::Mat& ny_out, bool debug=false, int max_search=100, double threshold=0.04, bool time_measure=false){

	//Find the occluding pixels from the validity map and the depth image

	o_out=cv::Mat::zeros(depth_img.rows,depth_img.cols,CV_8UC2);
	nx_out=cv::Mat::zeros(depth_img.rows,depth_img.cols,CV_32F);
	ny_out=cv::Mat::zeros(depth_img.rows,depth_img.cols,CV_32F);


	double durationB = cv::getTickCount();

	for(int i=1;i<depth_img.rows-1;i++){
			for(int j=1;j<depth_img.cols-1;j++){
				bool loop=true; // Used to break the loops if a pixel cannot be saved.
				if(validity_map.at<int>(i,j)==-1){
					o_out.at<cv::Vec2b>(i,j)=cv::Vec2b(0,0);
					//d_out.at<float>(i,j)=0;
				}
				else{
					int lbp=validity_map.at<int>(i,j);
					float v=0;
					cv::Mat ROI=depth_img.colRange(j-1,j+2).rowRange(i-1,i+2);//, vROI=validity_map.colRange(j-1,j+2).rowRange(i-1,i+2), iROI=depth_img.colRange(j-1,j+2).rowRange(i-1,i+2).clone();
					if(lbp>0){
						int u=0;
						//Find where the "holes" are
						for(int k=-1;k<=1&&loop;k++){
							for(int l=-1;l<=1&&loop;l++){
								if(lbp & (1 << (u)) ){ //We find in which direction the hole is
									//std::cout << lbp << " lbp and bit "<< u << " : " << (int)(lbp & (1 << (u))) << std::endl;
									if(getNextPixel(depth_img,validity_map,i+k,j+l,k,l,v,max_search,0)){
										//We found a correct pixel
										ROI.at<float>(1+k,1+l)=v;
									}
									else{
										//We couldn't find a valid pixel, we exit the search and label as negative
										loop=false;
										//o_out.at<unsigned char>(i,j)=0;

									}
								}
								u++;
							}
						}

						//cv::Mat ROI=depth_img.colRange(j-1,j+2).rowRange(i-1,i+2),xout,yout;
						float score = getOEScore(ROI);
						if(loop && abs(score)>threshold*ROI.at<float>(1,1)){
							//if(debug) stats( i, j,getOEScore(ROI,threshold),ROI,vROI,iROI);
//							if(score>0)
//								o_out.at<cv::Vec2b>(i,j)=cv::Vec2b(score,0);
//							else
//								o_out.at<cv::Vec2b>(i,j)=cv::Vec2b(0,-score);
						}
					}
					else{
						//cv::Mat ROI=depth_img.colRange(j-1,j+2).rowRange(i-1,i+2),xout,yout;
						float score = getOEScore(ROI);
						if(abs(score)>threshold*ROI.at<float>(1,1)){
							//if(debug) stats( i, j,getOEScore(ROI,threshold),ROI,vROI,iROI);
							if(score>0)
								o_out.at<cv::Vec2b>(i,j)=cv::Vec2b(score,0);
							else
								o_out.at<cv::Vec2b>(i,j)=cv::Vec2b(0,-score);
							//d_out.at<float>(i,j)=0;
						}
					}
				}
		}
	}

	if(time_measure){
		std::cout << "duration of oe is " << (cv::getTickCount()-durationB)/(cv::getTickFrequency()) <<std::endl;
		durationB = cv::getTickCount();
	}


	if(time_measure){
		std::cout << "duration of hc is " << (cv::getTickCount()-durationB)/(cv::getTickFrequency()) <<std::endl;
		durationB = cv::getTickCount();
	}
}


inline void buildHistDescriptor(vector<int>& d_samples, vector<float>&histogram,int vector_start=0,int step=30,int start=65,int classes=7,int weigth=10,bool debug=false){

	//Create depth descriptors
	//vector_start is used to complete the descriptor with others info like saturation or appearance

	float n=(float)d_samples.size()*weigth+classes;

	for(int i=0;i<classes;i++)
		histogram.push_back(1);

	if(debug) cout << d_samples.size() << " : this is size" << endl;

	for(auto it = d_samples.begin();it!=d_samples.end();it++){
		if((*it)<start)
			histogram[vector_start+0]+=weigth;
	}
	histogram[vector_start+0]/=n;

	for(int i=1;i<classes-1;i++){
		for(auto it = d_samples.begin();it!=d_samples.end();it++){
				if(start+step*(i-1)<(*it) && (*it)<start+i*step)
					histogram[vector_start+i]+=weigth;
			}
		histogram[vector_start+i]/=n;
	}


	for(auto it = d_samples.begin();it!=d_samples.end();it++){
		if((*it)>start+(classes-2)*step)
			histogram[vector_start+classes-1]+=weigth;
	}
	histogram[vector_start+classes-1]/=n;

	if(debug){
		ofstream os;
		os.open("/home/rmb-am/Slam_datafiles/descriptor_rgbd.txt",ios::app);
		for(int i=0;i<classes;i++)
			os << histogram[vector_start+i] << " ";
		os << endl;
		os.close();
	}

}

inline void change2DTo3D(cv::Point& p, double z, cv::Point3f& out, double f_x=531.15,double f_y=531.15,double image_size_px=240,double image_size_py=320){

	float x=(p.x-image_size_px)/f_x*(z/1000),y=(p.y-image_size_py)/f_y*(z/1000);
	out=cv::Point3f(y,x,z/1000);

}

inline double myMax(cv::Mat& ROI, int channel){

	assert(channel < ROI.channels());

	unsigned char max = 0  ;

	for( auto ptr = ROI.begin<cv::Vec2b>(); ptr!=ROI.end<cv::Vec2b>();ptr++){
		if((*ptr)[channel] > max)
			max=(*ptr)[channel];

	}

	return max;

}

inline void myMinMax(cv::Mat& ROI, double& min, double& max){

	assert(ROI.depth()==CV_32F);

	double local_min=10000,local_max=0;

	for( auto ptr = ROI.begin<float>(); ptr!=ROI.end<float>();ptr++){
			if((*ptr) > local_max)
				local_max=(*ptr);
			if((*ptr)<local_min && (*ptr)!=0)
				local_min=(*ptr);
		}

	min=local_min;
	max=local_max;
}

inline bool findDepthDifference(cv::Mat& rgb_img,cv::Mat sat_map, cv::Mat& d_img, cv::Mat& oe_map, cv::Point begin, cv::Point end,vector<pair<int,int> >& lineB, int nb_estimation, vector<float>& descriptor, vector<cv::Point3f>& point_vec,bool b_mode=false,int search_size=1,int type=-1,bool debug=false){

	assert(nb_estimation>=2 && type!=-1);
	int count=0;
	vector<int> dd_collector,s_collector;

	for(int i=0;i<nb_estimation;i++){

		cv::Point pt;
		if(!b_mode)
			pt = ((float)i/(float)nb_estimation) * begin + (1.-(float)i/(float)nb_estimation) * end;
		else{
			pair<int,int> p = lineB[rand()%(lineB.size())];
			pt=cv::Point(p.second,p.first);
		}


		double min,max;
		cv::Point minl,maxl;

		if(pt.x > search_size && pt.x<d_img.rows-(search_size+1) && pt.y > search_size && pt.y<d_img.cols-(search_size+1)){
			cv::Mat ROI = oe_map.rowRange(pt.x-search_size,pt.x+search_size+1).colRange(pt.y-search_size,pt.y+search_size+1);
			max = myMax(ROI,type-1);
			if(debug) cout << "max : " << max << endl;

			if(max>40){ //If <40, this point is not on the edge so we dont include it.
				count+=1;
				dd_collector.push_back((int)max);
				cv::Mat dROI = d_img.rowRange(pt.x-search_size,pt.x+search_size+1).colRange(pt.y-search_size,pt.y+search_size+1);
				myMinMax(dROI,min,max);
				cv::Point3f loc;//=cv::Vec3f(pt.y,pt.x,max/1000);
				if(type==1)
					change2DTo3D(pt,max,loc,522.5,523.5,258,318);
				else
					change2DTo3D(pt,min,loc,522.5,523.5,258,318);
				point_vec.push_back(loc);
				cv::Mat sROI = sat_map.rowRange(pt.x-search_size,pt.x+search_size+1).colRange(pt.y-search_size,pt.y+search_size+1);
				for(auto it=sROI.begin<unsigned char>();it!=sROI.end<unsigned char>();it++)
					s_collector.push_back((*it));
			}
		}

	}
	if(count>=1){
		buildHistDescriptor(dd_collector,descriptor);
		buildHistDescriptor(s_collector,descriptor,7,40,25,7,10,false);

		return true;
	}
	else{
		cout << "WARNING : DESCRIPTOR UNAVAILABLE ! Only " << count << "values "<< endl;
		return false;
	}

}



