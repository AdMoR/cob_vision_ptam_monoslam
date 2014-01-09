/*
 * rgbd_line.h
 *
 *  Created on: Jan 7, 2014
 *      Author: rmb-am
 */

#ifndef RGBD_LINE_H_
#define RGBD_LINE_H_

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

inline void getMatrixFromFile(cv::Mat& d, std::string d_name){

	cv::FileStorage fs(d_name.c_str(), cv::FileStorage::READ);
	fs["d"] >> d;

}

int isIncorrect(cv::Mat img, int i, int j,int kernelSize=3){

	//We use a lbp to know where the invalid pixels are
	int lbp=0;

	assert(kernelSize%2==1);


	for(int k=-floor(kernelSize/2);k<=floor(kernelSize/2);k++){
		for(int l=-floor(kernelSize/2);l<=floor(kernelSize/2);l++){
			if(i+k>=0 && i+k<img.rows && j+l>=0 && j+l<img.cols && !(k==0 && l==0)){
				if( img.at<float>(i+k,j+l)==0)
					lbp |= 1 << (l+k+2);
			}
		}
	}

	return lbp;
}

void validPixelMap(cv::Mat& depth_img, cv::Mat& map){

	map=cv::Mat::zeros(depth_img.rows,depth_img.cols,CV_32F);

	for(int i=0;i<depth_img.rows;i++){
		for(int j=0;j<depth_img.cols;j++){
			if(depth_img.at<float>(i,j)==0)
				map.at<float>(i,j)=-1;
			else{
				map.at<float>(i,j)=isIncorrect(depth_img,i,j);
			}
		}
	}

}

bool getNextPixel(cv::Mat& img, int i,int j,int x_direction,int y_direction,int ret,int max_search,int search_depth=0){
	if(i+x_direction>=0 && i+x_direction<img.rows && j+y_direction>=0 && j+y_direction<img.cols && search_depth<max_search ){
		if(img.at<float>(i+x_direction,j+y_direction)!=0){ //Next neighbor is a correct => return this value
			ret=img.at<float>(i+x_direction,j+y_direction);
			return true;
		}
		else //We continue in the same direction
			return getNextPixel(img, i+x_direction,j+y_direction,x_direction,y_direction, ret,max_search,search_depth+1);
	}
	else
		return false;
}

bool getOEScore(cv::Mat& ROI,double threshold){

	//std::cout << ROI.size() << std::endl;

	if(!(ROI.cols==3))
		return false;

	double maxi=0;

	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			if(!(i==1 && j==1) && abs(ROI.at<float>(1,1)-ROI.at<float>(i,j))>maxi)
				maxi=abs(ROI.at<float>(1,1)-ROI.at<float>(i,j));
		}
	}
	//std::cout << maxi << " maxi <<>> thres" << threshold*ROI.at<float>(1,1) << endl;

	return maxi>threshold*ROI.at<float>(1,1);
}

void getOccluding(cv::Mat& depth_img, cv::Mat& validity_map, cv::Mat&o_out, cv::Mat& nx_out,cv::Mat& ny_out, int max_search=100, double threshold=0.04, bool time_measure=false){

	o_out=cv::Mat::zeros(depth_img.rows,depth_img.cols,CV_8U);
	nx_out=cv::Mat::zeros(depth_img.rows,depth_img.cols,CV_32F);
	ny_out=cv::Mat::zeros(depth_img.rows,depth_img.cols,CV_32F);


	double durationB = cv::getTickCount();

	for(int i=1;i<depth_img.rows-1;i++){
			for(int j=1;j<depth_img.cols-1;j++){
				//std::cout << i << " " << j << std::endl;
				bool loop=true; // Used to break the loops if a pixel cannot be saved.
				if(validity_map.at<float>(i,j)==-1){
					o_out.at<unsigned char>(i,j)=0;
					//d_out.at<float>(i,j)=0;
				}
				else{
					int lbp=validity_map.at<float>(i,j),v=0;
					if(lbp>0){

						//Find where the "holes" are
						for(int k=-1;k<=1&&loop;k++){
							for(int l=-1;l<=1&&loop;l++){
								if(lbp & (1 << (l+k+2)) ){ //We find in which direction the hole is
									if(getNextPixel(depth_img,i,j,k,l,v,max_search,0)){
										//We found a correct pixel
										depth_img.at<float>(i+k,j+l)=v;
										//validity_map.at<float>(i,j)=0;
									}
									else{
										//We couldn't find a valid pixel, we exit the search
										loop=false;
										//cout << "pixel " << i << " " << j << " is lost" << endl;
										o_out.at<unsigned char>(i,j)=0;

									}
								}
							}
						}

						//cv::Mat ROI=depth_img.colRange(j-1,j+2).rowRange(i-1,i+2),xout,yout;

//						if(getOEScore(ROI,threshold))
//							o_out.at<unsigned char>(i,j)=255;
					}
					else{
						cv::Mat ROI=depth_img.colRange(j-1,j+2).rowRange(i-1,i+2),xout,yout;
						if(getOEScore(ROI,threshold)){
							o_out.at<unsigned char>(i,j)=255;
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

	cv::Sobel(depth_img,nx_out,5,1,0,3);
	cv::Sobel(depth_img,ny_out,5,0,1,3);
	float mx=mean(nx_out).val[0], my=mean(ny_out).val[0];
	cout << mx << " " << my << endl;

	for(int i=1;i<depth_img.rows-1;i++){
				for(int j=1;j<depth_img.cols-1;j++){
					if(validity_map.at<float>(i,j)!=0 || o_out.at<unsigned char>(i,j)==255){
						nx_out.at<float>(i,j)=mx+1;
						ny_out.at<float>(i,j)=my+1;
					}
				}
	}

	if(time_measure){
		std::cout << "duration of hc is " << (cv::getTickCount()-durationB)/(cv::getTickFrequency()) <<std::endl;
		durationB = cv::getTickCount();
	}
}


#endif /* RGBD_LINE_H_ */
