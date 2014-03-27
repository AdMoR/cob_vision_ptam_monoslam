/*
 * random_tree_rgbd.cpp
 *
 *  Created on: Nov 21, 2013
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
#include <boost/algorithm/string/regex.hpp>
#include "dataset_builder.cpp"
#include "utilities.h"
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include <tr1/unordered_map>
#include <omp.h>

//using namespace tr1;
using namespace Eigen;
using namespace std;
using namespace cv;


struct CameraPoseHypothesis{

//Structure representing a camera pose hypothesis
//We store also the error and the inliers of the pose

public:
	bool is_out;
	int number;
	SE3 pose;
	double error,error_line;
	vector<Vector3d> loc_inliers;
	vector<Vector3d> glo_inliers;

	CameraPoseHypothesis(){
		is_out=false;
		number=-1;
		pose=SE3();
		error=0;
		loc_inliers=vector<Vector3d>();
		glo_inliers=vector<Vector3d>();
	};

	//~CameraPoseHypothesis();
};

struct SplitNodeParam{
	int delta1,delta2,channel1,channel2,threshold;

	SplitNodeParam(int d1,int d2,int c1,int c2,int t):delta1(d1),delta2(d2),channel1(c1),channel2(c2),threshold(t){};
	SplitNodeParam():delta1(0),delta2(0),channel1(0),channel2(0),threshold(0){};
};

class SplitNode{

public:
	int NB_RAND_PARAM;
	int MAX_DEPTH;
	bool TRAINING_MODE;
	bool use_bagging;

	bool isLeaf;
	bool has_leaf_pose;
	int depth_;

	SplitNodeParam snp;
	Vector3d mode_;
	Matrix<double,4,4> leaf_pose_;
	double quality;
	SplitNode* left_;
	SplitNode* right_;

	//Bagging variable
	vector<int> bag;

	//Variable relative to the datas, each pixel has its own rgbd image and 3D label
	vector<vector<Mat> > color_img;
	vector<Mat> d_img;
	vector<Point> pixelSet;
	vector<Vector3d> labels;
	Vector3d mean_label;


	SplitNode(vector<Point>& pxSet, vector<Vector3d>& lb,vector<vector<Mat> >& rgb, vector<Mat>& d, int depth, int random_param_per_node,int max_depth,bool training,bool bagging):use_bagging(bagging),TRAINING_MODE(training),pixelSet(pxSet),labels(lb),color_img(rgb),d_img(d),depth_(depth),MAX_DEPTH(max_depth),quality(0),NB_RAND_PARAM(random_param_per_node),mean_label(Vector3d(0,0,0)){
		isLeaf=(depth>=MAX_DEPTH || labels.size()<10);
		has_leaf_pose=false;
		assert((!training) || labels.size()!=0);
		if(training)
			mean_of_vectors(labels,mean_label);
		for(int i=0;i<sqrt(pixelSet.size());i++)
			bag.push_back(rand()%(pixelSet.size()));
		if(isLeaf)
			leaf_pose_=SE3().matrix();
	}

	void operator=(const SplitNode& s){
		isLeaf=s.isLeaf;
		depth_=s.depth_;
		snp=s.snp;
		left_=s.left_;
		right_=s.right_;
		color_img=s.color_img;
		d_img=s.d_img;
		pixelSet=s.pixelSet;
		labels=s.labels;
		NB_RAND_PARAM=s.NB_RAND_PARAM;
		MAX_DEPTH=s.MAX_DEPTH;
	}


	//Defines the feature value on a pixel
	bool rgbd_feature(vector<Mat> & color_img,Mat & depth_img,int& delta1,int& delta2,int& channel1,int& channel2,int& threshold,int& x,int& y){

		double duration = cv::getTickCount();
		float depth_of_px = depth_img.at<float>(x,y);
		int rows=color_img.begin()->rows,cols=color_img.begin()->cols;

		if(color_img.begin()->size()==depth_img.size() && x>0 && y>0 && x<rows && y<cols && color_img.size()==3){
			int deltafirst,deltasecond;


			//Prepare
			deltafirst=delta1/depth_of_px;
			deltasecond=delta2/depth_of_px;

			//cout << "delta " << (getTickCount()-duration)/cv::getTickFrequency() << endl;
			duration = cv::getTickCount();


			//Check borders
			int xfirst=x+deltafirst,yfirst=y+deltafirst,xsecond=x+deltasecond,ysecond=y+deltasecond;


			if(x+deltafirst<0)
				xfirst=0;
			else{
				if(x+deltafirst>rows)
					xfirst=rows-1;
			}


			if(y+deltafirst<0)
				yfirst=0;
			else{
				if(y+deltafirst>cols)
					yfirst=cols-1;
				}


			if(x+deltasecond<0)
				xsecond=0;
			else{
				if(x+deltasecond>rows)
					xsecond=rows-1;}


			if(y+deltasecond<0)
				ysecond=0;
			else{
				if(y+deltasecond>cols)
					ysecond=cols-1;
			}

			//Ret feature
			return (color_img[channel1].at<unsigned char>(xfirst,yfirst)-color_img[channel2].at<unsigned char>(xsecond,ysecond))>threshold;


		}
		else{
			cout << "ERROR : problem in size of images !" << x << " " << y << " " << color_img.begin()->size() << " " << depth_img.size() << endl;
			exit(1);}
	}

	//Function which estimates the Variance reduction (best split  of labels)
	double findVariancereduction(vector<Vector3d>& myLabel, vector<bool>& left_or_right){

		vector<Vector3d> current_labels;

		current_labels=labels;

		assert(left_or_right.size()==current_labels.size());

		Vector3d mean=Vector3d(0,0,0),mean_left=Vector3d(0,0,0),mean_right=Vector3d(0,0,0);
		double sum=0,sum_left=0,sum_right=0;
		int size_right=0,size_left=0;

		//Get the mean of each members
		for(unsigned int i=0;i<current_labels.size();i++){
			//mean+=labels[i];
			if(left_or_right[i]){
				size_right+=1;
				mean_right+=current_labels[i];
			}
			else{
				size_left+=1;
				mean_left+=current_labels[i];
			}

		}

		mean=mean_label;

		if(size_left!=0)
			mean_left/=size_left;
		if(size_right!=0)
			mean_right/=size_right;

		//Build the variance reduction
		for(unsigned int i=0;i<current_labels.size();i++){
			sum+=(current_labels[i]-mean).squaredNorm();
			if(left_or_right[i]){
				if((current_labels[i]-mean_right).squaredNorm()<10000)
					sum_right+=(current_labels[i]-mean_right).squaredNorm();
			}
			else{
				if((current_labels[i]-mean_left).squaredNorm()<10000)
					sum_left+=(current_labels[i]-mean_left).squaredNorm();
			}
		}

		//Return the variance reduction
		return (sum-sum_left-sum_right); //WARNING : was before 1/n * ( sum total - sum right - sum left )
	}

	//Create the requested number of splitnode random param
	void getRandomparameter(vector<SplitNodeParam>& paramVector, int nb){
		for(int i=0;i<nb;i++){
			SplitNodeParam snp = SplitNodeParam(1000*(rand()%256-128),1000*(rand()%256-128),rand()%3,rand()%3,rand()%64-32);
			paramVector.push_back(snp);
		}
	}

	bool getDirection(int i,Point p){
		//True is right and false is left
		if(rgbd_feature(color_img[i],d_img[i],snp.delta1,snp.delta2,snp.channel1,snp.channel2,snp.threshold,p.x,p.y))
			return true;
		else
			return false;
	}

	//Function used to find the best parameter in a set for a given splitnode
	void findBestSplit(){
		cout << "start best split decision"<<endl;

		vector<SplitNodeParam> snpv;
		vector<Point> f_left,f_right;
		vector<Vector3d> f_left_label,f_right_label;
		vector<Mat> final_right_depth_img,final_left_depth_img;
		vector<vector<Mat> > final_right_rgb_img,final_left_rgb_img;
		SplitNodeParam bestParam;
		double best_var_red=0;
		double duration=getTickCount();
		getRandomparameter(snpv,NB_RAND_PARAM);
		cout << "random param " << (getTickCount()-duration)/cv::getTickFrequency() << endl;
		duration = cv::getTickCount();
		double durationb = cv::getTickCount();

		//Separate the pixels in left and right subsets
		for(auto ptr=snpv.begin();ptr!=snpv.end();ptr++){
			snp=(*ptr);
			//cout << "parameter : " << ptr->delta1 << " " << ptr->delta2 << endl;
			vector<bool> left_or_right;
			vector<Point> left,right;
			vector<Vector3d> left_label,right_label,myLabel;
			vector<Mat> r_d,l_d;
			vector<vector<Mat> > r_rgb,l_rgb;

			duration = cv::getTickCount();
			for(auto i=0;i<pixelSet.size();i++){
				durationb = cv::getTickCount();
				bool direction=rgbd_feature(color_img[i],d_img[i],snp.delta1,snp.delta2,snp.channel1,snp.channel2,snp.threshold,pixelSet[i].x,pixelSet[i].y);
				//cout << "monopx get dir" << (getTickCount()-durationb)/cv::getTickFrequency() << endl;
				durationb = cv::getTickCount();
				//cout << "on pixel number "<< i << " decision is " << direction << endl;
				left_or_right.push_back(direction);
				if(direction){
					right.push_back(pixelSet[i]);
					right_label.push_back(labels[i]);
					r_rgb.push_back(color_img[i]);
					r_d.push_back(d_img[i]);
				}
				else{
					left.push_back(pixelSet[i]);
					left_label.push_back(labels[i]);
					l_rgb.push_back(color_img[i]);
					l_d.push_back(d_img[i]);
				}

				//	cout << "push back" << (getTickCount()-durationb)/cv::getTickFrequency() << endl;

			}

			//cout << "find directions "  << endl;
			duration = cv::getTickCount();


			double varRed=findVariancereduction(myLabel,left_or_right);

			cout << "find best var red "  << endl;
			duration = cv::getTickCount();

			if(varRed>best_var_red){

					cout << varRed << " current best varRed" << endl;
					cout << left.size() << "  left" << endl;
					cout << right.size() << "  right" << endl;
					bestParam=(*ptr);

					f_left=left;
					f_right=right;

					f_left_label=left_label;
					f_right_label=right_label;

					final_left_depth_img=l_d;
					final_left_rgb_img=l_rgb;

					final_right_depth_img=r_d;
					final_right_rgb_img=r_rgb;
					best_var_red=varRed;

					cout << "best match update " << (getTickCount()-duration)/cv::getTickFrequency() << endl;
					duration = cv::getTickCount();

			}


		}


		cout << "Best var red is : " << best_var_red << endl;
		if(f_left.size()!=0 && f_right.size()!=0){
			snp=bestParam;
			left_ = new SplitNode(f_left,f_left_label,final_left_rgb_img,final_left_depth_img,depth_+1,NB_RAND_PARAM,MAX_DEPTH,TRAINING_MODE,use_bagging);
			right_ = new SplitNode(f_right,f_right_label,final_right_rgb_img,final_right_depth_img,depth_+1,NB_RAND_PARAM,MAX_DEPTH,TRAINING_MODE,use_bagging);
		}
		else{
			//If the best split doesnt split the parameters, we label it as a leaf
			cout << "Warning : non maximal depth leaf created" << endl;
			isLeaf=true;

		}
		quality=best_var_red;
		cout << "end of children creation " << (getTickCount()-duration)/cv::getTickFrequency() << endl;

	}


//	void predict(int i, Vector3d& result){
//		int left=0,right=0;
//		if(!isLeaf){
//			if(getDirection(i,pixelSet[i]))
//				right_->predict(result);
//			else
//				left_->predict(result);
//		}
//		else
//			result=mode_;
//	}



};


double gaussian_val(double x,double sigma){
	return (1/(pow((double)2*M_PI,(double)0.5)*sigma)*exp(-0.5*pow(x/sigma,(double)2)));
}


void from2DTo3D(Point& p, double z,Vector3d& position, SE3& label, double f_x=531.15,double f_y=531.15,double image_size_px=240,double image_size_py=320){

	float x=(p.x-image_size_px)/f_x*(z/1000),y=(p.y-image_size_py)/f_y*(z/1000);
	Vector4d tmp = (label.matrix())*toHomogeneousCoordinates(Vector3d(x,y,z/1000));
	position = Vector3d(tmp(0),tmp(1),tmp(2));
//	ofstream os;
//	os.open("/home/rmb-am/Slam_datafiles/truc.dfre",ios::app);
//	os << "translation : " << label.translation()(0) << " " << label.translation()(1) << " " << label.translation()(2) << endl;
//	os << "point " << p.x << " " << p.y << " " <<  x << " " << y << " " << z/1000 << endl;
//	os << "gloPos : " << position(0) << " " << position(1) << " " << position(2) << endl << endl;
//	os << label.matrix() << endl;	os.close();

}

//Gaussina mean / mean shift
void get_gaussian_average(vector<Vector3d>& vv,Vector3d mean,Vector3d& avg){
	double total_w=1;
	avg=mean;
	if(vv.size()!=0){
		for(auto ptr=vv.begin();ptr!=vv.end();ptr++){
			Vector3d diff=((*ptr)-mean);
			double w=gaussian_val(diff.norm(),0.1);
			avg+=w*(*ptr);
			total_w+=w;
		}
		avg/=total_w;
	}

}

void get_mean_shift(SplitNode* n){

	//Estimate the mean
	Vector3d mean=n->mean_label;
	Vector3d diff=Vector3d(10,10,10);

	//We now apply a mean shift algorithm with a gaussian kernel with sigma=0,1m
	while(diff.norm()>0.1){
		Vector3d avg=Vector3d(0,0,0);
		get_gaussian_average(n->labels,mean,avg);
		diff=mean-avg;
		mean=avg;
	}

	n->mode_=mean;

}


//Not used : tested to see if we could have a more complex leaf model
void get_leaf_pose(SplitNode* n){

	vector<Vector3d> loc_w,glo_w;
	SE3 tf=SE3();


	for(int i = 0 ; i<n->pixelSet.size(); i++){
		Vector3d position=Vector3d(0,0,0),diff=n->labels[i]-n->mode_;
		from2DTo3D(n->pixelSet[i],n->d_img[i].at<float>(n->pixelSet[i].x,n->pixelSet[i].y),position,tf);
		double gau = gaussian_val(diff.norm(),1);
		loc_w.push_back(gau*position);
		glo_w.push_back(gau*diff);

	}


	solveP3P(tf,loc_w,glo_w,0,n->labels.size());


	n->leaf_pose_=tf.matrix();
	n->has_leaf_pose=true;

//	ofstream os;
//	os.open("/home/rmb-am/Slam_datafiles/leaf_pose.txt",ios::app);
//	os << tf.matrix() <<endl;
//	os.close();

}

//Given an image, it samples a given number of pixels
void getPixelSubset(Mat& img ,vector<Point> & pixelSet,vector<Vector3d>& labels, SE3& frame_label, int numberOfPixels,int maxDelta=0){

	
	int initial_size=pixelSet.size();

	if(maxDelta<0)
		maxDelta=-maxDelta;

	int randx,randy;
	while(pixelSet.size()!=initial_size+numberOfPixels){
		randx=floor(rand()%img.rows);
		randy=floor(rand()%img.cols);

		float d = img.at<float>(randx,randy);
		if(randx<img.rows-maxDelta && randy<img.cols-maxDelta && randx>maxDelta && randy>maxDelta && d>0.1){
			Point p=Point(randx,randy);
			pixelSet.push_back(p);
			Vector3d ptLabel;
			//cout << "before : " << ptLabel << " with px : "<< randx << " " << randy <<  endl;
			from2DTo3D(p,d,ptLabel,frame_label);
			//cout << "after : " << ptLabel << endl;
			labels.push_back(ptLabel);
		}
	}

}

//Main function to train a RGBD forest
void growTree(SplitNode* n){
	//Recursive method to grow the regression random tree
	if(!n->isLeaf){
		assert(n->labels.size()!=0 && n->color_img.size()!=0 && n->d_img.size()!=0 && n->pixelSet.size()!=0 );
		double duration = cv::getTickCount();
		n->findBestSplit();
		cout << "find best split duration " << (getTickCount()-duration)/cv::getTickFrequency() << endl;

		if(!n->isLeaf){
			growTree((n->left_));
			growTree((n->right_));
		}
		else{
			assert(n->labels.size()!=0);
			cout << "getting non final node mode" << endl;
			get_mean_shift(n);
			if(n->pixelSet.size()>3)
				get_leaf_pose(n);
		}
//		n->left_->isLeaf=true;
//		n->right_->isLeaf=true;
//		get_mean_shift(n->left_);
//		get_mean_shift(n->right_);
	}
	else{
		assert(n->labels.size()!=0);
		cout << "getting node mode" << endl;
		get_mean_shift(n);
//		if(n->pixelSet.size()>3)
//			get_leaf_pose(n);
	}
}

//Function used to save the forest to file
void writeNodes(SplitNode* n,ofstream& myfile){

	if(!n->isLeaf){
		myfile << n->depth_ << " " << n->isLeaf << " " << n->snp.delta1 << " " << n->snp.delta2 << " " << n->snp.channel1 << " " << n->snp.channel2 << " "<< n->snp.threshold << " " << n->quality << endl;
		writeNodes((n->left_),myfile);
		writeNodes((n->right_),myfile);
	}
	else{
		myfile << n->depth_ << " " << n->isLeaf << " " << n->mode_(0) << " " << n->mode_(1) << " " << n->mode_(2) << " " << n->has_leaf_pose << " " << n->leaf_pose_(0,0) <<" " << n->leaf_pose_(0,1)<< " " << n->leaf_pose_(0,2) << " " << n->leaf_pose_(0,3) << " " << n->leaf_pose_(1,0) <<" " << n->leaf_pose_(1,1)<< " " << n->leaf_pose_(1,2)<< " " << n->leaf_pose_(1,3)<< " " << n->leaf_pose_(2,0) << " " << n->leaf_pose_(2,1)<< " " << n->leaf_pose_(2,2)<< " " << n->leaf_pose_(2,3)<< " " << n->labels.size() << endl;
	}
}

void writeTree(SplitNode* n, string fileName){
	ofstream myfile;
	myfile.open (fileName, ios::app);
	if(myfile.is_open()){
		writeNodes(n,myfile);
	}
	myfile.close();
}

//Function used to load a forest from file
void readNode(SplitNode* n,ifstream& myfile,int est_depth){
	string buffer;
	vector <string> fields;


	std::getline(myfile, buffer);
	boost::algorithm::split_regex( fields, buffer,  boost::regex(" ")  );
	if(fields.size()!=8 && fields.size()!=6 && fields.size()!=19)	{
		cout << "ERROR : not the right number of args in the line" << endl;
		return;
	}

	string depth=fields[0],isLeaf=fields[1];
	if(est_depth==atoi(depth.c_str())){
		n->isLeaf=(bool)atoi(isLeaf.c_str());
		if(!n->isLeaf){
			string delta1=fields[2],delta2=fields[3],channel1=fields[4],channel2=fields[5],thresh=fields[6];
			n->snp=SplitNodeParam(atoi(delta1.c_str()),atoi(delta2.c_str()),atoi(channel1.c_str()),atoi(channel2.c_str()),atoi(thresh.c_str()));
			n->quality=atof(fields[7].c_str());
		}
		else{
			n->mode_=Vector3d(atof(fields[2].c_str()),atof(fields[3].c_str()),atof(fields[4].c_str()));
//			n->has_leaf_pose=(bool)atoi(fields[5].c_str());
//			if(n->has_leaf_pose)
//				n->leaf_pose_ << atof(fields[6].c_str()),atof(fields[7].c_str()),atof(fields[8].c_str()) , atof(fields[9].c_str()),
//								 atof(fields[10].c_str()),atof(fields[11].c_str()),atof(fields[12].c_str()) ,atof(fields[13].c_str()),
//								 atof(fields[14].c_str()),atof(fields[15].c_str()),atof(fields[16].c_str()) , atof(fields[17].c_str()),
//								 0,0,0,1;
//			if(n->leaf_pose_.norm()>100)
			n->has_leaf_pose=false;
		}
	}
	else{
		cout << "ERROR IN TREE CONSTRUCTION" << endl;
	}


	if(!n->isLeaf){
		n->left_ = new SplitNode(n->pixelSet,n->labels,n->color_img,n->d_img,n->depth_+1,n->NB_RAND_PARAM,n->MAX_DEPTH,n->TRAINING_MODE,n->use_bagging);
		n->right_ = new SplitNode(n->pixelSet,n->labels,n->color_img,n->d_img,n->depth_+1,n->NB_RAND_PARAM,n->MAX_DEPTH,n->TRAINING_MODE,n->use_bagging);
		readNode((n->left_),myfile,est_depth+1);
		readNode((n->right_),myfile,est_depth+1);
	}
}

void readTree(SplitNode* n,string fileName){

	ifstream myfile;
	myfile.open (fileName, ios::in);
	if(myfile.is_open()){
		cout << "file opened" << endl;
		readNode(n,myfile,0);
	}
	myfile.close();
}


//Main function used for coordinate prediction
void predict(SplitNode* n, Point& p,vector<Mat>& rgb, Mat& d,Vector3d& result){

	if(!n->isLeaf){
		//cout << n->depth_ << endl;
		if(n->rgbd_feature(rgb,d,n->snp.delta1,n->snp.delta2,n->snp.channel1,n->snp.channel2,n->snp.threshold,p.x,p.y)){
			predict(n->right_,p,rgb,d,result);
		}
		else{
			predict(n->left_,p,rgb,d,result);
		}
	}
	else{
//		if(n->has_leaf_pose){
//			Vector3d position;
//			SE3 zero = SE3();
//			from2DTo3D(p,d.at<float>(p.x,p.y),position,zero);
//			Vector4d diff = n->leaf_pose_*toHomogeneousCoordinates(position);
//			Vector3d diff3d=Vector3d(diff(0),diff(1),diff(2));
//			result=n->mode_+diff3d;
//			if(result.norm()>10000)
//				cout << endl << result << endl << position << n->leaf_pose_ << endl << endl;
//		}
//		else
			result=n->mode_;
	}
}

void forestPrediction(vector<SplitNode*>& forest, Point& p, vector<Mat>& rgb, Mat& d,Vector3d& result,SE3& current_prior,bool use_prior=false){

	vector<Vector3d> results;
	result=Vector3d(0,0,0);

	for(int i=0;i<forest.size();i++){
		results.push_back(Vector3d(0,0,0));
		predict(forest[i], p, rgb, d, results[i]);
	}

	if(use_prior){
		SE3 zero = SE3();
		Vector3d position = Vector3d(0,0,0);
		from2DTo3D(p,d.at<float>(p.x,p.y),position,zero);
		double sum=0;

		//ofstream os;
		//os.open("/home/rmb-am/Slam_datafiles/iiiiiiiiiii2.dfre",ios::app);
		//os << ">>"<<current_prior.matrix()*toHomogeneousCoordinates(position) << endl;
		for(int i =0;i<results.size();i++){
			Vector4d diff = toHomogeneousCoordinates(results[i])-current_prior.matrix()*toHomogeneousCoordinates(position);
			float w = gaussian_val(diff.norm(),1.0);
			result+=w*results[i];
			sum+=w;


			//os <<">>"<< results[i] << endl<< w <<endl;

		}

		if(sum!=0 || result.norm()!=0)
			result/=sum;
		else{
			result=Vector3d(0,0,0);
			for(int i = 0; i<results.size();i++)
				result+=results[i];
			result/=results.size();
		}
		//os << endl;
		//os.close();
	}
	else
		mean_of_vectors(results,result);

}


//In the prediction phase, we need to sample camera pose hypotheses with 3D points
void sample_cam_pos(vector<SplitNode*>& forest,vector<CameraPoseHypothesis>& pose_hypothesis,int number_of_samples, Mat& rgb_img, Mat& d_img, vector<pair<SE3,float> >& pose_prior, bool prior=false){

	vector<Point> pxSet;
	vector<Vector3d> locPxPos,gloPxPos;
	SE3 zero=SE3();
	vector<cv::Mat> channel_v;
	split(rgb_img,channel_v);


	SE3 frame_label=SE3(),previous_transform=SE3();
	int randx,randy;
	int numberOfPixels = 3*number_of_samples;
	map<float,int> h_count;
	for(int i =0; i<pose_prior.size();i++) h_count.insert(make_pair(pose_prior[i].second,0));
	
	int tries=0;

	//Sample points for the camera hypotheses
	while(pxSet.size()!=numberOfPixels){
		randx=floor(rand()%d_img.rows);
		randy=floor(rand()%d_img.cols);

		float prob;
		float proba = (float)((float)pxSet.size()/(float)numberOfPixels);
		if(prior)
		{
			//Change prior prior according to current pt number
			for(auto ptr = pose_prior.begin();ptr!=pose_prior.end();ptr++){
				if(proba < ptr->second){
					previous_transform=ptr->first;
					prob=ptr->second;
					break;
				}
			}
		}

		float d = d_img.at<float>(randx,randy);
		if(randx<d_img.rows-0 && randy<d_img.cols-0 && randx>0 && randy>0 && d>0.1){
			Point p=Point(randx,randy);
			Vector3d ptLabel;
			Vector3d res=Vector3d(0,0,0);
			
			
			from2DTo3D(p,d,ptLabel,frame_label);
			forestPrediction(forest,p,channel_v,d_img,res,previous_transform,prior);
			//cout << " ururururru" << res << "      " << ptLabel << endl << endl;
			//Keep a point only if it is not too far from prior
		    Vector4d diff=(toHomogeneousCoordinates(res)-previous_transform.matrix()*toHomogeneousCoordinates(ptLabel));
			
			if( (!prior) || diff.norm()<0.7 || tries>50){
				if(prior)
					h_count.find(prob)->second++;
				locPxPos.push_back(ptLabel);
				pxSet.push_back(p);
				gloPxPos.push_back(res);
				tries=0; // max 50 tries
			}
			else{
				tries++;
			}
		}
	}

		CameraPoseHypothesis cph2;
		cph2.pose=previous_transform;
		cph2.number=0;
		pose_hypothesis.push_back(cph2);

	//Estimate the transform between local and global
	for(int k=1;k<number_of_samples;k++){
		SE3 se;
		solveP3P(se,locPxPos,gloPxPos,3*k);

		CameraPoseHypothesis cph;
		cph.pose=se;
		cph.number=k;

		pose_hypothesis.push_back(cph);
	}

	int i=0;
	for(auto ptr = h_count.begin();ptr!=h_count.end();ptr++){
		cout << i << " has generated " << ptr->second << " hypotheses!" << endl;
		i++;
	}

}

//Clean the memory
void destroyNode(SplitNode* n){

	delete n;

}

void destroyTree(SplitNode* n){

	if(!(n->isLeaf)){
		destroyTree(n->left_);
		destroyTree(n->right_);
	}
	destroyNode(n);
}


void trainRegressionForest(vector<Mat>& rgb_img, vector<Mat>& depth_img, vector<Sophus::SE3>& image_pose, int px_per_frame,int random_parameters_per_node,int nb_tree,int max_depth,int training, String save,bool use_bagging=false){

	cout << rgb_img.size() << " " << depth_img.size() << " " << (*(rgb_img.begin())).size() << " " << (*(depth_img.begin())).size() << endl;
	assert(rgb_img.size()==depth_img.size() && (*(rgb_img.begin())).size()==(*(depth_img.begin())).size());


	//#pragma omp parallel shared(rgb_img,depth_img,image_pose,px_per_frame,random_parameters_per_node,nb_tree,training,save)
	for(int k=0;k<nb_tree;k++){
		stringstream ss; ss << "/home/rmb-am/Slam_datafiles/RGBDRegressionForest" << rand()%15 << ".rf";string save2=ss.str();
		//Build the datas to train the trees
		vector<Point> pixelSet;
		vector<Vector3d> labels;
		vector<vector<Mat> > color_img;
		vector<Mat> d_img;

		for(unsigned int i=0;i<rgb_img.size();i++){
			vector<Mat> channel_v;
			Mat lbp,lab;
			getPixelSubset(depth_img[i],pixelSet,labels,image_pose[i],px_per_frame,0);
			split(rgb_img[i],channel_v);
			//insert other Mat in channels_v HERE
			for(int l=0;l<px_per_frame;l++){
				color_img.push_back(channel_v);
				d_img.push_back(depth_img[i]);
				//labels.push_back(image_coord[i]);
			}
		}

		cout << "labels created " << endl;

		//Create the root of the tree
		SplitNode* root;
		cout << "I use bagging" << endl;
		root=new SplitNode(pixelSet,labels,color_img,d_img,0,random_parameters_per_node,max_depth,training,use_bagging);

		cout << "root created " << root->isLeaf << endl;

		//Start the growing
		double duration = getTickCount();
		growTree(root);
		cout << "grow one tree time " << (getTickCount()-duration)/getTickFrequency() << endl;
		cout << "tree finished" << endl;

		//Save the parameters
		writeTree(root,save2);

		destroyTree(root);
	}
}


//Binary metric on SE3
int se3_metric(SE3& candidate, SE3& GT, float threshold_translation=0.45, float threshold_angle=5){

	Matrix<double,3,3> diff_angle = GT.rotation_matrix()*candidate.rotation_matrix().inverse();

	Vector3d diff_translation = GT.translation()-candidate.translation();

	if(!(diff_angle(0,0)>cos(threshold_angle*M_PI/180)&&(diff_angle(0,0)>cos(threshold_angle*M_PI/180))&&(diff_angle(0,0)>cos(threshold_angle*M_PI/180))) && !(diff_translation.norm()<threshold_translation))
		return -3;
	if(!(diff_angle(0,0)>cos(threshold_angle*M_PI/180)&&(diff_angle(0,0)>cos(threshold_angle*M_PI/180))&&(diff_angle(0,0)>cos(threshold_angle*M_PI/180))))
		return -2;
	if(!(diff_translation.norm()<threshold_translation))
		return -1;
	return 1;


}

//Continuous metric on SE3
float myMatrixNorm(Matrix<double,4,4>& M1,Matrix<double,4,4>& M2,double w_angle=7, double w_tr=1){

	Matrix<double,3,3> ppkc ;
	ppkc << M1(0,0)-M2(0,0), M1(0,1)-M2(0,1), M1(0,2)-M2(0,2),
			M1(1,0)-M2(1,0), M1(1,1)-M2(1,1), M1(1,2)-M2(1,2),
			M1(2,0)-M2(2,0), M1(2,1)-M2(2,1), M1(2,2)-M2(2,2);

	double tr = Vector3d(M1(0,3)-M2(0,3),M1(1,3)-M2(1,3),M1(2,3)-M2(2,3)).norm();
	double theta = ppkc.norm();
	//cout << theta << " " << tr << endl;
	return w_angle*theta+w_tr*tr;

//	Matrix<double,3,3> ppkc,lmdp,id,theta ;
//		ppkc << M1(0,0), M1(0,1), M1(0,2),
//				M1(1,0), M1(1,1), M1(1,2),
//				M1(2,0), M1(2,1), M1(2,2);
//		lmdp << M2(0,0), M2(0,1), M2(0,2),
//				M2(1,0), M2(1,1), M2(1,2),
//				M2(2,0), M2(2,1), M2(2,2);
//		id.setIdentity();
//
//		theta= id - ppkc*lmdp.inverse();
//
//
//		double tr = Vector3d(M1(0,3)-M2(0,3),M1(1,3)-M2(1,3),M1(2,3)-M2(2,3)).norm();
//
//		cout << theta << " " << tr << endl;
//		return w_angle*theta.norm()+w_tr*tr;

}

//Second phase of the prediction, once we have hypotheses, we reject the worst and refine the best ones
void refine_phase(vector<SplitNode*>& forest, Mat& rgb_img, Mat& d_img, int log_initial_hypothesis,int number_of_sample_points, int returned_hypohesis, vector<CameraPoseHypothesis>& pose_models, vector<pair<SE3,float> >& pose_prior,bool prior_use=false,float inlier_threshold=0.70, bool debug=true){

	//Declare main variables
	int K=pow(2,log_initial_hypothesis);
	vector<CameraPoseHypothesis> pose_hypothesis;
	SE3 zero = SE3();
	vector<cv::Mat> channel_v;
	split(rgb_img,channel_v);

	//Sample hypothesis
	sample_cam_pos(forest,pose_hypothesis,K,rgb_img,d_img,pose_prior,prior_use);

	//Enter the main loop
	for(int i=0;i<log_initial_hypothesis-log(returned_hypohesis);i++){

		int j=0;

		for(auto ptr=pose_hypothesis.begin();ptr!=pose_hypothesis.end();ptr++){

			if(!(ptr->is_out)){
				j++;

				//Prepare the points
				vector<Point> pxSet;
				vector<Vector3d> gloPxPos,locPxPos;
				getPixelSubset(d_img,pxSet,locPxPos,zero,number_of_sample_points,0);

				Matrix<double,4,4> H = (*ptr).pose.matrix();

				for(int k=0;k<number_of_sample_points;k++){
					Vector3d res=Vector3d(0,0,0);
					forestPrediction(forest,pxSet[k],channel_v,d_img,res,(*ptr).pose,true);
					gloPxPos.push_back(res);
				}

			
				//do X - Hx on every pt


				//cout << endl << H << endl;
				for(int k=0;k<number_of_sample_points;k++){
					Vector4d diff=(toHomogeneousCoordinates(gloPxPos[k])-H*toHomogeneousCoordinates(locPxPos[k]));
					ptr->error+=diff.squaredNorm();
					//cout << diff.squaredNorm() << endl;
					if(diff.squaredNorm()<=inlier_threshold){
						ptr->loc_inliers.push_back(locPxPos[k]);
						ptr->glo_inliers.push_back(gloPxPos[k]);
					}
				}
			}

		}
		//delete the lower half of the map (biggest error)
		if(debug) cout << "delete half" << endl;
		int init_size=pose_hypothesis.size();
		if(init_size>returned_hypohesis){
			for(int i=0;i<init_size/2;i++){
				float biggest_error=0;
				int num_to_del=-1,num=0;
				for(auto it = pose_hypothesis.begin();it!=pose_hypothesis.end();it++){
					if(!(it->is_out) && it->error>biggest_error){
						biggest_error=it->error;
						num_to_del=num;
					}
					num++;
				}
				if(debug) cout << " i delete hypothsis " << pose_hypothesis[num_to_del].number <<  " with error " << pose_hypothesis[num_to_del].error << endl;
				pose_hypothesis.erase(pose_hypothesis.begin()+num_to_del);
			}
		}

		//Refine the pose of the remaining candidates
		if(debug) cout << "refine " << endl;
		for(auto ptr=pose_hypothesis.begin();ptr!=pose_hypothesis.end();ptr++){
			if(!(ptr->is_out)){
				if(debug) cout << endl<< "numero " << ptr->number << " has " << ptr->loc_inliers.size() << " inliers " << " and error : " << ptr->error << endl;
				if(debug) cout << ptr->pose.matrix() <<endl<<endl;
				solveP3P((*ptr).pose,ptr->loc_inliers,ptr->glo_inliers,0,(int)ptr->loc_inliers.size());
				ptr->glo_inliers.clear();
				ptr->loc_inliers.clear();

			}
		}


		if(debug) cout << endl << endl;
	}
	//Show the winners
	for(auto it = pose_hypothesis.begin();it != pose_hypothesis.end(); it++){
		cout << "Numero : " << it->number << " " << endl << it->pose.matrix() << endl << " with error " << it->error << endl;
		pose_models.push_back((*it));
	//		ofstream os;
	//		os.open("/home/rmb-am/Slam_datafiles/iiiiiiiiiii.dfre",ios::app);
	//		os << it->pose.matrix() << endl<<endl;
	//		os.close();
	}


}




int dsmain(int argc, char* argv[]){

	//Constant parameter used to train the forest
	const int nb_frame_to_train=35;//50
	const int nb_points_per_frame=3000;//2000;
	const int nb_random_param_per_node=5000;//5000;
	const int nb_of_trees=1;
	const int max_depth=17;
	const int interval_bw_two_frames=30;//70

	const bool use_lab=false;

	//The training (set training to true) uses a video input created with the different images of a rosbag
	//It selects one image every interval_bw_two_frames until nb_frame_to_train is reached
	//The images and their 6D labels are saved to a file
	//The trained trees are written to file in /home/rmb-am/Slam_datafiles/RGBDRegressionForestX.rf where X is a random number

	//The prediction takes images in a folder to predict their pose in 6D
	//It loads the trees RGBDRegressionForestX.rf until the number of required trees is reached
	//The results are written to file and use binary metrics to assess of the success of the relocalisation.
	const bool training = false;

	const string tree_saving_location="/home/rmb-am/Slam_datafiles/RGBDRegressionForest1.rf";
	const string img_saving_location="/home/rmb-am/Slam_datafiles/training_img2/";

	ros::init(argc, argv,"rgbd_random_tree");

	if(training){
		DatasetBuilder rgbd_training_set(interval_bw_two_frames,use_lab);
		cout << "Init the databuilder" << endl;
		boost::thread image_acquisition(boost::bind(&DatasetBuilder::init,boost::ref(rgbd_training_set),"/odometry_combined","/head_cam3d_frame","/camera/rgb/image_color","/camera/depth_registered/image_raw"));
		cout << "Wait for the training images" << endl;
		while(true){
			cout << rgbd_training_set.rgb_img.size() << endl;
			if(rgbd_training_set.rgb_img.size()>nb_frame_to_train-1 && rgbd_training_set.d_img.size()>nb_frame_to_train-1){
				image_acquisition.interrupt();
				break;
			}
			sleep(1);
		}
		cout << "Frames have been acquired : " << rgbd_training_set.rgb_img.size() <<" differents frames" << endl;
		for(unsigned int i=0;i<rgbd_training_set.rgb_img.size();i++)
			assert(rgbd_training_set.rgb_img[i].size()==rgbd_training_set.d_img[i].size());
		rgbd_training_set.remember_training(img_saving_location);
		trainRegressionForest(rgbd_training_set.rgb_img,rgbd_training_set.d_img,rgbd_training_set.pose_vector,nb_points_per_frame,nb_random_param_per_node,nb_of_trees,max_depth,training,tree_saving_location);
	}
	else{
		vector<SplitNode*> forest;
		DatasetBuilder empty(0,use_lab);
		vector<Point> zero=vector<Point>();
		vector<vector<cv::Mat> > channel_v;
		vector<Vector3d> nothing;
		for(auto ptr=empty.rgb_img.begin();ptr!=empty.rgb_img.end();ptr++){
			vector<Mat> channels;
			split((*ptr),channels);
			channel_v.push_back(channels);

		}

		//Create the root
		//cout << "root created " << root->isLeaf << endl;

		//Read the parameters
		for(int i = 0; i<nb_of_trees;i++){
			stringstream ss;
			ss << "/home/rmb-am/Slam_datafiles/read_a_forest_" << i <<".rt";
			//ss << "/home/rmb-am/Slam_datafiles/RGBDRegressionForest1.rf";
			forest.push_back( new SplitNode(zero,nothing,channel_v,empty.d_img,0,nb_random_param_per_node,max_depth,training,false) );
			readTree(forest[i],ss.str().c_str());
		}



		//Load data
		//empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img_long/","/home/rmb-am/Slam_datafiles/frame_labels.txt");
		//empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img_biiig/","/home/rmb-am/Slam_datafiles/frame_label_biiig.txt");
		//empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img_3/","/home/rmb-am/Slam_datafiles/labels20.txt");
		empty.load_from_training("/home/rmb-am/Slam_datafiles/validation_img/","/home/rmb-am/Slam_datafiles/frame_labels_validation.txt");



		//See the prediction on a training set image
		for(int le_numero_gagnant = 0;le_numero_gagnant<empty.rgb_img.size();le_numero_gagnant++){
			cout << "numero gagnant is " << le_numero_gagnant << endl;
			cout << empty.pose_vector[le_numero_gagnant].matrix() << endl;
			vector<pair<Vector3d,int> > votes;
			vector<Point> pxSet;
			vector<Vector3d> local_labels;
			Vector3d unknown_position = Vector3d(0,0,0);
			vector<pair<SE3,float> >pose_prior;
			//pose_prior.push_back(make_pair(empty.pose_vector[le_numero_gagnant],1));
			SE3 previous_transform=empty.pose_vector[le_numero_gagnant];//SE3();
			vector<CameraPoseHypothesis> pose_models;


			refine_phase(forest,empty.rgb_img[le_numero_gagnant],empty.d_img[le_numero_gagnant],10,1000,4,pose_models,pose_prior,false,0.7);


			for(int i=0;i<pose_models.size();i++){
				Matrix<double,4,4> diff = pose_models[i].pose.matrix(), GT=previous_transform.matrix();
				ofstream os;
				os.open("/home/rmb-am/Slam_datafiles/reg_tree.txt",ios::app);
				os <<le_numero_gagnant<<" "  << myMatrixNorm(GT,diff) << " " << pose_models[i].error  << " " << se3_metric(previous_transform,pose_models[i].pose,0.7,10)  << " " << se3_metric(previous_transform,pose_models[i].pose,0.6,5)  << " " << se3_metric(previous_transform,pose_models[i].pose,0.5,5) <<endl;
				os.close();
			}
		}


		//Free the memory
		for(int i = 0; i<nb_of_trees;i++)
			destroyTree(forest[i]);

	}

}

