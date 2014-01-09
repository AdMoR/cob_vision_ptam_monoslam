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

//using namespace tr1;
using namespace Eigen;
using namespace std;
using namespace cv;


struct CameraPoseHypothesis{

public:
	SE3 pose;
	double error;
	vector<Vector3d> loc_inliers;
	vector<Vector3d> glo_inliers;

	CameraPoseHypothesis(){
		pose=SE3();
		error=0;
		loc_inliers=vector<Vector3d>();
		glo_inliers=vector<Vector3d>();
	};
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

	bool isLeaf;
	int depth_;

	SplitNodeParam snp;
	Vector3d mode_;
	double quality;
	SplitNode* left_;
	SplitNode* right_;

	//Variable relative to the datas, each pixel has its own rgbd image and 3D label
	vector<Mat> color_img,d_img;
	vector<Point> pixelSet;
	vector<Vector3d> labels;
	Vector3d mean_label;


	SplitNode(vector<Point>& pxSet, vector<Vector3d>& lb,vector<Mat>& rgb, vector<Mat>& d, int depth, int random_param_per_node,int max_depth,bool training):TRAINING_MODE(training),pixelSet(pxSet),labels(lb),color_img(rgb),d_img(d),depth_(depth),MAX_DEPTH(max_depth),quality(0),NB_RAND_PARAM(random_param_per_node),mean_label(Vector3d(0,0,0)){
		isLeaf=depth>=MAX_DEPTH;
		assert((!training) || labels.size()!=0);
		if(training)
			mean_of_vectors(labels,mean_label);
		for(unsigned int i=0;i<color_img.size();i++)
				assert(color_img[i].size()==d_img[i].size());
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



	bool rgbd_feature(Mat & color_img,Mat & depth_img,int delta1,int delta2,int channel1,int channel2,int threshold,int x,int y){
		//cout << color_img.size() << depth_img.size() << (x>0) << (y>0) << (x<color_img.rows) << (y<color_img.cols) << (color_img.channels()==3) << endl;
		if(color_img.size()==depth_img.size() && x>0 && y>0 && x<color_img.rows && y<color_img.cols && color_img.channels()==3){
			int deltafirst,deltasecond;
			assert(depth_img.at<float>(x,y)>=0 && depth_img.at<float>(x,y)<=10000);
			if(depth_img.at<float>(x,y)!=0){
				deltafirst=delta1/depth_img.at<float>(x,y);
				deltasecond=delta2/depth_img.at<float>(x,y);
			}
			else{
				deltafirst=delta1/6000;
				deltasecond=delta2/6000;
		   }
			vector<Mat> channel_v;
			split(color_img,channel_v);
			if(x+deltafirst>0 && y+deltafirst>0 && x+deltafirst<color_img.rows && y+deltafirst<color_img.cols && x+deltasecond>0 && y+deltasecond>0 && x+deltasecond<color_img.rows && y+deltasecond<color_img.cols){
				return (channel_v[channel1].at<unsigned char>(x+deltafirst,y+deltafirst)-channel_v[channel2].at<unsigned char>(x+deltasecond,y+deltasecond))>threshold;
			}
			else{
				int xfirst=x+deltafirst,yfirst=y+deltafirst,xsecond=x+deltasecond,ysecond=y+deltasecond;
				if(x+deltafirst<0)
					xfirst=0;
				if(x+deltafirst>color_img.rows)
					xfirst=color_img.rows-1;
				if(y+deltafirst<0)
					yfirst=0;
				if(y+deltafirst>color_img.cols)
					yfirst=color_img.cols-1;
				if(x+deltasecond<0)
					xsecond=0;
				if(x+deltasecond>color_img.rows)
					xsecond=color_img.rows-1;
				if(y+deltasecond<0)
					ysecond=0;
				if(y+deltasecond>color_img.cols)
					ysecond=color_img.cols-1;
				return (channel_v[channel1].at<unsigned char>(xfirst,yfirst)-channel_v[channel2].at<unsigned char>(xsecond,ysecond))>threshold;
			}
		}
		else{
			cout << "ERROR : problem in size of images !" << x << " " << y << " " << color_img.size() << " " << depth_img.size() << endl;
			return false;}
	}

	double findVariancereduction(vector<Vector3d> labels, vector<bool> left_or_right){

		Vector3d mean=Vector3d(0,0,0),mean_left=Vector3d(0,0,0),mean_right=Vector3d(0,0,0);
		double sum=0,sum_left=0,sum_right=0;
		int size_right=0,size_left=0;

		//Get the mean of each members
		for(unsigned int i=0;i<labels.size();i++){
			//mean+=labels[i];
			if(left_or_right[i]){
				size_right+=1;
				mean_right+=labels[i];
			}
			else{
				size_left+=1;
				mean_left+=labels[i];
			}
		}
		mean=mean_label;
		assert(mean.norm()<100000);
		if(size_left!=0)
			mean_left/=size_left;
		if(size_right!=0)
			mean_right/=size_right;

		//Build the variance reduction
		for(unsigned int i=0;i<labels.size();i++){
			sum+=(labels[i]-mean).squaredNorm();
			if(left_or_right[i]){
				if((labels[i]-mean_right).squaredNorm()<10000)
					sum_right+=(labels[i]-mean_right).squaredNorm();
			}
			else{
				if((labels[i]-mean_left).squaredNorm()<10000)
					sum_left+=(labels[i]-mean_left).squaredNorm();
			}
		}
		//cout  <<mean  << " " << mean_left  << " " << mean_right << " " << size_left << " " <<size_right << endl;
		//Return the variance reduction
		if((1./(float)labels.size())*(sum-sum_left-sum_right)>1000){
			//cout  <<mean  << " " << mean_left  << " " << mean_right << " "<< sum_left<< " " <<sum_right <<" " << size_left << " " <<size_right << endl;
		}
		return (sum-sum_left-sum_right); //WARNING : was before 1/n * ( sum total - sum right - sum left )
	}

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

	void findBestSplit(){
		cout << "start best split decision"<<endl;

		vector<SplitNodeParam> snpv;
		vector<Point> f_left,f_right;
		vector<Vector3d> f_left_label,f_right_label;
		vector<Mat> f_r_rgb_img,f_l_rgb_img,f_r_d_img,f_l_d_img;
		SplitNodeParam bestParam;
		double best_var_red=0;
		getRandomparameter(snpv,NB_RAND_PARAM);

		//Separate the pixels in left and right subsets
		for(auto ptr=snpv.begin();ptr!=snpv.end();ptr++){
			snp=(*ptr);
			cout << "parameter : " << ptr->delta1 << " " << ptr->delta2 << endl;
			vector<bool> left_or_right;
			vector<Point> left,right;
			vector<Vector3d> left_label,right_label;
			vector<Mat> r_rgb,r_d,l_rgb,l_d;
			for(auto i=0;i<pixelSet.size();i++){
				bool direction=getDirection(i,pixelSet[i]);
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

			}
			double varRed=findVariancereduction(labels,left_or_right);

			if(varRed>best_var_red){
				cout << varRed << " current best varRed" << endl;
				cout << left.size() << "  left" << endl;
				cout << right.size() << "  right" << endl;
				bestParam=(*ptr);

				f_left=left;
				f_right=right;

				f_left_label=left_label;
				f_right_label=right_label;

				f_l_d_img=l_d;
				f_l_rgb_img=l_rgb;

				f_r_d_img=r_d;
				f_r_rgb_img=r_rgb;
				best_var_red=varRed;
			}
		}


		cout << "Best var red is : " << best_var_red << endl;
		if(f_left.size()!=0 && f_right.size()!=0){
			snp=bestParam;
			left_ = new SplitNode(f_left,f_left_label,f_l_rgb_img,f_l_d_img,depth_+1,NB_RAND_PARAM,MAX_DEPTH,TRAINING_MODE);
			right_ = new SplitNode(f_right,f_right_label,f_r_rgb_img,f_r_d_img,depth_+1,NB_RAND_PARAM,MAX_DEPTH,TRAINING_MODE);
		}
		else{
			//If the best split doesnt split the parameters, we label it as a leaf
			cout << "Warning : non maximal depth leaf created" << endl;
			isLeaf=true;

		}
		quality=best_var_red;
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


void from2DTo3D(Point& p, double z, Vector3d& label, double f_x=531.15,double f_y=531.15,double image_size_px=240,double image_size_py=320){

	float x=(p.x-image_size_px)/f_x*(z/1000),y=(p.y-image_size_py)/f_y*(z/1000);
	label+=Vector3d(-z/1000,y,-x);

}

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



void getPixelSubset(Mat& img ,vector<Point> & pixelSet,vector<Vector3d>& labels, Vector3d frame_label, int numberOfPixels,int maxDelta=0){

	//CvRTrees r;
	int initial_size=pixelSet.size();

	if(maxDelta<0)
		maxDelta=-maxDelta;

	int randx,randy;
	while(pixelSet.size()!=initial_size+numberOfPixels){
		randx=floor(rand()%img.rows);
		randy=floor(rand()%img.cols);

		if(randx<img.rows-maxDelta && randy<img.cols-maxDelta && randx>maxDelta && randy>maxDelta && img.at<float>(randx,randy)!=0){
			Point p=Point(randx,randy);
			pixelSet.push_back(p);
			Vector3d ptLabel=frame_label;
			//cout << "before : " << ptLabel << " with px : "<< randx << " " << randy <<  endl;
			from2DTo3D(p,img.at<float>(randx,randy),ptLabel);
			//cout << "after : " << ptLabel << endl;
			labels.push_back(ptLabel);
		}
	}

}

void growTree(SplitNode* n){
	//Recursive method to grow the regression random tree
	if(!n->isLeaf){
		assert(n->labels.size()!=0 && n->color_img.size()!=0 && n->d_img.size()!=0 && n->pixelSet.size()!=0 );
		n->findBestSplit();
		cout << n->depth_<< " depth >>> params >>> " << n->snp.delta1 << " " << n->snp.delta2 << endl;
		if(!n->isLeaf){
			growTree((n->left_));
			growTree((n->right_));
		}
		else{
			assert(n->labels.size()!=0);
			cout << "getting non final node mode" << endl;
			get_mean_shift(n);
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
	}
}

void writeNodes(SplitNode* n,ofstream& myfile){

	if(!n->isLeaf){
		myfile << n->depth_ << " " << n->isLeaf << " " << n->snp.delta1 << " " << n->snp.delta2 << " " << n->snp.channel1 << " " << n->snp.channel2 << " "<< n->snp.threshold << " " << n->quality << endl;
		writeNodes((n->left_),myfile);
		writeNodes((n->right_),myfile);
	}
	else{
		myfile << n->depth_ << " " << n->isLeaf << " " << n->mode_(0) << " " << n->mode_(1) << " " << n->mode_(2) << " " << n->labels.size() << endl;
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


void readNode(SplitNode* n,ifstream& myfile,int est_depth){
	string buffer;
	vector <string> fields;


	std::getline(myfile, buffer);
	cout << "line acquired !" << endl;
	boost::algorithm::split_regex( fields, buffer,  boost::regex(" ")  );
	if(fields.size()!=8 && fields.size()!=6)	{
		cout << "ERROR : not the right number of args in the line" << endl;
		return;
	}

	string depth=fields[0],isLeaf=fields[1];
	if(est_depth==atoi(depth.c_str())){
		cout << "depth is ok" << endl;
		n->isLeaf=(bool)atoi(isLeaf.c_str());
		if(!n->isLeaf){
			cout << "we have a normal node" << endl;
			string delta1=fields[2],delta2=fields[3],channel1=fields[4],channel2=fields[5],thresh=fields[6];
			n->snp=SplitNodeParam(atoi(delta1.c_str()),atoi(delta2.c_str()),atoi(channel1.c_str()),atoi(channel2.c_str()),atoi(thresh.c_str()));
			n->quality=atof(fields[7].c_str());
		}
		else{
			cout << "we have a leaf here" << endl;
			n->mode_=Vector3d(atof(fields[2].c_str()),atof(fields[3].c_str()),atof(fields[4].c_str()));
			cout << "the mode is : " << n->mode_(0)<<" " << n->mode_(1) << " "<< n->mode_(2) << endl;
		}
	}
	else{
		cout << "ERROR IN TREE CONSTRUCTION" << endl;
	}


	if(!n->isLeaf){
		n->left_ = new SplitNode(n->pixelSet,n->labels,n->color_img,n->d_img,n->depth_+1,n->NB_RAND_PARAM,n->MAX_DEPTH,n->TRAINING_MODE);
		n->right_ = new SplitNode(n->pixelSet,n->labels,n->color_img,n->d_img,n->depth_+1,n->NB_RAND_PARAM,n->MAX_DEPTH,n->TRAINING_MODE);
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

void predict(SplitNode* n, Point& p,Mat& rgb, Mat& d,Vector3d& result){

	if(!n->isLeaf){
		//cout << n->depth_ << endl;
		if(n->rgbd_feature(rgb,d,n->snp.delta1,n->snp.delta2,n->snp.channel1,n->snp.channel2,n->snp.threshold,p.x,p.y)){
			predict(n->right_,p,rgb,d,result);
		}
		else{
			predict(n->left_,p,rgb,d,result);
		}
	}
	else
		result=n->mode_;

}

void sample_cam_pos(SplitNode* root,vector<CameraPoseHypothesis>& pose_hypothesis,int number_of_samples, Mat rgb_img, Mat d_img){

	vector<Point> pxSet;
	vector<Vector3d> locPxPos,gloPxPos;
	Vector3d zero=Vector3d(0,0,0);

	//Get 3 pixels for each transform
	getPixelSubset(d_img,pxSet,locPxPos,zero,3*number_of_samples,0); //what if 2 points are equal ???

	//Predict the world coord of the pxs
	for(int k=0;k<3*number_of_samples;k++){
		Vector3d res=Vector3d(0,0,0);
		predict(root,pxSet[k],rgb_img,d_img,res);
		gloPxPos.push_back(res);
	}

	//Estimate the transform betyeen local and global
	for(int k=0;k<number_of_samples;k++){
//		MatrixXd H(4,4),Htr(4,4);
//		MatrixXd Rot(3,3);
//		Vector3d tr;
//		MatrixXd M(3,4),X(3,4);
//
//
//		//Build the matrix as the concat of the vecs
//		buildMatrix(M,X,locPxPos,gloPxPos,3*k);
//
//
//		//Least square min
//		Htr=X.jacobiSvd(ComputeThinU | ComputeThinV).solve(M);
//		H=Htr.transpose();
//
//		for(int i=0;i<3;i++){
//			for(int j=0;j<3;j++){
//				Rot(i,j)=H(i,j);
//			}
//			tr(i)=H(i,3);
//		}
//
//
//		MatrixXd U=Rot.jacobiSvd(ComputeThinU | ComputeThinV).matrixU();
//		MatrixXd V=Rot.jacobiSvd(ComputeThinU | ComputeThinV).matrixV();
//		MatrixXd final_R=U*V.transpose();
//
//
//
//		//Add the SE3 to the hypothesis vector
		SE3 se;//=SE3(final_R,tr);
		//WRONG !!!!!!! cout << final_R.UnitX() << " " << final_R.UnitY() << " " << final_R.UnitZ()  << endl;
		//cout << tr(0) << " " << tr(1) << " " << tr(2) << endl;
		solveP3P(se,locPxPos,gloPxPos,3*k);

		CameraPoseHypothesis cph;
		cph.pose=se;

		pose_hypothesis.push_back(cph);
	}

}


void trainRegressionForest(vector<Mat>& rgb_img, vector<Mat>& depth_img, vector<Vector3d>& image_coord, int px_per_frame,int random_parameters_per_node,int nb_tree,int max_depth,int training, string save){

	cout << rgb_img.size() << " " << depth_img.size() << " " << (*(rgb_img.begin())).size() << " " << (*(depth_img.begin())).size() << endl;
	assert(rgb_img.size()==depth_img.size() && (*(rgb_img.begin())).size()==(*(depth_img.begin())).size());



	for(int k=0;k<nb_tree;k++){

		//Build the datas to train the trees
		vector<Point> pixelSet;
		vector<Vector3d> labels;
		vector<Mat> color_img,d_img;

		for(unsigned int i=0;i<rgb_img.size();i++){
			getPixelSubset(depth_img[i],pixelSet,labels,image_coord[i],px_per_frame,0);
			for(int l=0;l<px_per_frame;l++){
				color_img.push_back(rgb_img[i]);
				d_img.push_back(depth_img[i]);
				//labels.push_back(image_coord[i]);
			}
		}

		cout << "labels created " << endl;

		//Create the root of the tree
		SplitNode* root;
		root=new SplitNode(pixelSet,labels,color_img,d_img,0,random_parameters_per_node,max_depth,training);

		cout << "root created " << root->isLeaf << endl;

		//Start the growing
		growTree(root);

		cout << "tree finished" << endl;

		//Save the parameters
		writeTree(root,save);
	}
}

void refine_phase(SplitNode* root, Mat rgb_img, Mat d_img, int log_initial_hypothesis,int number_of_sample_points){

	//Declare main variables
	int K=pow(2,log_initial_hypothesis);
	vector<CameraPoseHypothesis> pose_hypothesis;
	Vector3d zero=Vector3d(0,0,0);

	//Sample hypothesis
	sample_cam_pos(root,pose_hypothesis,K,rgb_img,d_img);

	//Enter the main loop
	for(int i=0;i<log_initial_hypothesis;i++){

		int j=0;

		for(auto ptr=pose_hypothesis.begin();ptr!=pose_hypothesis.end();ptr++){


			j++;

			//Prepare the points
			vector<Point> pxSet;
			vector<Vector3d> gloPxPos,locPxPos;
			getPixelSubset(d_img,pxSet,locPxPos,zero,number_of_sample_points,0);

			for(int k=0;k<number_of_sample_points;k++){
				Vector3d res=Vector3d(0,0,0);
				predict(root,pxSet[k],rgb_img,d_img,res);
				gloPxPos.push_back(res);
			}

			//do X - Hx on every pt

			Matrix<double,4,4> H = (*ptr).pose.matrix();
			for(int k=0;k<number_of_sample_points;k++){
				Vector4d diff=(toHomogeneousCoordinates(gloPxPos[k])-H*toHomogeneousCoordinates(locPxPos[k]));
				ptr->error+=diff.norm();
				if(diff.norm()<=0.3){
					ptr->loc_inliers.push_back(locPxPos[k]);
					ptr->glo_inliers.push_back(gloPxPos[k]);
				}
			}

		}
		//delete the lower half of the map (biggest error)
		cout << "delete half" << endl;
		int init_size=pose_hypothesis.size();
		if(init_size>1){
			for(int i=0;i<init_size/2;i++){
				float biggest_error=0;
				vector<CameraPoseHypothesis>::iterator iter_to_del;
				for(auto it = pose_hypothesis.begin();it!=pose_hypothesis.end();it++){
					if(it->error>biggest_error){
						biggest_error=it->error;
						iter_to_del=it;
					}
				}
				pose_hypothesis.erase(iter_to_del);
			}
		}

		//Refine the pose of the remaining candidates
		cout << "refine " << endl;
		int bingo=1;
		for(auto ptr=pose_hypothesis.begin();ptr!=pose_hypothesis.end();ptr++){
			cout << "numero " << bingo << " has " << ptr->loc_inliers.size() << " inliers " << endl;
			bingo++;
			solveP3P((*ptr).pose,ptr->loc_inliers,ptr->glo_inliers,0,(int)ptr->loc_inliers.size());
		}

	}
	//Show the winner
	cout << "Winner is : " << pose_hypothesis.begin()->pose.translation() << " with error " << pose_hypothesis.begin()->error << endl;


}




int main(int argc, char* argv[]){

	const int nb_frame_to_train=5;
	const int nb_points_per_frame=2000;
	const int nb_random_param_per_node=5000;
	const int nb_of_trees=2;
	const int max_depth=7;
	const int interval_bw_two_frames=100;

	const bool training = false;
	const string tree_saving_location="/home/rmb-am/Slam_datafiles/RGBDRegressionForest7.rf";
	const string img_saving_location="/home/rmb-am/Slam_datafiles/training_img_long/";

	ros::init(argc, argv,"rgbd_random_tree");

	if(training){
		DatasetBuilder rgbd_training_set(interval_bw_two_frames);
		cout << "Init the databuilder" << endl;
		boost::thread image_acquisition(boost::bind(&DatasetBuilder::init,boost::ref(rgbd_training_set),"/odometry_combined","/head_cam3d_frame","/camera/rgb/image_color","/camera/depth_registered/image_raw"));
		//image_acquisition.join();
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
		cout << "vector cout : "<< endl << rgbd_training_set.pose_vector[0] << rgbd_training_set.pose_vector[1] << endl;
		trainRegressionForest(rgbd_training_set.rgb_img,rgbd_training_set.d_img,rgbd_training_set.pose_vector,nb_points_per_frame,nb_random_param_per_node,nb_of_trees,max_depth,training,tree_saving_location);
	}
	else{
		SplitNode* root;
		DatasetBuilder empty(0);
		vector<Point> zero=vector<Point>();
		root=new SplitNode(zero,empty.pose_vector,empty.rgb_img,empty.d_img,0,nb_random_param_per_node,max_depth,training);
		cout << "root created " << root->isLeaf << endl;

		//Read the parameters
		readTree(root,"/home/rmb-am/Slam_datafiles/read_a_forest.rt");

		cout << root->snp.delta1 << " " << root->snp.delta2 << " " << root->snp.channel1 << " " << root->snp.channel2  << " " << root->snp.threshold << endl;

		//Load data
		empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img/","/home/rmb-am/Slam_datafiles/frame_labels.txt");

		//See the prediction on a training set image
		int le_numero_gagnant = 3;
		cout << "numero gagnant is " << le_numero_gagnant << endl;
		vector<pair<Vector3d,int> > votes;
		vector<Point> pxSet;
		vector<Vector3d> local_labels;
		Vector3d unknown_position = Vector3d(0,0,0);
//		getPixelSubset(empty.d_img[le_numero_gagnant],pxSet,local_labels, unknown_position,700,20);
//
//
//		for(auto ptr=pxSet.begin();ptr!=pxSet.end();ptr++){
//			Vector3d result;
//			vector<pair<Vector3d,int> >::iterator current_ptr;
//			bool found = false;
//			predict(root,(*ptr),empty.rgb_img[le_numero_gagnant],empty.d_img[le_numero_gagnant],result);
//			for(auto it=votes.begin();it!=votes.end();it++){
//				if(it->first==result){
//					found = true ;
//					current_ptr = it;
//					break;
//				}
//			}
//			if(!found)
//				votes.push_back(make_pair(result,1));
//			else
//				current_ptr->second+=1;
//		}
//
//		map<int,Vector3d> ordered_by_vote;
//		for(auto ptr=votes.begin();ptr!=votes.end();ptr++){
//			cout << "Vector " << (*ptr).first(0) << " " << (*ptr).first(1) << " " << (*ptr).first(2) << " has " << ptr->second << " votes" << endl;
//			ordered_by_vote.insert(make_pair(ptr->second,ptr->first));
//		}
//
//		cout << "prediction is : " <<  (ordered_by_vote.rbegin()->second)(0) << " " << (ordered_by_vote.rbegin()->second)(1) << " " << (ordered_by_vote.rbegin()->second)(2) << endl;
//		cout << "label was " << empty.pose_vector[le_numero_gagnant] << endl;


		refine_phase(root,empty.rgb_img[le_numero_gagnant],empty.d_img[le_numero_gagnant],7,300);

	}

}

