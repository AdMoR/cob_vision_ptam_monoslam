/*
 * utilities.h
 *
 *  Created on: Oct 25, 2013
 *      Author: rmb-am
 */

#pragma once

#include <cv.h>
#include "global.h"
#include <opencv2/opencv.hpp>
#include <list>
#include <fstream>
#include <iostream>
#include <tf/tf.h>
#include <sophus/se3.h>
#include "keyframes.h"
#include <tgmath.h>
#include "transformations.h"

using namespace cv;
using namespace Eigen;
using namespace std;
using namespace ScaViSLAM;

inline  double getEstimforBlur(Mat img,int kernel_size = 3,int scale = 1,int delta = 0,int ddepth = CV_16S){

	Mat out,fout,bw;
	double min,max;
	Point mi,ma;


	if(img.channels()==3)
		cvtColor( img, bw , CV_RGB2GRAY );
	else
		bw=img;

	Laplacian( bw , out , ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
	minMaxLoc(out,&min,&max,&mi,&ma);

	return max;

}

inline void create_lbp_mat(Mat& gray, Mat& out){

	assert(gray.channels()==1);

	out = Mat::zeros(gray.rows,gray.cols,CV_8U);
	int kernelSize=3;

	for(int i=1; i<gray.rows-1;i++){
		for(int j = 1; j<gray.cols-1; j++){

			unsigned char lbp=0;
			int u=0;
			for(int k=-floor(kernelSize/2);k<=floor(kernelSize/2);k++){
				for(int l=-floor(kernelSize/2);l<=floor(kernelSize/2);l++){
					if(!(k==0 && l==0) && gray.at<unsigned char>(i+k,j+l)>gray.at<unsigned char>(i,j)){
						lbp |= 1 << (u);
					}
					u++;
				}
			}

			out.at<unsigned char>(i,j)=lbp;
		}
	}

}


inline  void buildMatrix(MatrixXd& M, MatrixXd& X, vector<Vector3d> locPxPos, vector<Vector3d> gloPxPos,int start=0, int max_i=3){

	for(int i=0;i<max_i;i++){
			for(int j=0;j<3;j++){

				M(i,j)=gloPxPos[start+i](j);
				X(i,j)=locPxPos[start+i](j);
			}
			M(i,3)=1;
			X(i,3)=1;
		}

}

inline  void solveP3P(SE3& seeee,vector<Vector3d> locPxPos, vector<Vector3d> gloPxPos,int start=0,int max_i=3){


	if(max_i==0)
		return;

	MatrixXd H(4,4),Htr(4,4);
	MatrixXd Rot(3,3);
	Vector3d tr;
	MatrixXd M(max_i,4),X(max_i,4);




	//Build the matrix as the concat of the vecs

	buildMatrix(M,X,locPxPos,gloPxPos,start,max_i);


	//Least square min
	Htr=X.jacobiSvd(ComputeThinU | ComputeThinV).solve(M);
	H=Htr.transpose();

	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			Rot(i,j)=H(i,j);
		}
		tr(i)=H(i,3);
	}


	MatrixXd U=Rot.jacobiSvd(ComputeThinU | ComputeThinV).matrixU();
	MatrixXd V=Rot.jacobiSvd(ComputeThinU | ComputeThinV).matrixV();
	MatrixXd final_R=U*V.transpose();



	//Add the SE3 to the hypothesis vector
	seeee.setRotationMatrix(final_R);
	seeee.translation()=tr;
	//WRONG !!!!!!! cout << final_R.UnitX() << " " << final_R.UnitY() << " " << final_R.UnitZ()  << endl;
	//cout << tr(0) << " " << tr(1) << " " << tr(2) << endl;



}

inline  void findLineConflicts(int id_l1, int id_l2, std::multimap<double,std::pair<Line,double>> candidates_l1, std::multimap<double,std::pair<Line,double>> candidates_l2, vector<pair<int,int> > & conflicts){
	if(id_l1!=id_l2){
		for(auto ptr=candidates_l1.begin();ptr!=candidates_l1.end();ptr++){
			cout << "a" << endl;
			for(auto it=candidates_l2.begin();it!=candidates_l2.end();it++){
				if((*ptr).second.first.global_id==(*it).second.first.global_id){
					if(id_l1<id_l2)
						conflicts.push_back(make_pair(id_l1,id_l2));
					else
						conflicts.push_back(make_pair(id_l2,id_l1));
					return;
				}
			}
		}
	}

}

inline  void createPermutedVector(tr1::unordered_map<int,Line>& line_vector,vector<pair<int,Line> > & permuted_line_vector, int permutation_number, vector<pair<int,int> > conflicts){

	//permuted_line_vector=line_vector;
	for(auto ptr=line_vector.begin();ptr!=line_vector.end();ptr++)
		permuted_line_vector.push_back(make_pair(ptr->first,ptr->second));

	//Retrieve permutation activation from bits of the permutation number
	for(unsigned int l=0;l<conflicts.size();l++){// n conflicts ==> 2^n possible permutations ==> n bits on the number
		//cout << "bit decision on conflict " << conflicts[l].first << " " << conflicts[l].second << " : " << ((permutation_number & ( 1 << l )) >> l) << endl;
		if(((permutation_number & ( 1 << l )) >> l)){ //If 1 then we permute i,j from conflict(l)
			for(auto ptr_l1=permuted_line_vector.begin();ptr_l1!=permuted_line_vector.end();ptr_l1++){
				for(auto ptr_l2=ptr_l1;ptr_l2!=permuted_line_vector.end();ptr_l2++){
					if(ptr_l1->first==conflicts[l].first && ptr_l2->first==conflicts[l].second){
						int save_id=ptr_l1->first;
						Line save_l=ptr_l1->second;
						ptr_l1->first=ptr_l2->first;
						ptr_l1->second=ptr_l2->second;
						ptr_l2->first=save_id;
						ptr_l2->second=save_l;
					}
				}
			}
		}
	}

}



inline  void dumpToFile(std::string frame_nb, double x, double y, double z, double rx=100000000000, double ry=100000000000, double rz=100000000000, double rw=100000000000, string filePath="/home/rmb-am/Slam_datafiles/measurements.txt"){

	ofstream myfile;
	myfile.open (filePath, ios::app);
	if(x==0 && y==0 && z==0 && rx==0 && ry==0 && rz==0 && rw==0){
		if(myfile.is_open())
			myfile << endl;
	}


	if(myfile.is_open()){
		if(rw<pow(10,6))
			myfile << frame_nb << " " << x << " " << y << " " << z << " " << rx << " " << ry << " " << rz << " " << rw <<  endl;
		else{
			if(rz<pow(10,6)){
				myfile << frame_nb << " " << x << " " << y << " " << z << " " << rx << " " << ry << " " << rz <<  endl;
			}
			else{
				if(ry<pow(10,6)){
					myfile << frame_nb << " " << x << " " << y << " " << z << " " << rx << " " << ry <<  endl;
				}
				else{
					if(rx<pow(10,6)){
						myfile << frame_nb << " " << x << " " << y << " " << z << " " << rx <<  endl;
					}
					else
						myfile << frame_nb << " " << x << " " << y << " " << z <<  endl;
				}
			}
		}
	}
	myfile.close();

}

inline  void getTrackedLinesToFile(tr1::unordered_map<int,Line> tracked_lines){
	int i=0;
	for(auto ptr=tracked_lines.begin();ptr!=tracked_lines.end();ptr++){
		if((*ptr).second.global_id==21)
			{i++;
			Vector6d v = (*ptr).second.optimizedPluckerLines;
			dumpToFile( std::to_string((*ptr).second.global_id),v(0,0), v(1,0), v(2,0),v(3,0), v(4,0),v(5,0),0, "/home/rmb-am/Slam_datafiles/map/lines.txt");
		}
	}
	//dumpToFile( "<<<<>>>>>",i,0,0,0,0,0,0, "/home/rmb-am/Slam_datafiles/map/lines.txt");

}



inline  void cross(Mat& img, Point p, Scalar color=Scalar(255,0,0)){

	line(img,Point(p.x-5,p.y-5),Point(p.x+5,p.y+5),color,3,0,0);
	line(img,Point(p.x-5,p.y+5),Point(p.x+5,p.y-5),color,3,0,0);
}

inline  void display(Mat img,int scalex,int scaley,string title="new_img"){

	Mat out;
	resize(img,out,Size(),scalex,scaley,INTER_LINEAR);
	imshow(title,out);
	waitKey(0);
}

inline  void getMatrixFromFile(Mat& d, string d_name){

	FileStorage fs(d_name.c_str(), FileStorage::READ);
	fs["d"] >> d;

}

inline void mean_of_vectors(vector<Vector3d>& vv,Vector3d& mean){

	if(vv.size()!=0){
		for(unsigned int i=0;i<vv.size();i++){
			mean+=vv[i];
		}
		mean/=(double)vv.size();

		if(mean.norm()>10000 ){
			cout << mean << " "<< vv.size() <<endl;
			for(int i=0;i<vv.size();i++)
				cout << vv[i](0)<< " " << vv[i](1) <<" " << vv[i](2) << endl;
		}
		assert(mean.norm()<10000);
	}

}

inline void showPoints(Mat img,vector<Point> vPoints, string title="show points", Scalar color=Scalar(255,0,0)){
	 Mat copy=img.clone();

	 for(int i=0; i<vPoints.size();i++){
		 cross(copy,vPoints[i],color);
	 }

	 imshow(title,copy);
	 waitKey(0);
}

inline void showPoints(Mat img,vector<Eigen::Vector2d> vPoints, string title="show points", Scalar color=Scalar(255,0,0)){
	 Mat copy=img.clone();

	 for(int i=0; i<vPoints.size();i++){
		 Point p=Point(vPoints[i][0],vPoints[i][1]);
		 cross(copy,p,color);
	 }

	 imshow(title,copy);
	 waitKey(0);
}

inline  void showPoints(Mat img, Point pointP, string title="show points", Scalar color=Scalar(255,0,0)){
	 Mat copy=img.clone();

	 cross(copy,pointP,color);


	 imshow(title,copy);
	 waitKey(0);
}

inline double pseudoVecProd(Vector2d v1, Vector2d v2){
	return v1(0)*v2(1)-v1(1)*v2(0);
}

inline  double scalar_prod(Vector2d v1,Vector2d v2){
	return v1(0)*v2(0)+v1(1)*v2(1);
}


inline  void projectReferencePoint(Vector3d line_equation,double rtheta, Vector2d& reference_point){
	//OP= r er + rt etheta -->polar coord
	//er = (a,b) if c<0 =-(a,b) else

	double mag,angle;
	cartesianToPolar(line_equation,mag,angle);
	Vector2d n=Vector2d(line_equation(0),line_equation(1)),l=Vector2d(-line_equation(1),line_equation(0));
	n.normalize();
	l.normalize();
	if(line_equation(2)>0){
		n=-n;
		l=-l;
	}

	//cout << mag << " " << rtheta << endl;
	reference_point= mag*n+rtheta*l;

}

inline  double segmentIntersect(Vector2d p1, Vector2d p2, Vector2d p3, Vector2d v2){
	//Some comments...
	//v2 should be the line equation and be normalized but not the other vectors
	Vector2d v1=(p2-p1),v3=p3-p1;

    if(pseudoVecProd(v1,v2)>0.0000001)
        return (pseudoVecProd(v3,v2)/pseudoVecProd(v1,v2));
    else
    {//cout << "probably an error colinear vectors in segment intersect" << endl;
        return -100;}

}


inline  double getLineDistance(Vector2d candidate_line_projection_point,Vector2d candidate_line_projection_point2, Vector3d tracked_line_equation,Vector2d tr_l_pt, Vector2d& found_point){
	//We find the distance between a segment and a line with the use of points
	//Either the starting point or ending point of the segment has the smallest distance with the line
	//We estimate the error with the distance between the segment point and its orthogonal projection on the tracked line
	Vector2d candidate_line_equation=candidate_line_projection_point2-candidate_line_projection_point;
	Vector2d line_equation_normal=Vector2d(tracked_line_equation(0),tracked_line_equation(1));
	line_equation_normal.normalize();
	Vector2d line_equation_lin=Vector2d(tracked_line_equation(1),-tracked_line_equation(0));
	line_equation_lin.normalize();

	double t=abs(segmentIntersect(candidate_line_projection_point,candidate_line_projection_point2,tr_l_pt,line_equation_normal));
	if(t>=1){
		return min(abs(scalar_prod(candidate_line_projection_point-tr_l_pt,line_equation_normal)),abs(scalar_prod(candidate_line_projection_point2-tr_l_pt,line_equation_normal)));
		}
	else{

		Vector2d candidate_equ=Vector2d(candidate_line_equation(0),candidate_line_equation(1));
		Vector2d intersectPoint=candidate_line_projection_point+t*candidate_equ;
		found_point=intersectPoint;
		return 0;
		}
}



inline void findIntersectionPoint(Vector4d plan1, Vector4d plan2, double z,Vector3d& ptOut){

	//2 planes p1 p2 intersect in a line l, we find a point on this line with an intersecting plane z=zo
	//x= - (   (d'+c'*z) - b'/b * (d+c*z)  )   /    ( a' - b'/b * a )
	//y= -(d+c*z)/b    -  a/b*x
	//z=z

	ptOut(0)=-((plan2(3)+plan2(2)*z)-(plan2(1)/plan1(1))*(plan1(3)+plan1(2)*z))/(plan2(0)-plan2(1)/plan1(1)*plan1(0));
	ptOut(1)=-(plan1(3)+plan1(2)*z)/plan1(1) - plan1(0)/plan1(1)*ptOut(0);
	ptOut(2)=z;


}



inline  void pluckerToFile(Vector6d pluckerLine,int line_id,String filePath="/home/rmb-am/Slam_datafiles/lineCoordEvol.txt"){

	dumpToFile(std::to_string(line_id),pluckerLine(0),pluckerLine(1),pluckerLine(2),pluckerLine(3),pluckerLine(4),pluckerLine(5),1000000000000,filePath);
}

inline  void pluckerToFile(Vector3d pluckerLine,int line_id,String filePath="/home/rmb-am/Slam_datafiles/lineCoordEvol.txt"){

	dumpToFile(std::to_string(line_id),pluckerLine(0),pluckerLine(1),pluckerLine(2),1000000000000,1000000000000,1000000000000,1000000000000,filePath);
}


inline  void matrixFromTwoVec(Vector4d& v1, Vector4d& v2, Matrix4d& out){
	int n=v1.size();

	if(v2.size()!=n){
		cout << "ERROR IN VECTOR SIZE!" <<endl;
		return;}

	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			out(i,j)=v1(i)*v2(j);
		}
	}
}

inline  void getProjectionVector(Vector2d pixelPoint,double focal_length, Vector3d& projectionDirection,bool verbose=false){

//	Vector2d cameraCenter=Vector2d(640,-480);
//
//	Vector2d preprojectionDirection=(0.5*cameraCenter-(cameraCenter-Vector2d(pixelPoint(1),-pixelPoint(0))));

	if(verbose){
		cout << "px : " << pixelPoint(0) << " " << pixelPoint(1) << endl;
		cout << "new px : " << 640-pixelPoint(0) << " " << 480-pixelPoint(1) << endl;

	}

	Vector3d cm=Vector3d(640-pixelPoint(0),480-pixelPoint(1),focal_length);




	projectionDirection=Vector3d(320-cm(0),240-cm(1),cm(2));

	if(verbose){
		cout << "cm : " << cm(0) << " " << cm(1) << endl;
		cout << "proj : " << projectionDirection(0) << " " << projectionDirection(1) << (projectionDirection(2)) << endl;
	}


	projectionDirection.normalize();

}

inline  void switchOne(Vector3d& Vec,int num){
	if(num<3)
		Vec(num)=-Vec(num);
}

inline void switchOne(Vector4d& Vec,int num){
	if(num<3)
		Vec(num)=-Vec(num);
}

inline  void switchAll(Vector4d& Vec){
	Vec=-Vec;
}



inline  double scalar_prod(Vector3d v1,Vector3d v2){
	return v1(0)*v2(0)+v1(1)*v2(1)+v1(2)*v2(2);
}

inline  void vect_product(Vector3d v1, Vector3d v2, Vector3d& out){
	out(2)=v1(0)*v2(1)-v1(1)*v2(0);
	out(0)=v1(1)*v2(2)-v1(2)*v2(1);
	out(1)=v1(2)*v2(0)-v1(0)*v2(2);
}

inline  void vect_product(Vector4d v1, Vector4d v2, Vector4d& out){
	out(2)=v1(0)*v2(1)-v1(1)*v2(0);
	out(0)=v1(1)*v2(2)-v1(2)*v2(1);
	out(1)=v1(2)*v2(0)-v1(0)*v2(2);
	out(3)=0;
}




inline  void get_plan_equ(Vector3d line_in_cam_frame, Vector4d& out, Vector3d camera_direction,bool verbose=false){

	Vector3d lineCoord3D=line_in_cam_frame,normal;
	lineCoord3D(2)=0;
	lineCoord3D(1)=-line_in_cam_frame(0);
	lineCoord3D(0)=line_in_cam_frame(1); //V=(1,-a/b)

	if(verbose)
		cout << "lineCOOrd3D" << lineCoord3D(0) << " " << lineCoord3D(1) << " " << line_in_cam_frame(2) << endl;

	vect_product(lineCoord3D,camera_direction,normal);

	normal.normalize();


	if(verbose)
				cout << "cam dir" << camera_direction(0) << " " << camera_direction(1)<< " " << camera_direction(2) << endl;
	if(verbose)
			cout << "normal" << normal(0) << " " << normal(1) << " " << normal(2) << endl;

	Vector3d ptC=Vector3d(320,240,531.15); //The point used is the lens center

	double d=-scalar_prod(normal,ptC);

	out(0)=normal(0);
	out(1)=normal(1);
	out(2)=normal(2);
	out(3)=d;

	if(verbose)
		cout << d << endl;


}

inline  void drawMyLine(Vector3d projectedHomogeneousLine, cv::Mat curFrameRGB, const string &WindowName, const cv::Scalar & color, bool verbose)
{
	if (projectedHomogeneousLine[1] != (double)0.0 && projectedHomogeneousLine[0] != (double)0.0)
	{
		int y = (-1) * (projectedHomogeneousLine[2] / projectedHomogeneousLine[1]);
		int x = (-1) * (projectedHomogeneousLine[2] / projectedHomogeneousLine[0]);

		if (y > 0 & x > 0)
		{
			if(verbose)	 cout<<"P: (0,"<<y<<") Q: ("<<x<<",0)"<<endl;
			cv::line(curFrameRGB, cv::Point(0, y), cv::Point(x, 0), color, 2, 8);
		}
		else if (y < 0 & x > 0)
		{
			int newx = (-projectedHomogeneousLine[2] - 480 * projectedHomogeneousLine[1]) / projectedHomogeneousLine[0];
			if(verbose)	 cout<<"P: ("<<x<<",0) Q: ("<<newx<<",480)"<<endl;
			cv::line(curFrameRGB, cv::Point(x, 0), cv::Point(newx, 480), color, 2, 8);
		}
		else if (y > 0 & x < 0)
		{
			int newy = (-projectedHomogeneousLine[2] - 640 * projectedHomogeneousLine[0]) / projectedHomogeneousLine[1];
			if(verbose)	 cout<<"P: (0,"<<y<<") Q: (640"<<newy<<")"<<endl;
			cv::line(curFrameRGB, cv::Point(0, y), cv::Point(640, newy), color, 2, 8);
		}else
		{
			cout<<"biak negatibo, x: "<<x<<" y: "<<y<<endl;
			cout<<"projectedHomogeneousLine[0]: "<<projectedHomogeneousLine[0]<<" projectedHomogeneousLine[1]: "<<projectedHomogeneousLine[1]<<" projectedHomogeneousLine[2]:"<<projectedHomogeneousLine[2]<<endl;
		}
		cv::imshow(WindowName, curFrameRGB);

	}
}

inline  void getTransformedNormalVect(Vector4d n,Matrix<double,4,4> transform,Vector4d& out,bool verbose=false){
	Matrix<double,3,3>rot;
	Vector3d tr;
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			rot(i,j)=transform(i,j);
		}
		tr(i)=transform(i,3);
	}
	//switchXYZ(n);

	Vector3d normal=Vector3d(n(0),n(1),n(2)),preout;
	preout=rot*normal;

	if(verbose)
		cout << "tr(2) : " << tr(2) << ",n(3) : " << n(3) <<" "<< n(2)<<" " << n(1) <<" "<< n(0) << endl;
	//The coordinates are different xyz --> -x(-y)z
	//w_tr=Vector3d(-tr(0),-tr(1),tr(2));

	double d=-scalar_prod(tr,normal)+0.0000028*n(3);

	if(verbose)
		cout <<"d = " << d << " = " << -scalar_prod(-tr,normal)<< " + " << 0.0000028*n(3) << endl;


	out(0)=preout(0);
	out(1)=preout(1);
	out(2)=preout(2);
	out(3)=d;


}

inline void computePluckerFromPlanes(Vector4d plane1, Vector4d plane2,Matrix<double,4,4> T_1_w,Matrix<double,4,4> T_2_w, Vector6d& out,int line_id=-1){
	Matrix4d member1,member2;
	Vector4d plane1w,plane2w;
	Matrix<double,4,4> T_w_1=T_1_w.inverse(),T_w_2=T_2_w.inverse(); //Express the planes in world coordinates

	getTransformedNormalVect(plane1,T_1_w,plane1w,true);

	getTransformedNormalVect(plane2,T_2_w,plane2w,true);
	Vector4d line3DEqu;

	vect_product(plane1w,plane2w,line3DEqu);

	matrixFromTwoVec(plane1w,plane2w,member1); //Compute PQt
	matrixFromTwoVec(plane2w,plane1w,member2); // and QPt
	Matrix4d preout=member1-member2;

	out=toPlueckerVec(preout).reverse();
}

inline  void convertEigenToCv(MatrixXd& eigenMat,int nbrow, int nbcol, Mat& cvMat){

	for(int i =0;i<nbrow;i++){
		for(int j=0;j<nbcol;j++){
			cvMat.at<double>(i,j)=eigenMat(i,j);
		}
	}

}

inline  void convertEigenToCv(Vector3d& eigenMat,int nbrow, Mat& cvMat){

	for(int i =0;i<nbrow;i++){

			cvMat.at<double>(i,0)=eigenMat(i);

	}

}

inline  void getMatrixPower(Vector3d & mat, Matrix3d & out, float p){
	//2 possibilities 0.5 and 2
	for(int i=0;i<3;i++){
		//for(int j=0;j<ncol;j++){
			if(p==0.5)
				out(i,i)=sqrt(mat(i));
			if(p==-0.5 && mat(i)!=0)
					out(i,i)=1/sqrt(mat(i));
			if(p==2)
				out(i,i)=(mat(i))*mat(i);
		}
	//}

}


  inline vector<pair<int,int>>  lineBresenham(int p1x, int p1y, int p2x, int p2y,int k=0)
{
    int F, x, y;
    vector<pair<int,int>> pixelsOnLine;

    if (p1x > p2x)  // Swap points if p1 is on the right of p2
    {
        swap(p1x, p2x);
        swap(p1y, p2y);
    }

    // Handle trivial cases separately for algorithm speed up.
    // Trivial case 1: m = +/-INF (Vertical line)
    if (p1x == p2x)
    {
        if (p1y > p2y)  // Swap y-coordinates if p1 is above p2
        {
            swap(p1y, p2y);
        }

        x = p1x;
        y = p1y;
        while (y <= p2y)
        {
            //cout<<"x: "<<x<<"y: "<<y<<endl;
            pixelsOnLine.push_back(make_pair(x,y));
            y++;
        }
        //return;
    }
    // Trivial case 2: m = 0 (Horizontal line)
    else if (p1y == p2y)
    {
        x = p1x;
        y = p1y;

        while (x <= p2x)
        {
        	//cout<<"x: "<<x<<"y: "<<y<<endl;
        	pixelsOnLine.push_back(make_pair(x,y));
            x++;
        }
        //return;
    }
    else
    {


    int dy            = p2y - p1y;  // y-increment from p1 to p2
    int dx            = p2x - p1x;  // x-increment from p1 to p2
    int dy2           = (dy << 1);  // dy << 1 == 2*dy
    int dx2           = (dx << 1);
    int dy2_minus_dx2 = dy2 - dx2;  // precompute constant for speed up
    int dy2_plus_dx2  = dy2 + dx2;
    pixelsOnLine.reserve(max(dy,dx));

    if (dy >= 0)    // m >= 0
    {
        // Case 1: 0 <= m <= 1 (Original case)
        if (dy <= dx)
        {
            F = dy2 - dx;    // initial F

            x = p1x;
            y = p1y;
            while (x <= p2x)
            {
            	//cout<<"x: "<<x<<"y: "<<y<<endl;
            	pixelsOnLine.push_back(make_pair(x,y));
                if (F <= 0)
                {
                    F += dy2;
                }
                else
                {
                    y++;
                    F += dy2_minus_dx2;
                }
                x++;
            }
        }
        // Case 2: 1 < m < INF (Mirror about y=x line
        // replace all dy by dx and dx by dy)
        else
        {
            F = dx2 - dy;    // initial F

            y = p1y;
            x = p1x;
            while (y <= p2y)
            {
            	//cout<<"x: "<<x<<"y: "<<y<<endl;
            	pixelsOnLine.push_back(make_pair(x,y));
                if (F <= 0)
                {
                    F += dx2;
                }
                else
                {
                    x++;
                    F -= dy2_minus_dx2;
                }
                y++;
            }
        }
    }
    else    // m < 0
    {
        // Case 3: -1 <= m < 0 (Mirror about x-axis, replace all dy by -dy)
        if (dx >= -dy)
        {
            F = -dy2 - dx;    // initial F

            x = p1x;
            y = p1y;
            while (x <= p2x)
            {
            	//cout<<"x: "<<x<<"y: "<<y<<endl;
            	pixelsOnLine.push_back(make_pair(x,y));
                if (F <= 0)
                {
                    F -= dy2;
                }
                else
                {
                    y--;
                    F -= dy2_plus_dx2;
                }
                x++;
            }
        }
        // Case 4: -INF < m < -1 (Mirror about x-axis and mirror
        // about y=x line, replace all dx by -dy and dy by dx)
        else
        {
            F = dx2 + dy;    // initial F

            y = p1y;
            x = p1x;
            while (y >= p2y)
            {
            	//cout<<"x: "<<x<<"y: "<<y<<endl;
            	pixelsOnLine.push_back(make_pair(x,y));
                if (F <= 0)
                {
                    F += dx2;
                }
                else
                {
                    x++;
                    F += dy2_plus_dx2;
                }
                y--;
            }
        }
    }
    }
    return pixelsOnLine;
}

  inline  void showLines(Mat img, vector<cv::Vec4i> lv, string title="show points", Scalar color=Scalar(255,0,0)){
	 Mat pcopy=img.clone(),copy;
	 cvtColor( pcopy,copy, CV_GRAY2RGB );

	 for(int i=0;i<lv.size();i++){
		 line(copy,Point(lv[i][0],lv[i][1]),Point(lv[i][2],lv[i][3]),color);
	 }


	 imshow(title,copy);
	 waitKey(0);
}







