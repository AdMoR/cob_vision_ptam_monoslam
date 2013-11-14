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


using namespace cv;
using namespace std;
using namespace ScaViSLAM;

double getEstimforBlur(Mat img,int kernel_size = 3,int scale = 1,int delta = 0,int ddepth = CV_16S){

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



void dumpToFile(std::string frame_nb, float x, float y, float z, float rx, float ry, float rz, float rw, string filePath="/home/rmb-am/Slam_datafiles/measurements.txt"){

	ofstream myfile;
	myfile.open (filePath, ios::app);
	if(myfile.is_open()){
		myfile << frame_nb << " " << x << " " << y << " " << z << " " << rx << " " << ry << " " << rz << " " << rw <<  endl;
	}
	myfile.close();
}

void getTrackedLinesToFile(tr1::unordered_map<int,Line> tracked_lines){
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


vector<pair<int,int>>  lineBresenham(int p1x, int p1y, int p2x, int p2y,int k=0)
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

void cross(Mat img, Point p, Scalar color=Scalar(255,0,0)){

	line(img,Point(p.x-5,p.y-5),Point(p.x+5,p.y+5),color,1,0,0);
	line(img,Point(p.x-5,p.y+5),Point(p.x+5,p.y-5),color,1,0,0);
}

void display(Mat img,int scalex,int scaley,string title="new_img"){

	Mat out;
	resize(img,out,Size(),scalex,scaley,INTER_LINEAR);
	imshow(title,out);
	waitKey(0);
}


void showPoints(Mat img,vector<Point> vPoints, string title="show points", Scalar color=Scalar(255,0,0)){
	 Mat copy=img.clone();

	 for(int i=0; i<vPoints.size();i++){
		 cross(copy,vPoints[i],color);
	 }

	 imshow(title,copy);
	 waitKey(0);
}

void showPoints(Mat img,vector<Eigen::Vector2d> vPoints, string title="show points", Scalar color=Scalar(255,0,0)){
	 Mat copy=img.clone();

	 for(int i=0; i<vPoints.size();i++){
		 Point p=Point(vPoints[i][0],vPoints[i][1]);
		 cross(copy,p,color);
	 }

	 imshow(title,copy);
	 waitKey(0);
}

void showPoints(Mat img, Point pointP, string title="show points", Scalar color=Scalar(255,0,0)){
	 Mat copy=img.clone();

	 cross(copy,pointP,color);


	 imshow(title,copy);
	 waitKey(0);
}

void showLines(Mat img, vector<cv::Vec4i> lv, string title="show points", Scalar color=Scalar(255,0,0)){
	 Mat pcopy=img.clone(),copy;
	 cvtColor( pcopy,copy, CV_GRAY2RGB );

	 for(int i=0;i<lv.size();i++){
		 line(copy,Point(lv[i][0],lv[i][1]),Point(lv[i][2],lv[i][3]),color);
	 }


	 imshow(title,copy);
	 waitKey(0);
}

bool gradTest(Mat img, Point p, Point directionVector, float refA,float thres=0.00,float thresA=0.1 ){

	Mat ROI;
	//cout << img.size()<< endl;
	if((p.x-4)>0 && (p.x+4)<img.cols && (p.y-4)>0 && (p.y+4)<img.rows)
		ROI=(img.colRange(p.x-4,p.x+4).rowRange(p.y-4,p.y+4)).clone();
	else
		return false;
	Mat dx,dy;

	//cout << "soebl"<< endl;
	Sobel( ROI, dx, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT );
	Sobel( ROI, dy, CV_32F, 0,1, 3, 1, 0, BORDER_DEFAULT );
	//cout << "endl sobebleble" << endl;

	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;

//	minMaxLoc( ROI, &minVal, &maxVal, &minLoc, &maxLoc );
//	cout << maxVal << "max ROI" << endl;
//	minMaxLoc( dx, &minVal, &maxVal, &minLoc, &maxLoc );
//	cout << maxVal << "max dx" << endl;
//	minMaxLoc( ROI, &minVal, &maxVal, &minLoc, &maxLoc );
//	cout << maxVal << "max ROI" << endl;
//	minMaxLoc( dy, &minVal, &maxVal, &minLoc, &maxLoc );
//	cout << maxVal << "max dy" << endl;

//	display(ROI,10,10,"ROI");
//	display(dx,10,10,"dx");
//	display(dy,10,10,"dy");
	cout << ROI.size() << endl;

	float intensity=abs(dx.at<float>(4,4))+abs(dy.at<float>(4,4));
	float angle=atan(dy.at<float>(4,4)/dx.at<float>(4,4));
	cout << intensity << " intensity<<>>angle " << angle<< endl;

	//float intensity=mean(abs(dx)+abs(dy));

	//cout << "end" << endl;
	return ((intensity>thres)&&(cos(angle-refA)<thresA));
}



void findLines(vector<Point> fastDetection, Mat img,vector<Point> & trueLines){

	for(unsigned int i=0;i<fastDetection.size();i++){
		for(unsigned int j=i+1; j<fastDetection.size();j++){


			//Test that the line is long enough
			Point diff=fastDetection[i]-fastDetection[j];
			if((abs(diff.x)+abs(diff.y))<30)
				continue;


			float diffAngle;
			if(diff.x!=0)
				diffAngle = atan(diff.y/diff.x);
			else{
				if(diff.y>0)
					diffAngle=M_PI/2;
				else
					diffAngle=-M_PI/2;
			}

			//Test the grad val of the mid point
			Point mid=(fastDetection[i]+fastDetection[j])*0.5;
			if(!gradTest(img,mid,diff,diffAngle))
				continue;

			//Test the grad val of the first quarter of the line
			Point fquarter=(3*fastDetection[i]+fastDetection[j])*0.25;
			if(!gradTest(img,fquarter,diff,diffAngle))
				continue;


			//Test the grad val of the last quarter of the line
			Point tquarter=(fastDetection[i]+3*fastDetection[j])*0.25;
			if(!gradTest(img,tquarter,diff,diffAngle)){
				continue;}


			//Test all the pixels
			vector<pair<int,int> > pixelOnLine= lineBresenham(fastDetection[i].x,fastDetection[i].y,fastDetection[j].x,fastDetection[j].y);
			float good=0,bad=0;
			for(auto ptr=pixelOnLine.begin(); ptr!=pixelOnLine.end();ptr++ ){
				if(gradTest(img,Point((*ptr).first,(*ptr).second),diff,diffAngle))
					good+=1;
			}


			//A majority should be edges
			if((good/((float)pixelOnLine.size()))>=0.5)
				trueLines.push_back(Point(i,j));
		}
	}

}

//void showLines(Mat img,vector<Line> vLines, string title="show lines", Scalar color=Scalar(255,0,0)){
//	 Mat copy=img.clone();
//
//	 for(int i=0; i<vLines.size();i++){
//		 line(copy,vLines[i].startingPoint2d,vLines[i].endingPoint2d,color,1,0);
//	 }
//
//	 imshow(title,copy);
//	 waitKey(0);
//}

