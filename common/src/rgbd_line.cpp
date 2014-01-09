
#include <opencv2/opencv.hpp>
#include "rgbd_line.h"
#include <iostream>
#include "timer.h"
using namespace std;
using namespace cv;






void build_normal_map(Mat& d_img){

	Mat Nx,Nxs,Nys,Ny,Gx,Gy,Gxx,Gxy,Gyx,Gyy,Gxx_2,Gxy_2,Gyx_2,Gyy_2,Gxshow,Gyshow,show,validity_map,test,oe,gray,dd,NyScale;
	vector<cv::Vec4i> lines;

	test=cv::Mat::zeros(d_img.rows,d_img.cols,CV_8UC1);
	Nys=cv::Mat::zeros(d_img.rows,d_img.cols,CV_8UC1);
	Mat frame=imread("/home/rmb-am/Slam_datafiles/training_img/rgb_4.png");
	cv::cvtColor(frame, gray, CV_BGR2GRAY);


	//Start of new method
	double durationA = cv::getTickCount();
	double durationB= cv::getTickCount();

	validPixelMap(d_img,validity_map);
	std::cout << "duration of validity map is " << (cv::getTickCount()-durationB)/(cv::getTickFrequency()) <<std::endl;
	durationB = cv::getTickCount();
	getOccluding(d_img,validity_map,oe,Nx,Ny,3);
	std::cout << "duration of occluding is " << (cv::getTickCount()-durationB)/(cv::getTickFrequency()) <<std::endl;
	durationB = cv::getTickCount();
	cv::HoughLinesP( oe, lines, 1, CV_PI/180, 45,40,10);
	std::cout << "duration of PHTf is " << (cv::getTickCount()-durationB)/(cv::getTickFrequency()) <<std::endl;
	durationB = cv::getTickCount();

	cout << "size of lines " << lines.size()<< " in " << (cv::getTickCount()-durationA)/(cv::getTickFrequency())<< endl;

	//Start of old method
	durationA = cv::getTickCount();
	cv::Mat dst, blur, src;
	cv::equalizeHist(gray, src );
	cv::GaussianBlur( src, blur, cv::Size(9, 9), 2, 2 );
	cv::Canny(blur, dst, 40, 60);
	int dilation_size=1;
	cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
		                                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		                                       cv::Point( dilation_size, dilation_size ) );
	cv::dilate( dst, dst, element );
	vector<cv::Vec4i> lines2;
	cv::HoughLinesP(dst, lines2, 1, CV_PI/180, 80, 80, 0 );
	cout << "size of lines2 " << lines2.size()<< " in " << (cv::getTickCount()-durationA)/(cv::getTickFrequency())<< endl;


	//Display
	for(auto ptr=lines.begin();ptr!=lines.end();ptr++){
		cv::line(frame, cv::Point((*ptr)[0],(*ptr)[1]), cv::Point((*ptr)[2], (*ptr)[3]), Scalar(255,0,0), 3, 8);
	}
	for(auto ptr=lines2.begin();ptr!=lines2.end();ptr++){
		cv::line(frame, cv::Point((*ptr)[0],(*ptr)[1]), cv::Point((*ptr)[2], (*ptr)[3]), Scalar(0,255,0), 2, 8);
	}

	Sobel(Nx,Gxx,5,1,0);
	Sobel(Nx,Gxy,5,0,1);

	pow(Gxx,2,Gxx_2);
	pow(Gxy,2,Gxy_2);
	pow(Gxx_2 + Gxy_2,0.5,Gx);


	durationA = cv::getTickCount();

	convertScaleAbs(  Ny, NyScale );
	cv::threshold(NyScale,test,100,255,THRESH_TOZERO);
	cv::threshold(NyScale,test,200,255,THRESH_TOZERO_INV);
	Canny(test,Gy,120,230);
	convertScaleAbs(  Gy, Gyshow );
	//cv::threshold(Gyshow,Gyshow,180,255,THRESH_BINARY);
	vector<cv::Vec4i> lines3;
	cv::HoughLinesP(Gyshow, lines3, 5, 3*CV_PI/180, 400, 70, 10 );

	cout << "size of lines3 " << lines3.size()<< " in " << (cv::getTickCount()-durationA)/(cv::getTickFrequency())<< endl;

	convertScaleAbs( d_img, show );
	convertScaleAbs(  Ny, Gxshow );



	for(auto ptr=lines3.begin();ptr!=lines3.end();ptr++){
			cv::line(frame, cv::Point((*ptr)[0],(*ptr)[1]), cv::Point((*ptr)[2], (*ptr)[3]), Scalar(0,0,255), 1, 8);
		}

	imshow("Gx",Gyshow);
	waitKey(0);

	imshow("Gy",oe);
	waitKey(0);

	imshow("depth image",frame);
	waitKey(0);
}


int main(int argc, char* argv[]){

	Mat d_img;
	cout << "ici" << endl;
	getMatrixFromFile(d_img,"/home/rmb-am/Slam_datafiles/training_img/d_4.yml");
	cout << "la" << endl;
	build_normal_map(d_img);


}
