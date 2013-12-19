/*
 * test.cpp
 *
 *  Created on: Oct 25, 2013
 *      Author: rmb-am
 */

//#include "utilities.h"

using namespace std;
using namespace cv;

//int main(){
//
//	string pathToFile="/home/rmb-am/git/cob_object_perception_intern/cob_vision_ptam_monoslam/wall.jpg";
//	int thres=30;
//	Mat img,copy;
//
//
//
//	//Method with fast
//	cvtColor( imread(pathToFile),img, CV_BGR2GRAY );
//	vector<KeyPoint> kpv;
//	vector<cv::Point> pVec;
//
//	 FAST(img, kpv, thres ,true );
//
//	for(auto iter=kpv.begin(); iter!=kpv.end();++iter){
//		//cv::Point p = cv::Point((*iter).pos[0],(*iter).pos[1]);
//		pVec.push_back((*iter).pt);
//	}
//	showPoints(img,pVec,"no blur");
//	cout << "Found " << pVec.size() << " pts." << endl;
//
//	blur(img,copy,Size(3,3));
//	kpv.clear();
//	pVec.clear();
//
//	FAST(img, kpv, thres+10 ,true );
//	cv::KeyPointsFilter::retainBest(kpv, 50);
//
//	for(auto iter=kpv.begin(); iter!=kpv.end();++iter){
//		//cv::Point p = cv::Point((*iter).pos[0],(*iter).pos[1]);
//		pVec.push_back((*iter).pt);
//	}
//	showPoints(copy,pVec,"blur+3");
//	cout << "Found " << pVec.size() << " pts." << endl;
//
//
//	//Method with Canny+Hough
//	 clock_t startTime = clock();
//	Mat src,dst,blur;
//	cv::equalizeHist( img, src );
//
//	cv::GaussianBlur( src, blur, cv::Size(9, 9), 2, 2 );
//
//	cv::Canny(blur, dst, 40, 60);
//	int dilation_size=1;
//	cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ), cv::Point( dilation_size, dilation_size ) );
//	cv::dilate( dst, dst, element );
//
//	vector<cv::Vec4i> lines;
//	cv::HoughLinesP(dst, lines, 1, CV_PI/180, 80, 80, 0 );
//    std::vector<cv::Rect> array;
//    cout<<"hough detected lines size: "<<lines.size()<<endl;
//    cout << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << endl;
//    showLines(img,lines,"line show");
//
//
//    vector<Point> gt;
//    for(auto ptr=lines.begin();ptr!=lines.end();ptr++){
//    	gt.push_back(Point((*ptr)[0],(*ptr)[1]));
//    	gt.push_back(Point((*ptr)[2],(*ptr)[3]));
//    }
//
//
//    //Method with fast
//    startTime = clock();
//    vector<Point> trueLines;
//    vector<Vec4i> lines2;
//    cout << "Entering find lines" << endl;
//    findLines(gt, img,trueLines);
//    cout << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << endl;
//
//    for(auto ptr=trueLines.begin();ptr!=trueLines.end();ptr++){
//    	lines2.push_back(Vec4i((gt[(*ptr).x]).x,(gt[(*ptr).x]).y,(gt[(*ptr).y]).x,(gt[(*ptr).y]).y));
//    }
//
//    showLines(img,lines2,"lines with fast");
//}



int main(){

//	Mat img_no_blur, img_blur,img_test;
//
//	double m;
//
//
////	cvtColor( imread("/home/rmb-am/git/cob_object_perception_intern/cob_vision_ptam_monoslam/1.pnm"), img_no_blur, CV_RGB2GRAY );
////	cvtColor( imread("/home/rmb-am/git/cob_object_perception_intern/cob_vision_ptam_monoslam/2.pnm"), img_blur, CV_RGB2GRAY );
////	cvtColor( imread("/home/rmb-am/git/cob_object_perception_intern/cob_vision_ptam_monoslam/3.pnm"), img_test, CV_RGB2GRAY );
//	img_no_blur=imread("/home/rmb-am/git/cob_object_perception_intern/cob_vision_ptam_monoslam/1.pnm");
//	img_blur=imread("/home/rmb-am/git/cob_object_perception_intern/cob_vision_ptam_monoslam/2.pnm");
//	img_test=imread("/home/rmb-am/git/cob_object_perception_intern/cob_vision_ptam_monoslam/3.pnm");
//
//	Mat blurred_no_blur;
//	GaussianBlur(img_no_blur,blurred_no_blur,Size(5,5),1,1,BORDER_DEFAULT);
//
//
//	cout << img_no_blur.channels() << endl;
//	m=getEstimforBlur(img_no_blur);
//
//	cout << " max is " << m << endl;
//
//	m=getEstimforBlur(img_blur);
//
//	cout << " max is " << m << endl;
//
//	m=getEstimforBlur(img_test);
//
//	cout << " max is " << m << endl;
//
//	m=getEstimforBlur(blurred_no_blur);
//
//	cout << " max is " << m << endl;







}
