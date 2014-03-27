
#include <opencv2/opencv.hpp>
#include "rgbd_line.h"
#include <iostream>
#include "timer.h"
#include "utilities.h"
#include "mono_frontend.h"
#include "transformations.h"
using namespace std;
using namespace cv;



void quickLinebuild(vector<Line>& line_vect,vector<Vector6d>& plu, vector< vector<float> >& descriptors, vector<cv::Vec4i> line_pt,SE3& transform,int type,int id_start=0){

	line_vect.clear();

	for(int i=0;i<plu.size();i++){
		Line newLine;
		newLine.global_id=id_start+i;
		Matrix<double,4,4> inv_tf = transform.matrix().inverse();
		newLine.GTPlucker=toPlueckerVec(inv_tf*toPlueckerMatrix(plu[i])*inv_tf.transpose());
		newLine.GTPlucker.normalize();
		newLine.optimizedPluckerLines=toPlueckerVec(inv_tf*toPlueckerMatrix(plu[i])*inv_tf.transpose());
		newLine.optimizedPluckerLines.normalize();
		newLine.d_descriptor=descriptors[i];
		newLine.startingPoint2d=cv::Point(line_pt[i][0],line_pt[i][1]);
		newLine.endPoint2d=cv::Point(line_pt[i][2],line_pt[i][3]);
		newLine.linearForm=calculateLinearForm(newLine.startingPoint2d.x,newLine.startingPoint2d.y,newLine.endPoint2d.x,newLine.endPoint2d.y);
		newLine.linearForm.normalize();
		newLine.type=type;
		if(newLine.linearForm(2)<0)
			newLine.linearForm*=-1;
		line_vect.push_back(newLine);
	}


}


void construction(cv::Mat& frame,cv::Mat& s_img,cv::Mat& d_img,cv::Mat& oe_val, vector<Vec4i>& lines,vector<Line>& linesOnFrame,SE3& tf,int nb_sample_pts,int type=-1){

	vector<vector<cv::Point3f> > lines_pts;
	vector<Vector6d> plu_vec;
	vector<vector<float> > descriptor_vec;

	//Construction of the descriptors
		for(auto ptr=lines.begin();ptr!=lines.end();ptr++){
			vector<float> descriptor;
			vector<cv::Point3f>depth_vec;
			vector<pair<int,int>> pixelsOnLine = lineBresenham((*ptr)[0], (*ptr)[1], (*ptr)[2],(*ptr)[3]);
			if(findDepthDifference(frame,s_img,d_img,oe_val,cv::Point((*ptr)[1],(*ptr)[0]), cv::Point((*ptr)[3], (*ptr)[2]),pixelsOnLine,nb_sample_pts,descriptor,depth_vec,true,1,1)){
				descriptor_vec.push_back(descriptor);
				lines_pts.push_back(depth_vec);
			}
			else{
				//delete line instead
				descriptor_vec.push_back(descriptor);
			}
		}

		//Construction of the Plucker
		for(auto ptr=lines_pts.begin();ptr!=lines_pts.end();ptr++){
			Vec6f line_val;

			fitLine((*ptr),line_val,CV_DIST_L2,0,0.01,0.01);
			Vector3d p1 = Vector3d(line_val[3],
								line_val[4],
								line_val[5]),
				p2 = Vector3d(line_val[3] +10*line_val[0] ,
								line_val[4] +10*line_val[1] ,
								line_val[5]+10*line_val[2] );

			Vector6d plu = computePlueckerLineParameters(p1,p2);
			plu_vec.push_back(plu);
		}


		quickLinebuild(linesOnFrame,plu_vec,descriptor_vec,lines,tf,type);


}




void build_normal_map(Mat& d_img , Mat& frame, Mat& s_img, vector<Line>& linesOnFrame,SE3& tf,int nb_sample_pts, int vote_for_line=60,int int_pt=6){


	//Preparation of a lot of variables
	Mat Nx,Nxs,Nys,Ny,Gx,Gy,Gxx,Gxy,Gyx,Gyy,Gxx_2,Gxy_2,Gyx_2,Gyy_2,Gxshow,Gyshow,show,validity_map,test,oe,oe_val,gray,dd,NyScale,NxScale;
	vector<cv::Mat> oe_channels;
	vector<cv::Vec4i> lines,lines2,lines3;

	bool display = false;
	test=cv::Mat::zeros(d_img.rows,d_img.cols,CV_8UC1);
	Nys=cv::Mat::zeros(d_img.rows,d_img.cols,CV_8UC1);

	cv::cvtColor(frame, gray, CV_BGR2GRAY);

	Mat frame_copy = frame.clone();

	int dilation_size=1;
	cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
			                                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
			                                       cv::Point( dilation_size, dilation_size ) );


	//Start of rgb lines extraction
	double  durationA = cv::getTickCount();
	cv::Mat dst, blur, src;
	cv::Mat occluding_e,occluded_e;
	cv::equalizeHist(gray, src );
	cv::GaussianBlur( src, blur, cv::Size(9, 9), 2, 2 );
	cv::Canny(blur, dst, 40, 60);
	cv::dilate( dst, dst, element );
	cv::HoughLinesP(dst, lines2, 1, CV_PI/180, 65, 80, 1 );
	cout << "size of lines2 " << lines2.size()<< " in " << (cv::getTickCount()-durationA)/(cv::getTickFrequency())<< endl;


	//Start of occluding edges extraction
	durationA = cv::getTickCount();
	double durationB= cv::getTickCount();

	validPixelMap(d_img,validity_map);
	getOccluding(d_img,validity_map,oe_val,Nx,Ny,false,70,0.04);
	cv::split(oe_val,oe_channels);
	cv::threshold(oe_channels[0],occluding_e,10,255,THRESH_BINARY);
	cv::dilate( occluding_e,occluding_e, element,Point(-1,-1),1 );
	cv::HoughLinesP( occluding_e, lines, 1, CV_PI/180,  vote_for_line,vote_for_line,int_pt);

	cv::threshold(oe_channels[1],occluded_e,10,255,THRESH_BINARY);
	cv::dilate( occluded_e,occluded_e, element,Point(-1,-1),1 );
	//cv::erode( occluded_e, occluded_e, element ,Point(-1,-1),2);
	cv::HoughLinesP( occluded_e, lines3, 1, CV_PI/180,  vote_for_line,vote_for_line,int_pt);



	//Construction of the lines
	construction(frame_copy,s_img,d_img,oe_val,lines, linesOnFrame,tf,nb_sample_pts,1);
	construction(frame_copy,s_img,d_img,oe_val,lines3, linesOnFrame,tf,nb_sample_pts,2);



//	//HC extraction
//	convertScaleAbs(  Ny, NyScale );
//	GaussianBlur(NyScale,NyScale,Size(5,5),1,1);
//	Canny(NyScale,Gy,50,180);
//	cv::dilate( Gy, Gy, element,Point(-1,-1),1 );
//	cv::erode( Gy, Gy, element ,Point(-1,-1),1);
//	convertScaleAbs(  Gy, Gyshow );
//	vector<cv::Vec4i> lines3;
//	cv::HoughLinesP(Gyshow, lines3, 5, 3*CV_PI/180, 150, 80, 5 );
//
//	convertScaleAbs(  Nx, NxScale );
//	GaussianBlur(NxScale,NxScale,Size(5,5),1,1);
//	Canny(NxScale,Gx,50,180);
//	cv::dilate( Gx, Gx, element,Point(-1,-1),1 );
//	cv::erode( Gx, Gx, element ,Point(-1,-1),1);
//	convertScaleAbs(  Gx, Gxshow );
//	vector<cv::Vec4i> lines4;
//	cv::HoughLinesP(Gxshow, lines4, 5, 3*CV_PI/180, 150, 80, 5 );
//	convertScaleAbs( d_img, show );


//	//Draw lines if needed
//	for(auto ptr=lines3.begin();ptr!=lines3.end();ptr++){
//			cv::line(frame, cv::Point((*ptr)[0],(*ptr)[1]), cv::Point((*ptr)[2], (*ptr)[3]), Scalar(0,0,255), 1, 8);
//		}
//	for(auto ptr=lines.begin();ptr!=lines.end();ptr++){
//				cv::line(frame, cv::Point((*ptr)[0],(*ptr)[1]), cv::Point((*ptr)[2], (*ptr)[3]), Scalar(0,255,255), 1, 8);
//			}

//imshow("",frame);
//	waitKey(0);


}

float score(vector<float>& v1, vector<float>& v2){

	float ret=0;

	if(v1.size()==0 || v2.size()==0)
		return 1;

	for(int i=0;i<v1.size();i++){
			ret+= pow(v1[i]-v2[i],2);
	}

	return pow(ret,0.5);
}

float plucker_similarity(Vector6d& v1,Vector6d& v2){
	if(v1.dot(v2)<0)
		v2*=-1;

	float sum=0;
	for(int i=0;i<6;i++)
		sum+=pow(0.5*(v1(i)-v2(i)),2);
	return pow(sum,0.5);
}


Line find_closest(Line tracked, vector<Line>& candidates,map<int,Line> already_matched,double max_err=1.2, bool debug=false){

	map<float,Line> scores;

	for(int i=0;i<candidates.size();i++){
		if(candidates[i].type!=tracked.type)continue;
		float err = plucker_similarity(tracked.optimizedPluckerLines,candidates[i].optimizedPluckerLines) + score(tracked.d_descriptor,candidates[i].d_descriptor);
		scores.insert(make_pair(err,candidates[i]));
		if(debug) cout << err << " = " << plucker_similarity(tracked.optimizedPluckerLines,candidates[i].optimizedPluckerLines) << " + " <<  score(tracked.d_descriptor,candidates[i].d_descriptor);
		if(debug) cout << endl;
	}

	for(auto ptr = scores.begin();ptr!=scores.end();ptr++){
		if(already_matched.find(ptr->second.global_id)==already_matched.end() && ptr->first<max_err)
			return ptr->second;
	}

	Line l;
	l.optimizedPluckerLines=Vector6d();
	l.optimizedPluckerLines <<1,1,1,1,1,1;
	l.global_id=-1;

	return l;

}


int not_main(int argc, char* argv[]){

	//This main gives an example of the extraction and description of depth lines
	//The main function is build_normal_map for the extraction and description

	vector<vector<float> > descriptor_vec, descriptor_vec2;
	vector<cv::Vec4i> lines1, lines2;
	vector<vector<cv::Point3f> > pt_lines1, pt_lines2;
	vector<Vector6d> plu_vec_1,plu_vec_2;
	Mat d_img,d_img2;
	Mat hsv,hsv2;
	vector<Mat> hsv_channel,hsv_channel2;
	Mat frame=imread("/home/rmb-am/Slam_datafiles/training_img/rgb_3.png"),frame2=imread("/home/rmb-am/Slam_datafiles/training_img/rgb_4.png");;
	Mat f1=frame.clone(),f2=frame2.clone();

	cv::cvtColor(frame, hsv, CV_BGR2HSV);
	cv::cvtColor(frame2, hsv2, CV_BGR2HSV);
	split(hsv,hsv_channel);
	split(hsv2,hsv_channel2);

	//Process the second frame
	double duration=cv::getTickCount();
	getMatrixFromFile(d_img2,"/home/rmb-am/Slam_datafiles/training_img/d_4.yml");
	cout << "depth : " << (getTickCount()-duration)/getTickFrequency() << endl;
	//build_normal_map(d_img2,frame2,hsv_channel2[1],descriptor_vec2,pt_lines2,lines2,plu_vec_2,25);
	cout << "time for the normal map func : " << (getTickCount()-duration)/getTickFrequency() << endl;

	ofstream os;
	os.open("/home/rmb-am/Slam_datafiles/descriptor_rgbd.txt",ios::app);
	os << endl;
	os.close();

	//Process the first frame
	getMatrixFromFile(d_img,"/home/rmb-am/Slam_datafiles/training_img/d_3.yml");
	//build_normal_map(d_img,frame,hsv_channel[1],descriptor_vec,pt_lines1,lines1,plu_vec_1,25);




	double min,max;
	cv::Point minl,maxl;
	int j=0;


	//Compare descriptor scores
	for(auto it1=descriptor_vec2.begin();it1!=descriptor_vec2.end();it1++){
		int i=0,best_i=0;
		float best=100;
		vector<float> scores;
		for(auto it2=descriptor_vec.begin();it2!=descriptor_vec.end();it2++){
			scores.push_back(score(*it2,*it1));
			if(score(*it2,*it1)<best){
				best_i=i;
				best=score(*it2,*it1);
				cout << best << " this is new best " << endl;
			}
			i++;
		}

		//Write to file the scores
		ofstream os;
		os.open("/home/rmb-am/Slam_datafiles/descriptor_rgbd.txt",ios::app);
		os << j << "is matched with" << best_i << ", score : " << pow(best,0.5) << endl;
		os.close();



		Mat RGBF = frame.clone();
		Mat RGBF2 = frame2.clone();
		int k=0;
		cout << "display" << scores.size() << " " << lines1.size()<< endl;
		for(auto it2=descriptor_vec.begin();it2!=descriptor_vec.end();it2++){
			float the_score = scores[k];
			cout << the_score << endl;
			cv::line(RGBF, cv::Point(lines1[k][0],lines1[k][1]), cv::Point(lines1[k][2],lines1[k][3]) , Scalar(255*(best/the_score),0,255*(1.-best/the_score)) , 2, 8);
			k++;
		}
		cv::line(RGBF2, cv::Point(lines2[j][0],lines2[j][1]), cv::Point(lines2[j][2],lines2[j][3]) , Scalar(255,255,255) , 2, 8);

		j++;

	}





}
