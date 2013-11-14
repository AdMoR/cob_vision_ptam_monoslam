#include "rgbd_grabber.h"
#include <boost/thread.hpp>
#include <fstream>
//#include "utilities.h"

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
namespace enc = sensor_msgs::image_encodings;
using namespace cv;

extern float g_focal_length,g_baseline,g_doff;


void dumpToFile(std::string frame_nb, float x, float y, float z, float rx, float ry, float rz, float rw, string filePath="/home/rmb-am/Slam_datafiles/measurements.txt",int p=0){

	ofstream myfile;
	myfile.open (filePath, ios::app);
	if(myfile.is_open()){
		myfile << frame_nb << " " << x << " " << y << " " << z << " " << rx << " " << ry << " " << rz << " " << rw <<  endl;
	}
	myfile.close();
}
//
//float depthToDisp(float depth)
//{
//  float scaled_disparity = g_focal_length/depth;
//  return scaled_disparity/g_baseline;
//}

namespace ScaViSLAM {
boost::mutex global_mutex;
bool new_frame;

cv::Mat global_rgb_img(cv::Size(640, 480), CV_8UC3);
cv::Mat global_disp_img(cv::Size(640, 480), CV_32F);





//void RgbdGrabber::kinect_callback(const ImageConstPtr& img_color, const ImageConstPtr& img_depth) {
//	boost::mutex::scoped_lock lock(global_mutex);
//	cv_bridge::CvImagePtr cv_ptr,cv_ptr2;
//
//	//double b=2.5,focal=532.,doff=109.;
//
//
//	try {
//		cv_ptr = cv_bridge::toCvCopy(img_color, enc::BGR8);
//		global_rgb_img = cv_ptr->image;
//
//		Mat show;
//		cv_ptr2 = cv_bridge::toCvCopy(img_depth, "32FC1" );
//
//
//		Mat depth;
//		(cv_ptr2->image).copyTo(depth);
//
//		Mat disparity=Mat::zeros(depth.rows,depth.cols,CV_32FC1),unit=Mat::ones(depth.rows,depth.cols,CV_32FC1);
//
//		//disparity=g_focal_length/depth/g_baseline;
//		//disparity=g_doff*unit-(unit/depth)*g_focal_length*g_baseline*(8);
//		disparity=(unit/depth)*g_focal_length*g_baseline;
//		//disparity=(unit/depth)*factor;
//
////		for(unsigned int i=0;i<10;i++){
////			for(unsigned int j=0;j<10;j++){
////				dumpToFile(" ",i+100,j+100,0,depth.at<float>(i+100,j+100),0,0,0,"/home/rmb-am/Slam_datafiles/depth.txt");
////			}
////		}
//	//	imshow("disparity",disparity);
//
//
//		global_disp_img=disparity;
//		new_frame = true;
//
//		}
//	catch (cv_bridge::Exception& e) {
//		ROS_ERROR("cv_bridge exception: %s", e.what());
//		return;
//	}
//
//}

void RgbdGrabber::kinect_callback(const ImageConstPtr& img_color,
		const stereo_msgs::DisparityImageConstPtr& img_depth) {
	boost::mutex::scoped_lock lock(global_mutex);
	cv_bridge::CvImagePtr cv_ptr;

	try {
		//ROS_ASSERT(img_mono->encoding == sensor_msgs::image_encodings::MONO8 && img_mono->step == img_mono->width);
		//cout << "before" << endl;
		cv_ptr = cv_bridge::toCvCopy(img_color, enc::BGR8);
		global_rgb_img = cv_ptr->image;
		//cout << "after" << endl;
		cv_ptr = cv_bridge::toCvCopy(img_depth->image, enc::TYPE_32FC1);
		global_disp_img = cv_ptr->image;
		// TODO: avoid copy
//		cv_bridge::CvImage out_msg;
//		//out_msg.header   = in_msg->header; // Same timestamp and tf frame as input image
//		out_msg.encoding = sensor_msgs::image_encodings::MONO8;
//		cv::Mat gray;
//		cv::cvtColor(global_rgb_img, gray, CV_BGR2GRAY);
//		out_msg.image = gray;
//		cv::imshow("kinect", out_msg.image);
//		cv::waitKey(2);
//		sensor_msgs::ImageConstPtr img_mono2;
//		//sensor_msgs::Image img_mono;
//		//out_msg.toImageMsg(img_mono);
//		img_mono2 = out_msg.toImageMsg();
//
//		if (first_frame_) {
//			initTracker(CVD::ImageRef(img_mono2->width, img_mono2->height));
//			first_frame_ = false;
//		} else {
//
//			CVD::BasicImage<CVD::byte> img_tmp(
//					(CVD::byte *) &img_mono2->data[0],
//					CVD::ImageRef(img_mono2->width, img_mono2->height));
//			CVD::copy(img_tmp, img_bw_);
//			bool tracker_draw = false;
//			static GVars3::gvar3<int> gvnDrawMap("DrawMap", 0,
//					GVars3::HIDDEN | GVars3::SILENT);
//			bool bDrawMap = mpMap->IsGood() && *gvnDrawMap;
//
//			if (ParamsAccess::fixParams->gui) {
//				CVD::copy(img_tmp, img_rgb_);
//				mGLWindow->SetupViewport();
//				mGLWindow->SetupVideoOrtho();
//				mGLWindow->SetupVideoRasterPosAndZoom();
//				tracker_draw = !bDrawMap;
//			}
//			mpTracker->TrackFrame(img_bw_, tracker_draw);
//			std::cout << mpMapMaker->getMessageForUser();
//
//			if (ParamsAccess::fixParams->gui) {
//				string sCaption;
//				if (bDrawMap) {
//					mpMapViewer->DrawMap(mpTracker->GetCurrentPose());
//					sCaption = mpMapViewer->GetMessageForUser();
//				} else {
//					sCaption = mpTracker->GetMessageForUser();
//				}
//				mGLWindow->DrawCaption(sCaption);
//				mGLWindow->DrawMenus();
//				mGLWindow->swap_buffers();
//				mGLWindow->HandlePendingEvents();
//			}
//		}

		// cv::imshow("kinect", global_rgb_img);
		Mat show;
		convertScaleAbs(  global_disp_img, show );
//		for( unsigned int i =0;i<10;i++){
//			for(unsigned int j=0; j<10;j++){
//				cout << global_disp_img.at<float>(200+i,200+j) << " " ;
//			}
//		}

		cv::imshow("kinect disp",show);
		//cv::waitKey(3);
		new_frame = true;
	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

}


void RgbdGrabber::initTracker(const CVD::ImageRef & size) {
	img_bw_.resize(size);
	img_rgb_.resize(size);
	cout << "first frame rgbd_tracker" << endl;
	mpCamera = new ATANCamera("kinect");
	mpMap = new ptam::Map;
	mpMapMaker = new MapMaker(*mpMap, *mpCamera);
	mpTracker = new Tracker(size, *mpCamera, *mpMap, *mpMapMaker);

	GVars3::GUI.RegisterCommand("exit", GUICommandCallBack, this);
	GVars3::GUI.RegisterCommand("quit", GUICommandCallBack, this);

	if (ParamsAccess::fixParams->gui) {
		mGLWindow = new GLWindow2(size, "PTAM");
		mpMapViewer = new MapViewer(*mpMap, *mGLWindow);
		GVars3::GUI.ParseLine("GLWindow.AddMenu Menu Menu");
		GVars3::GUI.ParseLine("Menu.ShowMenu Root");
		GVars3::GUI.ParseLine("Menu.AddMenuButton Root Reset Reset Root");
		GVars3::GUI.ParseLine(
				"Menu.AddMenuButton Root Spacebar PokeTracker Root");
		GVars3::GUI.ParseLine("DrawMap=0");
		GVars3::GUI.ParseLine(
				"Menu.AddMenuToggle Root \"View Map\" DrawMap Root");
	}

}

void RgbdGrabber::GUICommandCallBack(void *ptr, string sCommand, string sParams) {
	if (sCommand == "quit" || sCommand == "exit") {
		cout << "shutting down" << endl;
		ros::shutdown();
	}
}

bool RgbdGrabber::getFrame(cv::Mat * rgb, cv::Mat * disp) {
	boost::mutex::scoped_lock lock(global_mutex);
	if (new_frame) {
		*rgb = global_rgb_img;
		*disp = global_disp_img;
		new_frame = false;
		return true;
	}
	return false;
}

void RgbdGrabber::initialize() {
	first_frame_ = true;
	//= true; //TEST
}

//void cback(const ImageConstPtr& img_depth,	const stereo_msgs::DisparityImageConstPtr& img_disp){
//
//	boost::mutex::scoped_lock lock(global_mutex);
//	cv_bridge::CvImagePtr cv_ptr,cv_ptr2;
//	Mat di_img,de_img;
//
//	cv_ptr = cv_bridge::toCvCopy(img_depth, enc::TYPE_32FC1);
//	de_img = cv_ptr->image;
//	cv_ptr2 = cv_bridge::toCvCopy(img_disp->image, enc::TYPE_32FC1);
//	di_img = cv_ptr2->image;
//
//	Mat ratio = di_img.mul(de_img);
//
//	for( unsigned int i =0;i<10;i++){
//				for(unsigned int j=0; j<10;j++){
//					cout << ratio.at<float>(200+i,200+j)     ;
//				}
//			}
//	cout << endl;
//
//}

void RgbdGrabber::operator()() {
	typedef sync_policies::ApproximateTime<sensor_msgs::Image,stereo_msgs::DisparityImage> MySyncPolicy;
	typedef sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::Image> MySyncPolicy2;


	kinect_rgb_sub.subscribe(node_handle, "/camera/rgb/image_color", 1);
	kinect_depth_sub.subscribe(node_handle,"/camera/depth_registered/image_raw", 1);
 	kinect_disp_sub.subscribe(node_handle,"/camera/depth_registered/disparity", 1);
	 Synchronizer<MySyncPolicy> sync(MySyncPolicy(3), kinect_rgb_sub,kinect_disp_sub);
//	Synchronizer<MySyncPolicy> sync(MySyncPolicy(3), kinect_rgb_sub,kinect_disp_sub);
	Synchronizer<MySyncPolicy2> sync2(MySyncPolicy2(3), kinect_rgb_sub, kinect_depth_sub);

	sync.registerCallback(	boost::bind(&RgbdGrabber::kinect_callback, this, _1, _2));


	ros::spin();
}

}

