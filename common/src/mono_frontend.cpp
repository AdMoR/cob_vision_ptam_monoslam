#include "mono_frontend.h"

#include <stdint.h>

#include <visiontools/accessor_macros.h>
#include <visiontools/performance_monitor.h>
#include <visiontools/stopwatch.h>

#include "transformations.h"
#include "backend.h"
#include "homography.h"
#include "matcher.hpp"
#include "maths_utils.h"
#include "quadtree.h"
#include "pose_optimizer.h"
#include "utilities.h"

static bool line_mode=true;

namespace ScaViSLAM {


cv_bridge::CvImage grayImage;
cv_bridge::CvImage colorImage;
char* window_name = "Edge Map";
int edgeThresh = 1;
int lowThreshold;
int highThreshold;
int const max_highThreshold = 300;
int const max_lowThreshold = 100;
int ratio = 2;
int kernel_size = 3;
cv::Mat dst2;

Matrix<double, 4, 4> transformMatrix ;

double rolling_average=500;

MonoFrontend::MonoFrontend(FrameData<StereoCamera> * frame_data,
		PerformanceMonitor * per_mon) :
	frame_data_(frame_data), per_mon_(per_mon),
			neighborhood_(new Neighborhood()), se3xyz_stereo_(frame_data->cam),
			unique_point_id_counter_(-1), tracker_(*frame_data_) {
}


void MonoFrontend::initialize(ros::NodeHandle & nh) {
	params_.newpoint_clearance = pangolin::Var<int>("newpoint_clearance", 2);
	params_.covis_thr = pangolin::Var<int>("frontend.covis_thr", 15);
	params_.num_frames_metric_loop_check = pangolin::Var<int>(
			"frontend.num_frames_metric_loop_check", 200);
	params_.new_keyframe_pixel_thr = pangolin::Var<int>(
			"frontend.new_keyframe_pixel_thr", 70);
	params_.new_keyframe_featuerless_corners_thr = pangolin::Var<int>(
			"frontend.new_keyframe_featuerless_corners_thr", 2);
	params_.save_dense_cloud = pangolin::Var<bool>("frontend.save_dense_cloud",
			true);
	params_.livestream =  pangolin::Var<bool>
		      ("framepipe.livestream",false);
	pangolin::Var<size_t> use_n_levels_in_frontent("use_n_levels_in_frontent",
			2);
	USE_N_LEVELS_FOR_MATCHING = use_n_levels_in_frontent;

	fast_grid_.resize(USE_N_LEVELS_FOR_MATCHING);

	for (int l = 0; l < USE_N_LEVELS_FOR_MATCHING; ++l) {
		int dim = std::max(3 - (int) (l * 0.5), 1);
		int num_cells = dim * dim;
		double inv_fac = pyrFromZero_d(1., l);

		int total_num_feat = 2000 * inv_fac * inv_fac;
		int num_feat_per_cell = total_num_feat / num_cells;
		int bound = std::max((int) num_feat_per_cell / 3, 10);

		fast_grid_.at(l) = FastGrid(frame_data_->cam_vec[l].image_size(),
				num_feat_per_cell, bound, 25, cv::Size(dim, dim));
	}
	first_frame_ = true;
	edges = calculateEdges();

	if(params_.livestream)
	{
	rgbd_camera_info_ = nh.subscribe("/camera/rgb/camera_info", 1, &MonoFrontend::camera_info_cb, this);
	client_3D_data = nh.serviceClient<msgpkg::point_cloud_server>("get_kinect_reg_3D_coord", true);//persistent connection
	ros::spin();
	}

}


void MonoFrontend::camera_info_cb(const sensor_msgs::CameraInfoConstPtr& rgbd_camera_info)
{
	camera_matrix(0,0) = rgbd_camera_info->K[0];
	camera_matrix(0,1) = rgbd_camera_info->K[1];
	camera_matrix(0,2) = rgbd_camera_info->K[2];
	camera_matrix(1,0) = rgbd_camera_info->K[3];
	camera_matrix(1,1) = rgbd_camera_info->K[4];
	camera_matrix(1,2) = rgbd_camera_info->K[5];
	camera_matrix(2,0) = rgbd_camera_info->K[6];
	camera_matrix(2,1) = rgbd_camera_info->K[7];
	camera_matrix(2,2) = rgbd_camera_info->K[8];
   rgbd_camera_info_.shutdown();
}


void MonoFrontend::computeSVDPluecker(vector<pair<int,int>> pixelsOnLine)
{
	//L*X=0 if point lies on line, else it is a plane  where L* is dual plücker matrix
/*
   [0      l34  l42   l23] [x]   [0]
   [-l34   0    l14  -l13] [y] = [0]        =>
   [-l42  -l14  0     l12] [z]   [0]
   [-l23  l13  -l12   0  ] [1]   [0]

SVD

[ 0   0   0  1  z  y] [l12]   [0]
[ 0  -1   z  0  0 -x] [l13]   [0]
[1    0  -y  0 -x  0] [l14] = [0]
[-z   y   0 -x  0  0] [l23]   [0]
                      [l24]
                      [l34]

   */
	//vector<pair<int,int>> pixelsOnLine = lineBresenham(p1x,p1y,p2x,p2y);
	//cout<<"pixelsOnLine: "<<pixelsOnLine.size()<<endl;
	Timer timer;
	timer.start();
	cv::Mat coordinates (cv::Mat::zeros(pixelsOnLine.size()*4,6,CV_32FC1));
	cv::Mat pluckerLine (cv::Mat::zeros(1,6,CV_32FC1));
	Vector6d pluckerLines;
	Vector3d vec;
	int iCoord = 0;
	int count = 0;
	for (vector<pair<int, int>>::iterator it = pixelsOnLine.begin(); it != pixelsOnLine.end(); it++)
	{
		if (count % 10 == 0)
		{
			//cout<<"x: "<<it->first<<"y: "<<it->second<<endl;
			if (request3DCoords(it->first, it->second,&vec))
			{
				//cout<<"a: "<<vec[0]<<" b: "<<vec[1]<<" c: "<<vec[2]<<endl;
				coordinates.at<float> (iCoord, 0) = 0;
				coordinates.at<float> (iCoord, 1) = 0;
				coordinates.at<float> (iCoord, 2) = 0;
				coordinates.at<float> (iCoord, 3) = 1;
				coordinates.at<float> (iCoord, 4) = vec[2];
				coordinates.at<float> (iCoord, 5) = vec[1];
				coordinates.at<float> (iCoord+1, 0) = 0;
				coordinates.at<float> (iCoord+1, 1) = -1;
				coordinates.at<float> (iCoord+1, 2) = vec[2];
				coordinates.at<float> (iCoord+1, 3) = 0;
				coordinates.at<float> (iCoord+1, 4) = 0;
				coordinates.at<float> (iCoord+1, 5) = -1*vec[0];
				coordinates.at<float> (iCoord+2, 0) = 1;
				coordinates.at<float> (iCoord+2, 1) = 0;
				coordinates.at<float> (iCoord+2, 2) = -1*vec[1];
				coordinates.at<float> (iCoord+2, 3) = 0;
				coordinates.at<float> (iCoord+2, 4) = -1*vec[0];
				coordinates.at<float> (iCoord+2, 5) = 0;
				coordinates.at<float> (iCoord+3, 0) = -1*vec[2];
				coordinates.at<float> (iCoord+3, 1) = vec[1];
				coordinates.at<float> (iCoord+3, 2) = 0;
				coordinates.at<float> (iCoord+3, 3) = -1*vec[0];
				coordinates.at<float> (iCoord+3, 4) = 0;
				coordinates.at<float> (iCoord+3, 5) = 0;
				iCoord=iCoord+4;
			}
		}
		count++;
	}
	if (iCoord < pixelsOnLine.size())
	{
		coordinates.resize(iCoord);
		cout<<"resize, iCoord: "<<iCoord<<" lineSize: "<<pixelsOnLine.size()<<endl;
	}

	cv::Mat sv; //singular values
	cv::Mat u; //left singular vectors
	cv::Mat vt; //right singular vectors, transposed, 3x3

	//if there have been valid coordinates available, perform SVD
	//last column of v = last row of v-transposed is x, so that y is minimal
	if (coordinates.rows > 5)
	{
		cv::SVD::compute(coordinates, sv, u, vt,cv::SVD::MODIFY_A);
		//cout<<vt<<endl;
		pluckerLine = vt.row(5);
		std::cout << "SVD performed, plücker Line: "<<pluckerLine<<endl;
	}
	timer.stop();
	std::cout << timer.getElapsedTimeInSec()<< " s computing SVD\n";

}

void MonoFrontend::computeLines(std::vector<Line> &linesOnCurrentFrame, SE3 & T_cur_from_w, bool firstFrame, int frame_id)
{
	cv::Mat dst, blur, src;
	grayImage.encoding = sensor_msgs::image_encodings::MONO8;
	grayImage.image = frame_data_->cur_left().uint8;
	colorImage.image = frame_data_->cur_left().color_uint8;
	cv::Mat firstRGBFrame = colorImage.image.clone();

	transformMatrix = T_cur_from_w.matrix();

	// Apply Histogram Equalization
	cv::equalizeHist( grayImage.image, src );

	cv::GaussianBlur( src, blur, cv::Size(9, 9), 2, 2 );

	cv::Canny(blur, dst, 40, 60);
	int dilation_size=1;
	cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
	                                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
	                                       cv::Point( dilation_size, dilation_size ) );
	cv::dilate( dst, dst, element );

	vector<cv::Vec4i> lines;
	//apply probabilistic Hough Transformation
	cv::HoughLinesP(dst, lines, 1, CV_PI/180, 80, 80, 0 );

    std::vector<cv::Rect> array;

  //  cout<<"map size: "<<tracked_lines.size()<<endl;
    linesOnCurrentFrame.reserve(lines.size());
    cout<<"hough detected lines size: "<<lines.size()<<endl;
//    int aux=0;
    for( size_t i = 0; i < lines.size(); i++ )
    {
        //cv::line( colorImage.image, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255,0,0), 2, 8 );

// cout<<"x1: "<<lines[i][0]<<" y1: "<<lines[i][1]<<" x2: "<<lines[i][2]<<" y2: "<<lines[i][3]<<endl;
    	vector<pair<int,int>> pixelsOnLine = lineBresenham(lines[i][0], lines[i][1], lines[i][2],lines[i][3]);
        std::vector<int> descriptor = computeLineDescriptorSSD(lines[i][0], lines[i][1], lines[i][2],lines[i][3], pixelsOnLine);
//        std::vector<int> descriptor = computeLineDescriptor(lines[i][0], lines[i][1], lines[i][2],lines[i][3], pixelsOnLine);
//        cout<<"descriptor size: "<<descriptor.size()<<endl;

//        for(std::vector<int>::size_type i = 0; i != descriptor.size(); i++) {
//        	cout<<"descriptor["<<i<<"]: "<<descriptor[i]<<endl;
//        }
//    	computeSVDPluecker(pixelsOnLine);
        Vector6d pluckerLines;
        Vector3d startingPoint;
        Vector3d endPoint;
	        if (request3DCoordsAndComputePlueckerParams(pluckerLines, startingPoint, endPoint, lines[i][0], lines[i][1], lines[i][2],lines[i][3]))
			{
//			cout << "plucker param: " << pluckerLines[0] << " " << pluckerLines[1] << " " << pluckerLines[2] << " "
//					<< pluckerLines[3] << " " << pluckerLines[4] << " " << pluckerLines[5] << endl;

	        	cv::line( colorImage.image, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255,0,0), 2, 8 );


				Line currentLine;
				currentLine.global_id=getNewUniquePointId(); //We use the same counter as there is no reason to create another
				currentLine.anchor_id=frame_id;
				currentLine.linearForm = calculateLinearForm(lines[i][0], lines[i][1], lines[i][2],lines[i][3]);
				currentLine.linearForm.normalize();
				currentLine.descriptor = descriptor;
				currentLine.pluckerLinesObservation = pluckerLines;
				currentLine.optimizedPluckerLines = pluckerLines;
				currentLine.count = 3;
				currentLine.startingPoint2d = cv::Point(lines[i][0], lines[i][1]);
				currentLine.endPoint2d = cv::Point(lines[i][2], lines[i][3]);
				currentLine.startingPoint3d = startingPoint;
				currentLine.endPoint3d = endPoint;
				linesOnCurrentFrame.push_back(currentLine);
				if (firstFrame)
				{
//					if(aux==0)
//					{
//					cv::line( firstRGBFrame, cv::Point(lines[i][0], lines[i][1]),
//						                cv::Point(lines[i][2], lines[i][3]), cv::Scalar(165,100,200), 2, 8 );
//					cv::imshow("1. lerroa", firstRGBFrame);
//					cv::waitKey(1);
//					++aux;
//					}
					//Previously uncommented and used for ADD_TO...
					//int id1 = getNewUniquePointId();
//					cout<<"id1: "<<id1<<endl;
//					cout<<"plücker: "<<currentLine.nonOptimizedPluckerLines<<endl;
					ADD_TO_MAP_LINE(currentLine.global_id, currentLine, &tracked_lines);

				}
			}
//	        else
//	        	cout <<""<<endl;
	        	//cout << "No 3D coordinates for line" << endl;
    }

//    cv::imshow("current lines", colorImage.image);
//    cv::waitKey(1);


}
 /**
 * Draws a line between two points p1(p1x,p1y) and p2(p2x,p2y).
 * This function is based on the Bresenham's line algorithm and is highly
 * optimized to be able to draw lines very quickly. There is no floating point
 * arithmetic nor multiplications and divisions involved. Only addition,
 * subtraction and bit shifting are used.
 */
vector<pair<int,int>>  MonoFrontend::lineBresenham(int p1x, int p1y, int p2x, int p2y)
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


int MonoFrontend::computeMeanIntensityOfNeighborhood(const int x, const int y, const int sizeOfNeighborhood)
{
	//sizeOfNeighborhood should odd number 3,5,7,9...
	assert(sizeOfNeighborhood >= 3);
	assert(sizeOfNeighborhood % 2 == 1);
	int sum, lastValue = 0;
	cv::Scalar tmp;
//	cout<<"sizeOfNeighbourhood: "<<sizeOfNeighborhood<<endl;
//	cout<<"x: "<<x<<" y: "<<y<<endl;
	int halfSize=(sizeOfNeighborhood/2); //fixme: better variable name
	for(int i = x-halfSize; i<=x+halfSize; i++)
	{
		for(int j= y-halfSize; j<=y+halfSize; j++)
		{
			//cout<<"new x: "<<i<<" new y: "<<j<<endl;
			if((i>=0 && i<640) && (j>=0 && j<480))
			{
			tmp = frame_data_->cur_left().uint8.at<uchar> (j, i);
			lastValue = tmp.val[0];
			sum=sum+tmp.val[0];
			}
			else
			{
				sum = sum + lastValue;
			}

		}

	}
	int mean = sum/(sizeOfNeighborhood*sizeOfNeighborhood);
//	cout<<"mean: "<<mean<<endl;
	return mean;
}
std::vector<int> MonoFrontend::computeLineDescriptorSSD(int x1,int y1, int x2, int y2, vector<pair<int,int>> & pixelsOnLine)
{
	if (x1 > x2) // Swap points if p1 is on the right of p2
	{
		swap(x1, x2);
		swap(y1, y2);
	}
	int dx = x2 - x1;
	int dy = y2 - y1;
	float norm = sqrt(dx*dx + dy*dy);
	int zeroCount = 0;
	int oneCount = 0;
	std::vector<int> descriptor;
	//cv::Mat tmp = frame_data_->cur_left().uint8;
	if(norm>0.0)
	{
		float normalizedDirectionVectorX = dx / norm;
		float normalizedDirectionVectorY = dy / norm;
		float normalizedNormalVectorX = normalizedDirectionVectorY;
		float normalizedNormalVectorY = -normalizedDirectionVectorX;
		descriptor.reserve(pixelsOnLine.size());
//		cout<<"pixels on line"<<pixelsOnLine.size()<<endl;
		int difference=0;
		std::pair<int,int> * pairArr;
		pairArr = new std::pair<int,int> [3];
		int yTopParallelLine, xTopParallelLine, yDownParallelLine, xDownParallelLine, minus1y, minus1x, plus1y, plus1x = 0;
		int lastValTop, lastValDown=0;
		int topLineSum, downLineSum=0;
		for (vector<pair<int, int>>::iterator it = pixelsOnLine.begin(); it != pixelsOnLine.end(); it++)
		{
//			cout << "x: " << it->first << " y: " << it->second << endl;
			int topBlockSum=0;
			int downBlockSum=0;

			if(it==pixelsOnLine.begin())
			{
				minus1y = it->second - normalizedDirectionVectorY*1;
				minus1x = it->first - normalizedDirectionVectorX*1;
				vector<pair<int, int>>::iterator plus1it = std::next(it);
				pairArr[0] = make_pair(minus1x,minus1y);
				pairArr[1] = make_pair(it->first,it->second);
				pairArr[2] = make_pair(plus1it->first,plus1it->second);

			}
			else if(next(it)==pixelsOnLine.end())
			{
				vector<pair<int, int>>::iterator minus1it = std::prev(it);
				plus1y = it->second + normalizedDirectionVectorY*1;
				plus1x = it->first + normalizedDirectionVectorX*1;
				pairArr[0] = make_pair(minus1it->first,minus1it->second);
				pairArr[1] = make_pair(it->first,it->second);
				pairArr[2] = make_pair(plus1x,plus1y);
			}
			else
			{
				vector<pair<int, int>>::iterator minus1it = std::prev(it);
				vector<pair<int, int>>::iterator plus1it = std::next(it);
				pairArr[0] = make_pair(minus1it->first,minus1it->second);
				pairArr[1] = make_pair(it->first,it->second);
				pairArr[2] = make_pair(plus1it->first,plus1it->second);
			}
			for(int arrayIndex=0;arrayIndex<3;arrayIndex++)
			{
				//cout<<"minusx: "<<pairArr[arrayIndex].first<<" minusy: "<<pairArr[arrayIndex].second<<endl;
				for (int i = 1; i <= 5; i++)
				{
					yTopParallelLine = pairArr[arrayIndex].second + normalizedNormalVectorY * i;
					xTopParallelLine = pairArr[arrayIndex].first  + normalizedNormalVectorX * i;
					//cout << "xTopParallelLine: " << xTopParallelLine << " yTopParallelLine: " << yTopParallelLine
					//		<< endl;
					yDownParallelLine = pairArr[arrayIndex].second - normalizedNormalVectorY * i;
					xDownParallelLine = pairArr[arrayIndex].first - normalizedNormalVectorX * i;
					if((xTopParallelLine>=0 && xTopParallelLine<640) && (yTopParallelLine>=0 && yTopParallelLine<480))
					{
					cv::Scalar a = frame_data_->cur_left().uint8.at<uchar> (yTopParallelLine, xTopParallelLine);
					lastValTop = a.val[0];
					topBlockSum = topBlockSum + a.val[0];
					}else{ //if we are accesing out of frame, use last good value
					//	cout<<"catch a block computeLineDescriptorSSD, trying to access out of frame with x: "<<xTopParallelLine<<" y: "<<yTopParallelLine<<endl;
						topBlockSum = topBlockSum + lastValTop;
					}

					if((xDownParallelLine>=0 && xDownParallelLine<640) && (yDownParallelLine>=0 && yDownParallelLine<480))
					{
					cv::Scalar b = frame_data_->cur_left().uint8.at<uchar> (yDownParallelLine, xDownParallelLine);
					lastValDown = b.val[0];
					downBlockSum = downBlockSum + b.val[0];
					}else {
					//	cout<<"catch b block computeLineDescriptorSSD, trying to access out of frame with x: "<<xDownParallelLine<<" y: "<<yDownParallelLine<<endl;
						downBlockSum = downBlockSum + lastValDown;
					}




				}
				//cout<<"topSum: "<<topSum<<endl;
				//cout<<"downSum: "<<downSum<<endl;
			}
			topLineSum = topLineSum + topBlockSum;
			downLineSum = downLineSum + downBlockSum;
			difference = downBlockSum - topBlockSum;
//			cout<<"difference: "<<difference<<endl;
			descriptor.push_back(difference);
		}
		//define direction, the brighter side, so we identify line even if it is rotated 180 degreees
		if(topLineSum<downLineSum)
		{
			//cout<<"changing line direction"<<endl;
			std::reverse(descriptor.begin(),descriptor.end());
		}

	}
	else
	{
		cout<<"division by zero when calculating normalized direction vector of a line"<<endl;
	}
//	cout<<"descriptor.size(): "<<descriptor.size()<<endl;
	return descriptor;
}

std::vector<int> MonoFrontend::computeLineDescriptor(int x1,int y1, int x2, int y2, vector<pair<int,int>> & pixelsOnLine)
{
	if (x1 > x2) // Swap points if p1 is on the right of p2
	{
		swap(x1, x2);
		swap(y1, y2);
	}
	int dx = x2 - x1;
	int dy = y2 - y1;
	float norm = sqrt(dx*dx + dy*dy);
	int zeroCount = 0;
	int oneCount = 0;
	std::vector<int> descriptor;
	//cv::Mat tmp = frame_data_->cur_left().uint8;
	if(norm>0.0)
	{
	float normalizedDirectionVectorX = dx / norm;
	float normalizedDirectionVectorY = dy / norm;
	float normalizedNormalVectorX = normalizedDirectionVectorY;
	float normalizedNormalVectorY = -normalizedDirectionVectorX;
	descriptor.reserve(pixelsOnLine.size());
	for (vector<pair<int, int>>::iterator it = pixelsOnLine.begin(); it != pixelsOnLine.end(); ++it)
	{
//		cout << "x: " << it->first << " y: " << it->second << endl;
		int y5 = it->second + normalizedNormalVectorY * 5;
		int x5 = it->first + normalizedNormalVectorX * 5;
		int minus5y = it->second - normalizedNormalVectorY * 5;
		int minus5x = it->first - normalizedNormalVectorX * 5;
		//todo:check if the new lines are inside image matrix

//		cout << "x5: " << x5 << " y5: " << y5 << endl;
//		cout << "minus5x: " << minus5x << " minus5y: " << minus5y << endl;
//		cv::circle(tmp, cv::Point(x5, y5),2,cv::Scalar(255,0,0));
//		cv::circle(tmp, cv::Point(minus5x, minus5y),2,cv::Scalar(0,255,0));

		int meanIntensityOfNeighborhood_a = computeMeanIntensityOfNeighborhood(x5, y5, 15);

		int meanIntensityOfNeighborhood_b = computeMeanIntensityOfNeighborhood(minus5x, minus5y, 15);

//		cv::Scalar a = frame_data_->cur_left().uint8.at<uchar> (y5, x5);
//		cv::Scalar b = frame_data_->cur_left().uint8.at<uchar> (minus5y, minus5x);
//		cout << "w(a): " << a.val[0] << endl;
//		cout << "w(b): " << b.val[0] << endl;
		if (meanIntensityOfNeighborhood_a > meanIntensityOfNeighborhood_b)
		{
			descriptor.push_back(1);
			++oneCount;
		}
		else
		{
			descriptor.push_back(0);
			++zeroCount;
		}
	}
//	cout<<"oneCount: "<<oneCount<<endl;
//	cout<<"zeroCount: "<<zeroCount<<endl;
	//define direction, the brighter side, so we identify line even if it is rotated 180 degreees
	if(oneCount<zeroCount)
	{
		//cout<<"changing line direction"<<endl;
		for (vector<int>::iterator it = descriptor.begin(); it != descriptor.end(); it++)
		{
			int i = it - descriptor.begin();
			if (descriptor[i] == 1)
			{

				descriptor[i] = 0;
			}
			else if (descriptor[i] == 0)
			{
				descriptor[i] = 1;
			}
		}
	}

	}
	else
	{
		cout<<"division by zero when calculating normalized direction vector of a line"<<endl;
	}
//	cv::imshow("test1", tmp);
//	cv::waitKey(1);
	return descriptor;

}

bool MonoFrontend::request3DCoords(int x, int y, Vector3d *output)
{


	if(params_.livestream)
	{
	//ask kinect registered pointcloud the values x,y,z for our u,v image coordinates
		msgpkg::point_cloud_server pcl_server;
		pcl_server.request.x = x;
		pcl_server.request.y = y;
		if (client_3D_data.call(pcl_server))
		{
			if (!std::isnan(pcl_server.response.z))
			{
				(*output)[0] = pcl_server.response.x;
				(*output)[1] = pcl_server.response.y;
				(*output)[2] = pcl_server.response.z;
				return true;
			}
		}
		else
		{
			ROS_ERROR("Failed to call service get_kinect_reg_3D_coord");
		}
		return false;
	}
	else
	{
		const StereoCamera & cam = frame_data_->cam;
		Vector2d uv_pyr(x,y);
		Vector2i uv_pyri = uv_pyr.cast<int> ();

		double disp = interpolateDisparity(frame_data_->disp, uv_pyri, 0);
		if (disp > 0)
		{
			Vector2i uvi = zeroFromPyr_2i(uv_pyri, 0);
			Vector3d uvu_pyr = Vector3d(uv_pyr[0], uv_pyr[1], uv_pyr[0] - disp);

			Vector3d uvu_0 = zeroFromPyr_3d(uvu_pyr, 0);

			Vector3d xyz_cur = cam.unmap_uvu(uvu_0);

			cout<<"xyz_cur[0]: "<<xyz_cur[0]<<" xyz_cur[1]: "<<xyz_cur[1]<<" xyz_cur[2]: "<<xyz_cur[2]<<endl;
			if (!std::isnan(xyz_cur[2]))
			{
				(*output)[0] = xyz_cur[0];
				(*output)[1] = xyz_cur[1];
				(*output)[2] = xyz_cur[2];
				return true;
			}
		}
		return false;

		//frame_data_->disp.
	}
}

bool MonoFrontend::request3DCoordsAndComputePlueckerParams(Vector6d &pluckerLines, Vector3d &startingPoint, Vector3d &endPoint, int x1, int y1, int x2, int y2)
{
	if (x1 > x2) // Swap points if p1 is on the right of p2
	{
		swap(x1, x2);
		swap(y1, y2);
	}
	int dx = x2 - x1;
	int dy = y2 - y1;
	float norm = sqrt(dx*dx + dy*dy);
	float normalizedDirectionVectorX = dx / norm;
	float normalizedDirectionVectorY = dy / norm;
	int count=0;
	bool point1Found=false;
	//cout<<"initial points: ("<<x1<<", "<<y1<<"), ("<<x2<<", "<<y2<<")"<<endl;
	while(!point1Found && count<5*6)
	{
		if(request3DCoords(x1, y1, &startingPoint))
		{
			point1Found=true;
			//cout<<"found starting points: ("<<x1<<", "<<y1<<")"<<endl;
			//cout<<"starting point "<<*startingPoint<<endl;
		}
		else
		{
		x1= x1 + /*5*/normalizedDirectionVectorX;
		y1= y1 + /*5*/normalizedDirectionVectorY;
		++count;
		//cout<<"starting points: ("<<x1<<", "<<y1<<")"<<endl;
		}
	}

	count=0;
	while(point1Found && count<5*6)
	{
		if(request3DCoords(x2, y2,&endPoint))
		{
			//cout<<"found ending points: ("<<x2<<", "<<y2<<")"<<endl;
			//cout<<"endPoint "<<*endPoint<<endl;

			//cout << startingPoint[0] << " " << startingPoint[1] << " " << startingPoint[2] << "before " << endl;

	          Vector4d transformedStartingPoint = transformMatrix.inverse()*toHomogeneousCoordinates(startingPoint);
	          Vector4d transformedEndingPoint = transformMatrix.inverse()*toHomogeneousCoordinates(endPoint);

	          startingPoint[0] = transformedStartingPoint[0]/transformedStartingPoint[3];
	          startingPoint[1] = transformedStartingPoint[1]/transformedStartingPoint[3];
	          startingPoint[2] = transformedStartingPoint[2]/transformedStartingPoint[3];
	          endPoint[0] = transformedEndingPoint[0]/transformedEndingPoint[3];
	          endPoint[1] = transformedEndingPoint[1]/transformedEndingPoint[3];
	          endPoint[2] = transformedEndingPoint[2]/transformedEndingPoint[3];

	          //cout << startingPoint[0] << " " << startingPoint[1] << " " << startingPoint[2] << "after "<< endl;


			pluckerLines = computePlueckerLineParameters(startingPoint,endPoint);
			return true;
		}
		else
		{
			x2= x2 - /*5*/normalizedDirectionVectorX;
			y2= y2 - /*5*/normalizedDirectionVectorY;
			++count;
			//cout<<"ending points: ("<<x2<<", "<<y2<<")"<<endl;
		}
	}
//	cout<<"no points found"<<endl;
	return false;
}


void MonoFrontend::initTracker(const CVD::ImageRef & size) {
	img_bw_.resize(size);
	img_rgb_.resize(size);
	cout << "initTracker" << endl;
	mpCamera = new ATANCamera("kinect");
	mpMap = new ptam::Map;
	mpMapMaker = new MapMaker(*mpMap, *mpCamera);
	mpTracker = new Tracker(size, *mpCamera, *mpMap, *mpMapMaker);

	//fixme: windows crashes because it is in the same thread as scavislams main opengl window
	if (ParamsAccess::fixParams->gui) {
		mGLWindow = new GLWindow2(size, "PTAM");
		mpMapViewer = new MapViewer(*mpMap, *mGLWindow);
		GVars3::GUI.ParseLine("GLWindow.AddMenu Menu Menu");
		GVars3::GUI.ParseLine("Menu.ShowMenu Root");
		GVars3::GUI.ParseLine("Menu.AddMenuButton Root Reset Reset Root");
		GVars3::GUI.ParseLine("Menu.AddMenuButton Root Spacebar PokeTracker Root");
		GVars3::GUI.ParseLine("DrawMap=0");
		GVars3::GUI.ParseLine("Menu.AddMenuToggle Root \"View Map\" DrawMap Root");
	}

}

void MonoFrontend::recomputeFastCorners(const Frame & frame,
		ALIGNED<QuadTree<int> >::vector * feature_tree) {
	int num_levels = frame.cell_grid2d.size();
	feature_tree->resize(num_levels);

	for (int level = 0; level < num_levels; ++level) {
		feature_tree->at(level) = QuadTree<int> (Rectangle(0, 0, frame_data_->cam_vec.at(level).width(),frame_data_->cam_vec.at(level).height()), 1);
		FastGrid::detect(frame.pyr.at(level), frame.cell_grid2d.at(level),&(feature_tree->at(level)));
	}
}

void MonoFrontend::processFirstFrame() {
	frameCounter=0;
	draw_data_.clear();
	T_cur_from_actkey_ = SE3();

	ALIGNED<QuadTree<int> >::vector feature_tree;

	per_mon_->start("dense point cloud");

	actkey_id = getNewUniquePointId();
	FrontendVertex vf;
	vf.T_me_from_w = T_cur_from_actkey_;
	neighborhood_->vertex_map.insert(make_pair(actkey_id, vf));

	if (frame_data_->have_disp_img) {
		vector<cv::Mat> hsv_array(3);
		hsv_array[0] = cv::Mat(frame_data_->disp.size(), CV_8UC1);

		frame_data_->disp.convertTo(hsv_array[0], CV_8UC1, 5., 0.);
		hsv_array[1] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);
		hsv_array[2] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);

		cv::Mat hsv(frame_data_->disp.size(), CV_8UC3);
		cv::merge(hsv_array, hsv);
		cv::cvtColor(hsv, frame_data_->color_disp, CV_HSV2BGR);
#ifdef SCAVISLAM_CUDA_SUPPORT
		frame_data_->gpu_disp_32f.upload(frame_data_->disp);
#endif
	} else {
#ifdef SCAVISLAM_CUDA_SUPPORT
		calcDisparityGpu();
#else
		calcDisparityCpu();
#endif
	}

	Frame kf = Frame(frame_data_->cur_left().pyr_uint8, frame_data_->disp).clone();

	per_mon_->start("fast");
	computeFastCorners(5, &feature_tree, &kf.cell_grid2d);
	per_mon_->stop("fast");

	std::vector<Line> linesOnCurrentFrame;
	//Here T_cur_from_actkey = T_cur_from_w = Id
	computeLines(linesOnCurrentFrame,T_cur_from_actkey_, true,actkey_id);

	addNewPoints(actkey_id, feature_tree);

	AddToOptimzerPtr to_optimizer(new AddToOptimzer(true));
	to_optimizer->newkey_id = actkey_id;
	to_optimizer->kf = kf;

	assert(to_optimizer->kf.cell_grid2d.size()>0);

	//make sure keyframe is added before pushing to optimiser_stack!!
	keyframe_map.insert(make_pair(actkey_id, kf));
	keyframe_id2num.insert(make_pair(actkey_id, keyframe_id2num.size()));
	keyframe_num2id.push_back(actkey_id);

	to_optimizer_stack.push(to_optimizer);

#ifdef SCAVISLAM_CUDA_SUPPORT
	tracker_.computeDensePointCloudGpu(T_cur_from_actkey_);
#else
	tracker_.computeDensePointCloudCpu(T_cur_from_actkey_);
#endif
	per_mon_->stop("dense point cloud");
}

//new lines are only added when a new keyframe is added
void MonoFrontend::addNewLinesToKeyFrame(std::vector<Line> &linesOnCurrentFrame, AddToOptimzerPtr & to_optimizer,
		std::vector<int> &localIDsofNewLinesToBeAdded, const SE3 &T_cur_from_w)
{
	int index = 0;
	for (auto iter = linesOnCurrentFrame.begin(); iter != linesOnCurrentFrame.end(); ++iter)
	{

//		for (auto b : localIDsofNewLinesToBeAdded)
//		{
//			if(index==b)
//			{
				//cout<<"insertin new line, index: "<<index<<endl;
				int newLineId = (*iter).global_id;
				(*iter).optimizedPluckerLines=toPlueckerVec(T_cur_from_w.matrix().inverse()*toPlueckerMatrix((*iter).optimizedPluckerLines)*T_cur_from_w.matrix().inverse().transpose());
				//GuidedMatcher<StereoCamera>::drawLine((*iter).linearForm, colorImage.image, "new", cv::Scalar(0,186,86),false);
				ADD_TO_MAP_LINE(newLineId, (*iter), &tracked_lines);

//				break;
//			}
//			if(index<b)
//			{
//				break;
//			}
//		}
//		++index;
	}
	to_optimizer->tracked_lines=tracked_lines;
}


void MonoFrontend::updateOptimizedPluckerParameters(tr1::unordered_map<int,Line> &tracked_lines, LineTable &tracked_lines_result)
{
	//cout<<"tracked_lines_result.size(): "<<tracked_lines_result.size()<<endl;
	  if(tracked_lines_result.size()>=tracked_lines.size())
	  {
		  for (auto it = tracked_lines.begin(); it != tracked_lines.end(); ++it)
		  {
			  //cout<<"replacing line "<<(*it).first;

			  if(GET_MAP_ELEM_IF_THERE((*it).first, &tracked_lines_result))
			  {
				  Line & l = GET_MAP_ELEM_REF((*it).first, &tracked_lines_result);
				  //cout<<" old values: "<<(*it).second.optimizedPluckerLines<<" with new: "<< l.optimizedPluckerLines<<endl;
				  (*it).second.optimizedPluckerLines = l.optimizedPluckerLines;
			  }
		  }
	  }
}

bool MonoFrontend::processFrame(bool * is_frame_dropped) {
	cv_bridge::CvImage out_msg;
	out_msg.encoding = sensor_msgs::image_encodings::MONO8;
//	out_msg.image = frame_data_->cur_left().uint8;
//	sensor_msgs::ImageConstPtr img_mono;
//	img_mono = out_msg.toImageMsg();
//	if (first_frame_) {
//		initTracker(CVD::ImageRef(img_mono->width, img_mono->height));
//		first_frame_ = false;
//	} else {
//		// TODO: avoid copy
//		CVD::BasicImage<CVD::byte> img_tmp((CVD::byte *) &img_mono->data[0],
//				CVD::ImageRef(img_mono->width, img_mono->height));
//		CVD::copy(img_tmp, img_bw_);
//		bool tracker_draw = false;
//		static GVars3::gvar3<int> gvnDrawMap("DrawMap", 0,
//				GVars3::HIDDEN | GVars3::SILENT);
//		bool bDrawMap = mpMap->IsGood() && *gvnDrawMap;
//
//		if (ParamsAccess::fixParams->gui) {
//			CVD::copy(img_tmp, img_rgb_);
//			mGLWindow->SetupViewport();
//			mGLWindow->SetupVideoOrtho();
//			mGLWindow->SetupVideoRasterPosAndZoom();
//			tracker_draw = !bDrawMap;
//		}
//		mpTracker->TrackFrame(img_bw_, tracker_draw);
//		std::cout << mpMapMaker->getMessageForUser();
//
//		if (ParamsAccess::fixParams->gui) {
//			string sCaption;
//			if (bDrawMap) {
//				mpMapViewer->DrawMap(mpTracker->GetCurrentPose());
//				sCaption = mpMapViewer->GetMessageForUser();
//			} else {
//				sCaption = mpTracker->GetMessageForUser();
//			}
//			mGLWindow->DrawCaption(sCaption);
//			mGLWindow->DrawMenus();
//			mGLWindow->swap_buffers();
//			mGLWindow->HandlePendingEvents();
//		}
//	}

	draw_data_.clear();

	//Get the estimation of the blurriness of the image
//	double m =getEstimforBlur(frame_data_->cur_left().pyr_uint8[0]);
//	cout << rolling_average << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << m << endl;
//	if(m<0.75*rolling_average)
//		imshow("BLUUUUUUUUUUUUUUUUR",frame_data_->cur_left().pyr_uint8[0]);
//	rolling_average=(4*rolling_average+m)/5;
	//Could be used to stop here the processing of the frame and keep the old values


	const ALIGNED<StereoCamera>::vector & cam_vec = frame_data_->cam_vec;

	per_mon_->start("dense tracking");
#ifdef SCAVISLAM_CUDA_SUPPORT
	tracker_.denseTrackingGpu(&T_cur_from_actkey_);
#else
	tracker_.denseTrackingCpu(&T_cur_from_actkey_); //updates SE3 transformation between old key frame and new key frame
#endif
	per_mon_->stop("dense tracking");

	per_mon_->start("stereo");

	if (frame_data_->have_disp_img) {
		vector<cv::Mat> hsv_array(3);
		hsv_array[0] = cv::Mat(frame_data_->disp.size(), CV_8UC1);

		frame_data_->disp.convertTo(hsv_array[0], CV_8UC1, 5., 0.);
		hsv_array[1] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);
		hsv_array[2] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);

		cv::Mat hsv(frame_data_->disp.size(), CV_8UC3);
		cv::merge(hsv_array, hsv);
		cv::cvtColor(hsv, frame_data_->color_disp, CV_HSV2BGR);
#ifdef SCAVISLAM_CUDA_SUPPORT
		frame_data_->gpu_disp_32f.upload(frame_data_->disp);
#endif
	} else {
#ifdef SCAVISLAM_CUDA_SUPPORT
		calcDisparityGpu();
#else
		calcDisparityCpu();
#endif
	}
	per_mon_->stop("stereo");


	per_mon_->start("fast");
	//
	ALIGNED<QuadTree<int> >::vector feature_tree;
	cur_frame_ = Frame(frame_data_->cur_left().pyr_uint8, frame_data_->disp);
	computeFastCorners(6, &feature_tree, &cur_frame_.cell_grid2d);
	per_mon_->stop("fast");
//	cv::imshow("disp",frame_data_->disp);
//    cv::imshow("color", frame_data_->cur_left().color_uint8);
//    cv::imshow("mono", frame_data_->cur_left().uint8);
//    cv::waitKey(2);
	std::vector<Line> linesOnCurrentFrame;
	Timer t1;
	t1.start();
	SE3 T_act_from_w = GET_MAP_ELEM(actkey_id,neighborhood_->vertex_map).T_me_from_w;
	SE3 T_cur_from_world = T_cur_from_actkey_*T_act_from_w;
	computeLines(linesOnCurrentFrame,T_cur_from_world, false,actkey_id);
	t1.stop();
	std::cout << t1.getElapsedTimeInSec()<< " s for computeLines\n";
	timercount1=timercount1+t1.getElapsedTimeInSec();
	frameCounter++;
	if (frameCounter==100)
	{
		cout<<"average time for computeLines: "<<timercount1/100<<endl;
		timercount1=0;
		//frameCounter=0;
	}


	per_mon_->start("match");
	ScaViSLAM::TrackData<3> track_data;
	int num_new_feat_matched=5;
	SE3 T_anchorkey_from_w;
	bool matched_enough_features = matchAndTrack(feature_tree, //input is feature_tree
			&track_data, &num_new_feat_matched, T_anchorkey_from_w);
	per_mon_->stop("match");

	if (matched_enough_features == false) //TODO add the info with matched lines
		return false;



	per_mon_->start("process points");
	PointStatistics point_stats(USE_N_LEVELS_FOR_MATCHING);
	tr1::unordered_set<CandidatePoint3Ptr> matched_new_feat;
	ALIGNED<QuadTree<int> >::vector point_tree(USE_N_LEVELS_FOR_MATCHING);
	for (int l = 0; l < USE_N_LEVELS_FOR_MATCHING; ++l) {
		point_tree.at(l) = QuadTree<int> (
				Rectangle(0, 0, cam_vec[l].width(), cam_vec[l].height()), 1);
	}

	AddToOptimzerPtr to_optimizer = processMatchedPoints(track_data,
			num_new_feat_matched, &point_tree, &matched_new_feat, &point_stats); /* if (id_obs.point_id<num_new_feat_matched)
	 calculates to_optimizer->new_point_list else to_optimizer->track_point_list*/

	std::vector<int>  localIDsofNewLinesToBeAdded;
	Timer t2;
	t2.start();

	updateOptimizedPluckerParameters(tracked_lines, neighborhood_->tracked_lines_result);

	SE3 T_cur_from_w;
	GuidedMatcher<StereoCamera>::lineMatcher(linesOnCurrentFrame, tracked_lines,
			T_cur_from_actkey_, actkey_id, neighborhood_->vertex_map, camera_matrix,
			&colorImage.image, edges, this, to_optimizer, localIDsofNewLinesToBeAdded,
			T_cur_from_w);
	t2.stop();



	//Dump the transform to the anchorKf
	tf::Vector3 v=tf::Vector3(T_cur_from_actkey_.matrix()(0,3),T_cur_from_actkey_.matrix()(1,3),T_cur_from_actkey_.matrix()(2,3));
	tf::Quaternion q(T_cur_from_actkey_.so3().unit_quaternion().x(),T_cur_from_actkey_.so3().unit_quaternion().y(),T_cur_from_actkey_.so3().unit_quaternion().z(),T_cur_from_actkey_.so3().unit_quaternion().w());
	stringstream ss;
	ss << actkey_id << "-" << getNewUniquePointId() ;
	dumpToFile(ss.str(), v.x(), v.y(), v.z(), q.x(), q.y(), q.z(), q.w(), "/home/rmb-am/Slam_datafiles/map/aux_map.txt");
	//Dump line info
	 getTrackedLinesToFile(tracked_lines);



	std::cout << t2.getElapsedTimeInSec()<< " s for LineMatcher\n";
	timercount2=timercount2+t2.getElapsedTimeInSec();
	if (frameCounter==100)
	{
		cout<<"average time for LineMatcher: "<<timercount2/100<<endl;
		timercount2=0;
		frameCounter=0;
	}

	per_mon_->stop("process points");

	per_mon_->start("drop keyframe");

	int other_id = -1;
	SE3 T_cur_from_other;

	ALIGNED<QuadTree<int> >::vector other_point_tree;
	PointStatistics other_stat(USE_N_LEVELS_FOR_MATCHING);
	if (shallWeSwitchKeyframe(to_optimizer->track_point_list, &other_id,to_optimizer->tracked_lines,
			&T_cur_from_other, &other_point_tree, &other_stat)) //decide if we switch keyframes, last two parameters are not used in function!
	{
		actkey_id = other_id;
		T_cur_from_actkey_ = T_cur_from_other;

	} else {

		*is_frame_dropped = shallWeDropNewKeyframe(point_stats); /* if the incremental translation/parallax exceeded
		 a threshold, or the number of tracked feature dropped below a critical limit, current video frame is added as a new keyframe Vi to the graph
		 */
		if (*is_frame_dropped) {
			addNewLinesToKeyFrame(linesOnCurrentFrame, to_optimizer, localIDsofNewLinesToBeAdded, T_cur_from_w);
			addNewKeyframe(feature_tree, to_optimizer, &matched_new_feat,linesOnCurrentFrame,
					&point_tree, &point_stats);//to_optimizer is pushed to stack here
		}
	}
	per_mon_->stop("drop keyframe");

	per_mon_->start("dense point cloud");
#ifdef SCAVISLAM_CUDA_SUPPORT
	tracker_.computeDensePointCloudGpu(T_cur_from_actkey_);
#else
	tracker_.computeDensePointCloudCpu(T_cur_from_actkey_);
#endif
	per_mon_->stop("dense point cloud");
	return true;
}

void MonoFrontend::addNewKeyframe(
		const ALIGNED<QuadTree<int> >::vector & feature_tree,
		const AddToOptimzerPtr & to_optimizer,
		tr1::unordered_set<CandidatePoint3Ptr> * matched_new_feat,
		vector<Line> newLinesOnFrame,
		ALIGNED<QuadTree<int> >::vector * point_tree,
		PointStatistics * point_stats) {

	Matrix3i add_flags;
	add_flags.setZero();

	cout << "NEW KF <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;

	static pangolin::Var<int> ui_min_num_points("ui.min_num_points", 25, 20,
			200);

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			if (point_stats->num_points_grid3x3(i, j) <= ui_min_num_points) {
				add_flags(i, j) = 1;
			}
		}
	}

	int oldkey_id = actkey_id;
	actkey_id = getNewUniquePointId();


	//Init the frame vertex vars
	const SE3 & T_oldkey_from_w = GET_MAP_ELEM(oldkey_id,neighborhood_->vertex_map).T_me_from_w;

	tr1::unordered_map<int, int> num_matches;
	FrontendVertex vf;
	vf.T_me_from_w = T_cur_from_actkey_ * T_oldkey_from_w;


	//We don't need to add the newly found lines to the tracked ones as it was done in the previous function

	//Add candidates pts to neighborhood
	for (tr1::unordered_set<CandidatePoint3Ptr>::const_iterator it =matched_new_feat->begin(); it != matched_new_feat->end(); ++it) {
		const CandidatePoint3Ptr & p = *it;
		neighborhood_->point_list.push_back(p); //add candidate points to neighborhood
	}


	//Start covisibility count of the new frame
	//For points
	for (list<NewTwoViewPoint3Ptr>::const_iterator it =	to_optimizer->new_point_list.begin(); it!= to_optimizer->new_point_list.end(); ++it) {
		const NewTwoViewPoint3Ptr & p = *it;
		ADD_TO_MAP_ELEM(p->anchor_id, 1, &num_matches); //Add one to the covis val
		vf.feat_map.insert(make_pair(p->point_id, p->feat_newkey)); //The new points features discovered before are entered in the vertex
	}

	//For lines
	for (tr1::unordered_map<int, Line>::const_iterator it =	to_optimizer->tracked_lines.begin(); it!= to_optimizer->tracked_lines.end(); ++it) {
		Line p = (it->second);
		ADD_TO_MAP_ELEM(p.anchor_id, 1, &num_matches); //Add one to the covis val
		vf.line_map.insert(make_pair(p.global_id, p)); //The new lines features discovered before are entered in the vertex
	}

	//mono_frontend newpoint_map
	//Same type as before, we remove if the point doesnt satisfy some condition
	tr1::unordered_map<int, list<CandidatePoint3Ptr> >::const_iterator it = newpoint_map.begin();
	for (; it != newpoint_map.end(); ++it) {
		newpoint_map[it->first].remove_if(RemoveCondition(*matched_new_feat));
	}

	//Get the vertex of the previous Kf
	const FrontendVertex & oldactive_vertex = GET_MAP_ELEM(oldkey_id,neighborhood_->vertex_map);

	//Get the feature map of the previous Kf
	const ImageFeature<3>::Table & old_feat_map = GET_MAP_ELEM(oldkey_id,neighborhood_->vertex_map).feat_map;

	//Get the line map of the previous Kf
	const ALIGNED<Line>::int_hash_map & old_line_map = GET_MAP_ELEM(oldkey_id,neighborhood_->vertex_map).line_map;


	//For all the currently tracked points
	for (list<TrackPoint3Ptr>::const_iterator itp =	to_optimizer->track_point_list.begin(); itp	!= to_optimizer->track_point_list.end(); ++itp) {

		const TrackPoint3Ptr & p = *itp;

		if (IS_IN_SET(p->global_id, old_feat_map)) { //If a tracked point is in the set of the feature map of the prev Kf
			ADD_TO_MAP_ELEM(oldkey_id, 1, &num_matches); //Add one to prev Kf covis
		}

		//Get the list of the prev Kf neighbors
		for (multimap<int, int>::const_iterator it = oldactive_vertex.strength_to_neighbors.begin(); it != oldactive_vertex.strength_to_neighbors.end(); ++it) {
			int other_pose_id = it->second;//Get the id
			const ImageFeature<3>::Table & other_feat_map = GET_MAP_ELEM(other_pose_id, neighborhood_->vertex_map).feat_map;//Get the table of the neighbors of the prev Kf

			if (IS_IN_SET(p->global_id, other_feat_map)) { //If the tracked point was also in the feat map of the prev Kf neighborhood
				ADD_TO_MAP_ELEM(other_pose_id, 1, &num_matches);//Add one to the covis of this node
			}
		}

		vf.feat_map.insert(make_pair(p->global_id, p->feat)); // Put the tracked points in the feature map
	}

	//For all the currently tracked lines
	for (tr1::unordered_map<int, Line>::const_iterator itp =to_optimizer->tracked_lines.begin(); itp!= to_optimizer->tracked_lines.end(); ++itp) {

		Line  p = itp->second;

		if (IS_IN_SET(p.global_id, old_line_map)) { //If a tracked point is in the set of the feature map of the prev Kf /*HEEEEEEEEEEEEERE !!!!!!!!!!!!!!!!!!*/
			ADD_TO_MAP_ELEM(oldkey_id, 1, &num_matches); //Add one to prev Kf covis
		}

		//Get the list of the prev Kf neighbors
		for (multimap<int, int>::const_iterator it = oldactive_vertex.strength_to_neighbors.begin(); it != oldactive_vertex.strength_to_neighbors.end(); ++it) {
			int other_pose_id = it->second;//Get the id
			const ALIGNED<Line>::int_hash_map & other_line_map = GET_MAP_ELEM(other_pose_id, neighborhood_->vertex_map).line_map;//Get the table of the neighbors of the prev Kf

			if (IS_IN_SET(p.global_id, other_line_map)) { //If the tracked point was also in the feat map of the prev Kf neighborhood
				ADD_TO_MAP_ELEM(other_pose_id, 1, &num_matches);//Add one to the covis of this node
			}
		}
		//vf.line_map.insert(make_pair(p->global_id,Line(p->global_id,p->linearForm,p->descriptor,p->pluckerLinesObservation,p->count,p->startingPoint2d,p->endPoint2d,p->startingPoint3d,p->endPoint3d))); // Put the tracked lines in the feature map
		vf.line_map.insert(make_pair(p.global_id,p)); // Put the tracked lines in the feature map
	}

	//Finally build the covis map of the vertex
	for (tr1::unordered_map<int, int>::const_iterator it = num_matches.begin(); it!= num_matches.end(); ++it) {
		int pose_id = it->first;
		int num_machtes = it->second;

		if (num_machtes > params_.covis_thr) {
			vf.strength_to_neighbors.insert(make_pair(num_machtes, pose_id));
		}
	}

	neighborhood_->vertex_map.insert(make_pair(actkey_id, vf));

	addMorePoints(actkey_id, feature_tree, add_flags, point_tree,&(point_stats->num_matched_points));

	to_optimizer->newkey_id = actkey_id;
	to_optimizer->oldkey_id = oldkey_id;
	to_optimizer->T_newkey_from_oldkey = T_cur_from_actkey_;

	Frame kf = cur_frame_.clone();
	to_optimizer->kf = kf;
	//make sure keyframe is added before pushing to optimiser_stack!!
	keyframe_map.insert(make_pair(actkey_id, kf));
	keyframe_id2num.insert(make_pair(actkey_id, keyframe_id2num.size()));
	keyframe_num2id.push_back(actkey_id);
	to_optimizer_stack.push(to_optimizer);

	T_cur_from_actkey_ = SE3();
}

bool MonoFrontend::shallWeSwitchKeyframe(
		const list<TrackPoint3Ptr> & trackpoint_list,  int * other_id, tr1::unordered_map<int, Line> tracked_lines,
		SE3 * T_cur_from_other,
		ALIGNED<QuadTree<int> >::vector * other_point_tree,
		PointStatistics * other_stat) //other_stat and other_point_tree not used??
{
	static pangolin::Var<float> ui_parallax_thr("ui.parallax_thr", 0.75f, 0, 2);

	double min_dist = 0.5 * ui_parallax_thr;
	int closest = -1;

	const SE3 & T_act_from_w = GET_MAP_ELEM(actkey_id,neighborhood_->vertex_map).T_me_from_w;

	ALIGNED<FrontendVertex>::int_hash_map::const_iterator it = 	neighborhood_->vertex_map.begin();
	for (; it != neighborhood_->vertex_map.end(); ++it) {

		int other_id = it->first;

		if (other_id != actkey_id) {
			const SE3 & T_other_from_w = it->second.T_me_from_w;
			SE3 T_diff = T_cur_from_actkey_ * T_act_from_w	* T_other_from_w.inverse();
			double dist = T_diff.translation().norm();

			if (dist < min_dist) {
				*T_cur_from_other = T_diff;
				min_dist = dist;
				closest = other_id;
			}
		}
	}

	if (closest != -1) {
		ImageFeature<3>::Table feat_table = GET_MAP_ELEM(closest,neighborhood_->vertex_map).feat_map;
		ALIGNED<Line>::int_hash_map line_map = GET_MAP_ELEM(closest,neighborhood_->vertex_map).line_map;

		int count = 0;

		for (list<TrackPoint3Ptr>::const_iterator it = trackpoint_list.begin(); it!= trackpoint_list.end(); ++it) {
			const TrackPoint3Ptr & p = *it;
			if (IS_IN_SET(p->global_id, feat_table)) { //if many points in common, change kf
				++count;
			}
		}

		// tr1::unordered_map<int, Line> tracked_lines;
		for ( tr1::unordered_map<int, Line>::const_iterator it = tracked_lines.begin(); it!= tracked_lines.end(); ++it) {
					Line p = (*it).second;
					if (IS_IN_SET(p.global_id, line_map)) {
						++count;
					}
				}

		//TODO : place here the same structure with the check of lines

		//TODO: add more sophisticated check
		if (count > 100) {
			*other_id = closest;
			return true;
		}
	}
	return false;
}

bool MonoFrontend::shallWeDropNewKeyframe(const PointStatistics & point_stats) {
	int num_featuerless_corners = 0;

	//TODO : ADD THE LINE INFO

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			if (point_stats.num_points_grid2x2(i, j) < 15)
				++num_featuerless_corners;

	static pangolin::Var<float> ui_parallax_thr("ui.parallax_thr", 0.75f, 0, 2);

	return num_featuerless_corners
			> params_.new_keyframe_featuerless_corners_thr
			|| T_cur_from_actkey_.translation().norm() > ui_parallax_thr
			|| av_track_length_ > 75.;
}

#ifdef SCAVISLAM_CUDA_SUPPORT
//TODO: Method too long
void MonoFrontend::
calcDisparityGpu()
{
	static pangolin::Var<int> stereo_method("ui.stereo_method",2,1,4);
	static pangolin::Var<int> num_disp16("ui.num_disp16",2,1,10);
	int num_disparities = num_disp16*16;

	if (stereo_method==1)
	{
		cv::StereoBM stereo_bm;
		stereo_bm.state->preFilterCap = 31;
		stereo_bm.state->SADWindowSize = 7;
		stereo_bm.state->minDisparity = 0;

		stereo_bm.state->textureThreshold = 10;
		stereo_bm.state->uniquenessRatio = 15;
		stereo_bm.state->speckleWindowSize = 100;
		stereo_bm.state->speckleRange = 32;
		stereo_bm.state->disp12MaxDiff = 1;

		stereo_bm.state->numberOfDisparities = num_disparities;

		stereo_bm(frame_data_->cur_left().pyr_uint8[0],
				frame_data_->right.uint8,
				frame_data_->disp,
				CV_32F);

		frame_data_->gpu_disp_32f.upload(frame_data_->disp);
		frame_data_->gpu_disp_32f.convertTo(frame_data_->gpu_disp_16s,CV_16S, 1.);
		cv::gpu::drawColorDisp(frame_data_->gpu_disp_16s,
				frame_data_->gpu_color_disp,num_disparities);
		frame_data_->gpu_color_disp.download(frame_data_->color_disp);

	}
	else if (stereo_method==2)
	{
		cv::gpu::StereoBM_GPU gpu_stereo_bm(cv::gpu::StereoBM_GPU::PREFILTER_XSOBEL,
				num_disparities);

		gpu_stereo_bm(frame_data_->cur_left().gpu_uint8,
				frame_data_->right.gpu_uint8,
				frame_data_->gpu_disp_16s);
		cv::gpu::drawColorDisp(frame_data_->gpu_disp_16s,
				frame_data_->gpu_color_disp,
				num_disparities);
		frame_data_->gpu_disp_16s.convertTo(frame_data_->gpu_disp_32f,CV_32F,1.);
		frame_data_->gpu_disp_32f.download(frame_data_->disp);
		frame_data_->gpu_color_disp.download(frame_data_->color_disp);
	}
	else if (stereo_method==3)
	{
		cv::gpu::StereoBeliefPropagation gpu_stereo_bm(num_disparities);
		gpu_stereo_bm(frame_data_->cur_left().gpu_uint8,
				frame_data_->right.gpu_uint8,
				frame_data_->gpu_disp_16s);
		cv::gpu::drawColorDisp(frame_data_->gpu_disp_16s,
				frame_data_->gpu_color_disp,
				num_disparities);
		frame_data_->gpu_disp_16s.convertTo(frame_data_->gpu_disp_32f,CV_32F,1.);
		frame_data_->gpu_disp_32f.download(frame_data_->disp);
		frame_data_->gpu_color_disp.download(frame_data_->color_disp);
	}
	else if (stereo_method==4)
	{
		static pangolin::Var<int> stereo_iters("ui.stereo_iters",4,1,20);
		static pangolin::Var<int> stereo_levels("ui.stereo_levels",4,1,5);
		static pangolin::Var<int> stereo_nr_plane("ui.stereo_nr_plane",1,1,10);
		cv::gpu::StereoConstantSpaceBP gpu_stereo_bm(num_disparities,
				stereo_iters,
				stereo_levels,
				stereo_nr_plane);
		gpu_stereo_bm(frame_data_->cur_left().gpu_uint8,
				frame_data_->right.gpu_uint8,
				frame_data_->gpu_disp_16s);

		cv::gpu::drawColorDisp(frame_data_->gpu_disp_16s,
				frame_data_->gpu_color_disp,
				num_disparities);
		frame_data_->gpu_disp_16s.convertTo(frame_data_->gpu_disp_32f,CV_32F,1.);
		frame_data_->gpu_disp_32f.download(frame_data_->disp);
		frame_data_->gpu_color_disp.download(frame_data_->color_disp);
	}
}

#else

void MonoFrontend::calcDisparityCpu() {
	static pangolin::Var<int> num_disp16("ui.num_disp16", 2, 1, 10);
	int num_disparities = num_disp16 * 16;
	cv::StereoBM stereo_bm;
	stereo_bm.state->preFilterCap = 31;
	stereo_bm.state->SADWindowSize = 7;
	stereo_bm.state->minDisparity = 0;

	stereo_bm.state->textureThreshold = 10;
	stereo_bm.state->uniquenessRatio = 15;
	stereo_bm.state->speckleWindowSize = 100;
	stereo_bm.state->speckleRange = 32;
	stereo_bm.state->disp12MaxDiff = 1;

	stereo_bm.state->numberOfDisparities = num_disparities;

	stereo_bm(frame_data_->cur_left().pyr_uint8[0], frame_data_->right.uint8,
			frame_data_->disp, CV_32F);

	vector<cv::Mat> hsv_array(3);
	hsv_array[0] = cv::Mat(frame_data_->disp.size(), CV_8UC1);

	frame_data_->disp.convertTo(hsv_array[0], CV_8UC1, 5., 0.);
	hsv_array[1] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);
	hsv_array[2] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);

	cv::Mat hsv(frame_data_->disp.size(), CV_8UC3);
	cv::merge(hsv_array, hsv);
	cv::cvtColor(hsv, frame_data_->color_disp, CV_HSV2BGR);
}

#endif

void MonoFrontend::computeFastCorners(int trials,ALIGNED<QuadTree<int> >::vector * feature_tree,vector<CellGrid2d> * cell_grid_2d) {
	bool debug=true;


	const ALIGNED<StereoCamera>::vector & cam_vec = frame_data_->cam_vec;
	feature_tree->resize(USE_N_LEVELS_FOR_MATCHING);
	cell_grid_2d->resize(USE_N_LEVELS_FOR_MATCHING);




	for (int level = 0; level < USE_N_LEVELS_FOR_MATCHING; ++level) {
		feature_tree->at(level) = QuadTree<int> (Rectangle(0, 0, cam_vec.at(level).width(),	cam_vec.at(level).height()), 1);

		fast_grid_.at(level).detectAdaptively(frame_data_->cur_left().pyr_uint8.at(level), trials,&(feature_tree->at(level)));

		cell_grid_2d->at(level) = fast_grid_.at(level).cell_grid2d();


//
//		if(debug==true){
//			vector<cv::Point> pVec;
//			for(auto iter=feature_tree->at(level).begin_bfs(); iter.reached_end()==false;++iter){
//				cv::Point p = cv::Point((*iter).pos[0],(*iter).pos[1]);
//				pVec.push_back(p);
//			}
//			showPoints(frame_data_->cur_left().pyr_uint8.at(level),pVec);
//		}
	}


}

void MonoFrontend::addNewPoints(int new_keyframe_id,
		const ALIGNED<QuadTree<int> >::vector & feature_tree) {
	vector<int> num_points(USE_N_LEVELS_FOR_MATCHING, 0);
	const ALIGNED<StereoCamera>::vector & cam_vec = frame_data_->cam_vec;

	ALIGNED<QuadTree<int> >::vector point_tree(USE_N_LEVELS_FOR_MATCHING);

	for (int l = 0; l < USE_N_LEVELS_FOR_MATCHING; ++l) {
		point_tree.at(l) = QuadTree<int> (
				Rectangle(0, 0, cam_vec[l].width(), cam_vec[l].height()), 1);
	}

	Matrix3i add_flags;
	add_flags.setOnes();
	return addMorePoints(new_keyframe_id, feature_tree, add_flags, &point_tree,	&num_points);
}

void MonoFrontend::addMorePoints(int new_keyframe_id,const ALIGNED<QuadTree<int> >::vector & feature_tree,	const Matrix3i & add_flags,	ALIGNED<QuadTree<int> >::vector * point_tree, vector<int> * num_points) {
	addMorePointsToOtherFrame(		new_keyframe_id, SE3(), feature_tree, add_flags,frame_data_->disp, point_tree, num_points);
}

//TODO: method too long
void MonoFrontend::addMorePointsToOtherFrame(int new_keyframe_id,
		const SE3 & T_newkey_from_cur,
		const ALIGNED<QuadTree<int> >::vector & feature_tree,
		const Matrix3i & add_flags, const cv::Mat & disp_img,
		ALIGNED<QuadTree<int> >::vector * point_tree, vector<int> * num_points) {

	const StereoCamera & cam = frame_data_->cam;

	int CLEARANCE_RADIUS = params_.newpoint_clearance;
	int CLEARANCE_DIAMETER = CLEARANCE_RADIUS * 2 + 1;

	float third = 1. / 3.;
	int third_width = cam.width() * third;
	int third_height = cam.height() * third;
	int twothird_width = cam.width() * 2 * third;
	int twothird_height = cam.height() * 2 * third;
	pangolin::Var<int> var_num_max_points("ui.num_max_points", 300, 50, 1000);

	for (int level = 0; level < USE_N_LEVELS_FOR_MATCHING; ++level) {
		DrawItems::Point2dVec new_point_vec;

		int num_max_points = pyrFromZero_i(var_num_max_points, level);
		for (QuadTree<int>::EquiIter iter = feature_tree.at(level).begin_equi(); !iter.reached_end(); ++iter) {
			Vector2d uv_pyr = iter->pos;
			Vector2i uv_pyri = uv_pyr.cast<int> ();

			double disp = interpolateDisparity(disp_img, uv_pyri, level);
			if (disp > 0) {
				Vector2i uvi = zeroFromPyr_2i(uv_pyri, level);

				if (!cam.isInFrame(uvi, 1))
					continue;

				int i = 2;
				int j = 2;
				if (uvi[0] < third_width)
					i = 0;
				else if (uvi[0] < twothird_width)
					i = 1;
				if (uvi[1] < third_height)
					j = 0;
				else if (uvi[1] < twothird_height)
					j = 1;

				if (add_flags(i, j) == 0)
					continue;

				Rectangle win(uv_pyr[0] - CLEARANCE_RADIUS,
						uv_pyr[1] - CLEARANCE_RADIUS, CLEARANCE_DIAMETER,
						CLEARANCE_DIAMETER);

				if ((*point_tree)[level].isWindowEmpty(win)) {
					new_point_vec.push_back(GlPoint2f(uv_pyr[0], uv_pyr[1]));

					Vector3d uvu_pyr = Vector3d(uv_pyr[0], uv_pyr[1],
							uv_pyr[0] - disp);

					Vector3d uvu_0 = zeroFromPyr_3d(uvu_pyr, level);

//					Vector3d xyz_cur2;
//					Vector3d xyz_cur;
//					if(params_.livestream)
//					{
//						if(request3DCoords(uvu_0[0],uvu_0[1], &xyz_cur2))
//						{
//							xyz_cur = xyz_cur2;
//						}
//						else
//						{
//							cout<<"failed getting coordinates"<<endl;
//							continue;
//						}
//					}
//					else
//					{
//						xyz_cur = cam.unmap_uvu(uvu_0);
//					}
//					Vector3d tmp= cam.unmap_uvu(uvu_0);
//					cout<<"xyz_cur kinect: "<<xyz_cur<<endl;
//					cout<<"xyz_cur disp: "<<tmp<<endl;
//					double unscaled_d = sqrt( pow((tmp[0]-xyz_cur[0]), 2) + pow((tmp[1]-xyz_cur[1]),2) + pow((tmp[2]-xyz_cur[2]),2));
//					cout<<"distance: "<<unscaled_d<<endl;
					Vector3d xyz_cur = cam.unmap_uvu(uvu_0);
					(*point_tree)[level].insert(uv_pyr, (*num_points)[level]);
					draw_data_.new_points2d.at(level).push_back(
							GlPoint2f(uv_pyr));
					draw_data_.new_points3d.at(level).push_back(
							GlPoint3f(xyz_cur));

					double dist = xyz_cur.norm();
					Vector3d normal = -xyz_cur / dist;

					int new_point_id = getNewUniquePointId();

					newpoint_map[new_keyframe_id].push_front(
							CandidatePoint3Ptr(
									new CandidatePoint<3> (new_point_id,
											T_newkey_from_cur * xyz_cur,
											new_keyframe_id, uvu_pyr, level,
											normal)));
					++(*num_points)[level];

					if ((*num_points)[level] > num_max_points)
						break;
				}
			}
		}
	}
}

int MonoFrontend::getNewUniquePointId() {
	++unique_point_id_counter_;
	return unique_point_id_counter_;
}



//TODO: method too long
AddToOptimzerPtr MonoFrontend::processMatchedPoints(
		const TrackData<3> & track_data, int num_new_feat_matched,
		ALIGNED<QuadTree<int> >::vector * point_tree,
		tr1::unordered_set<CandidatePoint3Ptr> * matched_new_feat,
		PointStatistics * stats) {
	AddToOptimzerPtr to_optimizer(new AddToOptimzer);
	const StereoCamera & cam = frame_data_->cam;
	SE3XYZ se3xyz(cam);

	static pangolin::Var<float>
			max_reproj_error("ui.max_reproj_error", 2, 0, 5);

	int half_width = cam.width() * 0.5;
	int half_height = cam.height() * 0.5;
	float third = 1. / 3.;
	int third_width = cam.width() * third;
	int third_height = cam.height() * third;
	int twothird_width = cam.width() * 2 * third;
	int twothird_height = cam.height() * 2 * third;

	int num_track_points = 0;
	double sum_track_length = 0.f;

	for (list<IdObs<3> >::const_iterator it = track_data.obs_list.begin(); it
			!= track_data.obs_list.end(); ++it) {
		IdObs<3> id_obs = *it;
		const Vector3d & point = track_data.point_list.at(id_obs.point_id);
		const Vector3d & uvu_pred = se3xyz_stereo_.map(T_cur_from_actkey_,
				point);
		const Vector3d & uvu = id_obs.obs;
		Vector3d diff = uvu - uvu_pred;
		const CandidatePoint3Ptr & ap = track_data.ba2globalptr.at(
				id_obs.point_id);
		int factor = zeroFromPyr_i(1, ap->anchor_level);

		if (abs(diff[0]) < max_reproj_error * factor && abs(diff[1])
				< max_reproj_error * factor && abs(diff[2]) < 3.
				* max_reproj_error) {

			int i = 1;
			int j = 1;
			if (uvu[0] < half_width)
				i = 0;
			if (uvu[1] < half_height)
				j = 0;
			++(stats->num_points_grid2x2(i, j));

			i = 2;
			j = 2;
			if (uvu[0] < third_width)
				i = 0;
			else if (uvu[0] < twothird_width)
				i = 1;
			if (uvu[1] < third_height)
				j = 0;
			else if (uvu[1] < twothird_height)
				j = 1;
			++(stats->num_points_grid3x3(i, j));

			++(stats->num_matched_points)[ap->anchor_level];
			Vector2d curkey_uv_pyr = pyrFromZero_2d(se3xyz.map(SE3(), point),
					ap->anchor_level);
			Vector2d uv_pyr = pyrFromZero_2d(Vector2d(uvu.head(2)),
					ap->anchor_level);

			ALIGNED<DrawItems::Point2dVec>::int_hash_map & keymap =
					draw_data_.tracked_anchorpoints2d.at(ap->anchor_level);
			ALIGNED<DrawItems::Point2dVec>::int_hash_map::iterator keymap_it =
					keymap.find(ap->anchor_id);

			if (keymap_it != keymap.end()) {
				keymap_it->second.push_back(GlPoint2f(curkey_uv_pyr));
			} else {
				DrawItems::Point2dVec point_vec;
				point_vec.push_back(GlPoint2f(curkey_uv_pyr));
				keymap.insert(make_pair(ap->anchor_id, point_vec));
			}

			(*point_tree).at(ap->anchor_level).insert(uv_pyr, ap->point_id);

			if (id_obs.point_id < num_new_feat_matched) {
				(*matched_new_feat).insert(ap);
				draw_data_.newtracked_points2d.at(ap->anchor_level) .push_back(
						DrawItems::Line2d(uv_pyr, curkey_uv_pyr));

				sum_track_length += (uv_pyr - curkey_uv_pyr).norm();
				++num_track_points;

				SE3 T_w_from_anchor = GET_MAP_ELEM(ap->anchor_id,
						neighborhood_->vertex_map) .T_me_from_w.inverse();
				draw_data_.newtracked_points3d.at(ap->anchor_level) .push_back(
						GlPoint3f(T_w_from_anchor * ap->xyz_anchor));

				ImageFeature<3> feat(uvu, ap->anchor_level);
				NewTwoViewPoint3Ptr np(
						new NewTwoViewPoint<3> (ap->point_id, ap->anchor_id,
								ap->xyz_anchor, ap->anchor_obs_pyr,
								ap->anchor_level, ap->normal_anchor, feat)); //create new point to be tracked
				to_optimizer->new_point_list.push_back(np);
			} else {
				ImageFeature<3> feat(id_obs.obs, ap->anchor_level);
				to_optimizer->track_point_list.push_back(
						TrackPoint3Ptr(new TrackPoint<3> (ap->point_id, feat)));
				draw_data_.tracked_points2d.at(ap->anchor_level) .push_back(
						DrawItems::Line2d(uv_pyr, curkey_uv_pyr));

				sum_track_length += (uv_pyr - curkey_uv_pyr).norm();
				++num_track_points;
				SE3 T_w_from_anchor = GET_MAP_ELEM(ap->anchor_id,
						neighborhood_->vertex_map) .T_me_from_w.inverse();
				draw_data_.tracked_points3d.at(ap->anchor_level) .push_back(
						GlPoint3f(T_w_from_anchor * ap->xyz_anchor));
			}
		}
	}
	av_track_length_ = sum_track_length / num_track_points;

	// cerr << av_track_length_ << endl;

	return to_optimizer;
}

bool MonoFrontend::matchAndTrack(
		const ALIGNED<QuadTree<int> >::vector & feature_tree, //input is feature_tree
		TrackData<3> * track_data, int * num_new_feat_matched, SE3 &T_anchorkey_from_w) {

	BA_SE3_XYZ_STEREO ba;
	*num_new_feat_matched = 0;

	const FrontendVertex & active_vertex = GET_MAP_ELEM(actkey_id,neighborhood_->vertex_map);

	GuidedMatcher<StereoCamera>::match(keyframe_map, T_cur_from_actkey_,cur_frame_, feature_tree, frame_data_->cam_vec, actkey_id,neighborhood_->vertex_map, newpoint_map[actkey_id],
#ifdef SCAVISLAM_CUDA_SUPPORT
			4,
#else
			8,
#endif
			22, 10, track_data, T_anchorkey_from_w);

	pangolin::Var<int> var_num_max_points("ui.num_max_points", 300, 50, 1000);
	for (multimap<int, int>::const_iterator it =active_vertex.strength_to_neighbors.begin(); it != active_vertex.strength_to_neighbors.end() && (int) (2* track_data->obs_list.size()) < var_num_max_points; ++it)
	{
		GuidedMatcher<StereoCamera>::match(keyframe_map, T_cur_from_actkey_,cur_frame_, feature_tree, frame_data_->cam_vec, actkey_id,neighborhood_->vertex_map, newpoint_map[it->second],
#ifdef SCAVISLAM_CUDA_SUPPORT
				4,
#else
				8,
#endif
				22, 10, track_data);

	}
	*num_new_feat_matched += track_data->obs_list.size();

	GuidedMatcher<StereoCamera>::match(keyframe_map, T_cur_from_actkey_,cur_frame_, feature_tree, frame_data_->cam_vec, actkey_id,neighborhood_->vertex_map, neighborhood_->point_list,
#ifdef SCAVISLAM_CUDA_SUPPORT
			4,
#else
			8,
#endif
			22, 10, track_data);

	cout << "tracked_lines = " << tracked_lines.size() << endl;

	if (track_data->obs_list.size() + tracked_lines.size() < 20) {
		cout << "Tracked lines : " << tracked_lines.size() << ", Pt List : " << track_data->obs_list.size() << endl;
		return false;
	}

	OptimizerStatistics opt = ba.calcFastMotionOnly(track_data->obs_list,se3xyz_stereo_, PoseOptimizerParams(true, 2, 15),&T_cur_from_actkey_, &track_data->point_list);
	return true;
}

}
