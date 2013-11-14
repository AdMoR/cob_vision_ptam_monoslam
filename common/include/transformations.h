// This file is part of ScaViSLAM.
//
// Copyright 2011 Hauke Strasdat (Imperial College London)
//
// ScaViSLAM is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// any later version.
//
// ScaViSLAM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with ScaViSLAM.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SCAVISLAM_TRANSFORMATIONS_H
#define SCAVISLAM_TRANSFORMATIONS_H

#include <list>

#include <sophus/se3.h>
#ifdef MONO
#include <sophus/sim3.h>
#endif

#include <visiontools/linear_camera.h>

#include "maths_utils.h"
#include "stereo_camera.h"

namespace ScaViSLAM
{

using namespace Eigen;
using namespace Sophus;
using namespace VisionTools;

//TODO: clean, hide implementation and remove stuff not needed here

struct AnchoredPoint3d
{
  AnchoredPoint3d(const Vector3d & p_a, int frame_id)
    : p_a(p_a), frame_id(frame_id)
  {
  }
  Vector3d p_a;
  int frame_id;
};

inline Matrix<double,4,4>
toPlueckerMatrix( const Vector6d & vec)
{
	Matrix<double,4,4> plueckerMatrix;
	plueckerMatrix(0,0) = 0;
	plueckerMatrix(0,1) = vec[0];
	plueckerMatrix(0,2) = vec[1];
	plueckerMatrix(0,3) = vec[2];
	plueckerMatrix(1,0) = -1*vec[0];
	plueckerMatrix(1,1) = 0;
	plueckerMatrix(1,2) = vec[3];
	plueckerMatrix(1,3) = -1*vec[4];
	plueckerMatrix(2,0) = -1*vec[1];
	plueckerMatrix(2,1) = -1*vec[3];
	plueckerMatrix(2,2) = 0;
	plueckerMatrix(2,3) = vec[5];
	plueckerMatrix(3,0) = -1*vec[2];
	plueckerMatrix(3,1) =  vec[4];
	plueckerMatrix(3,2) = -1*vec[5];
	plueckerMatrix(3,3) = 0;
	return plueckerMatrix;
}

inline Vector6d
computePlueckerLineParameters(Vector3d a, Vector3d b)
{
	//L={l12,l13,l14,l23,l42,l34}
    //if the plücker coordinates correspond a real line on 3D -> l12*l34 + l13*l42 + l14*l23=0
	//computation using only two points

	Vector6d plueckerParam;

	plueckerParam[0] =  a[0] * b[1] -  b[0] * a[1];
	plueckerParam[1] = a[0] * b[2] -  b[0] * a[2];
    plueckerParam[2] = a[0] -  b[0];
    plueckerParam[3] = a[1] * b[2] -  b[1] * a[2];
    plueckerParam[4] = -1*(a[1]  -  b[1]); //to get l42 instead of l24
    plueckerParam[5] = a[2]  -  b[2]; //what if it is zero? use L2 Norm

    plueckerParam.normalize();

    int constraint = static_cast<int> (plueckerParam[0]*plueckerParam[5]+plueckerParam[1]*plueckerParam[4]+plueckerParam[2]*plueckerParam[3]);
    if(constraint!=0)
    {
    	cout<<"a: "<<a<<endl;
    	cout<<"b: "<<b<<endl;
    	cout<<"constraint!=0"<<endl;
    	cout<<plueckerParam<<endl;
    }
    assert(constraint==0);

    return plueckerParam;
}

inline Vector3d
calculateLinearForm(int x1, int y1, int x2, int y2)
{
	//Ax+By+c=0
	//For points P and Q
	//(py – qy)x + (qx – px)y + (pxqy – qxpy) = 0
	Vector3d linearForm;
	linearForm[0] = y1 - y2;
	linearForm[1] = x2 - x1;
	linearForm[2] = x1*y2 - x2*y1;
	return linearForm;
}

inline Vector6d
toPlueckerVec( const Matrix<double,4,4> & plueckerMatrix)
{
	Vector6d plueckerVector;
	plueckerVector[0] = plueckerMatrix(0,1);
	plueckerVector[1] = plueckerMatrix(0,2);
	plueckerVector[2] = plueckerMatrix(0,3);
	plueckerVector[3] = plueckerMatrix(1,2);
	plueckerVector[4] = plueckerMatrix(3,1);
	plueckerVector[5] = plueckerMatrix(2,3);
	return plueckerVector;
}

inline Matrix<double,3,4>
computeProjectionMatrix(const Matrix<double,3,3> camera_matrix, const Matrix<double,4,4> T_cur_from_w)
{
	Matrix<double, 3, 4> I;
	I<<1,0,0,0,
	   0,1,0,0,
	   0,0,1,0;
	Matrix<double,3,4> projectionsMatrix = camera_matrix*I*T_cur_from_w;
//	cout<<"cameraMatrix: "<<camera_matrix<<endl;
//	cout<<"projectionMatrix: "<<projectionsMatrix<<endl;
	return projectionsMatrix;
}

inline  Vector3d
computeLineProjectionMatrix2(Matrix<double,3,4> projectionsMatrix, Matrix<double,4,4> pluckerMatrix)
{
	//l× = PLPT
	//l = [lx(3,2); lx(1,3); lx(2,1)];
	Matrix<double,3,3> result = projectionsMatrix*pluckerMatrix*projectionsMatrix.transpose();
	Vector3d projectedHomogeneousLine(result(2,1), result(0,2), result(1,0));
	return projectedHomogeneousLine;


}

inline Vector4d
toHomogeneousCoordinates(Vector3d vec)
{
	Vector4d homoVec(vec[0], vec[1], vec[2], 1);
	return homoVec;
}

inline  Matrix <double, 3, 6>
computeLineProjectionMatrix(Matrix<double,3,4> projectionsMatrix)
{
	//Line Projection Matrix
	//P = (p2 ∧ p3)
	//	  (p3 ∧ p1)
	//    (p1 ∧ p2)

	//Plücker Dual Coordinates
	//L* = P(Q)T - Q(P)T

	//(p2 ∧ p3)
	Matrix<double, 4, 4> p1 = projectionsMatrix.row(1).transpose() * projectionsMatrix.row(2) - projectionsMatrix.row(2).transpose() * projectionsMatrix.row(1);
	//(p3 ∧ p1)
	Matrix<double, 4, 4> p2 = projectionsMatrix.row(2).transpose() * projectionsMatrix.row(0) - projectionsMatrix.row(0).transpose() * projectionsMatrix.row(2);
	//(p1 ∧ p2)
	Matrix<double, 4, 4> p3 = projectionsMatrix.row(0).transpose() * projectionsMatrix.row(1) - projectionsMatrix.row(1).transpose() * projectionsMatrix.row(0);


//	cout<<p1<<endl;
//	cout<<p2<<endl;
//	cout<<p3<<endl;

	//rewrite rule for dual plücker coordinates
	//l12 : l13 : l14 : l23 : l42 : l34 = l*34 : l*42 : l*23 : l*14 : l*13 : l*12 .
	//apparently in this case it is not necessary to rewrite it

	Matrix <double, 3, 6> result;
	Vector6d plueckerVector1;
	plueckerVector1<<p1(0,1), p1(0,2), p1(0,3), p1(1,2), p1(3,1), p1(2,3);
	//plueckerVector1.normalize();
	Vector6d plueckerVector2;
	plueckerVector2<<p2(0,1), p2(0,2), p2(0,3), p2(1,2), p2(3,1), p2(2,3);
	//plueckerVector2.normalize();
	Vector6d plueckerVector3;
	plueckerVector3<<p3(0,1), p3(0,2), p3(0,3), p3(1,2), p3(3,1), p3(2,3);
	//plueckerVector3.normalize();
	result <<plueckerVector1[0], plueckerVector1[1], plueckerVector1[2], plueckerVector1[3], plueckerVector1[4], plueckerVector1[5],
			plueckerVector2[0], plueckerVector2[1], plueckerVector2[2], plueckerVector2[3], plueckerVector2[4], plueckerVector2[5],
			plueckerVector3[0], plueckerVector3[1], plueckerVector3[2], plueckerVector3[3], plueckerVector3[4], plueckerVector3[5];

//		tmp<< p1(2,3), p1(3,1), p1(1,2), p1(0,3), p1(0,2), p1(0,1),
//				p2(2,3), p2(3,1), p2(1,2), p2(0,3), p2(0,2), p2(0,1),
//				p3(2,3), p3(3,1), p3(1,2), p3(0,3), p3(0,2), p3(0,1);
//		cout<<"tmp: "<<tmp<<endl;
//	cout<<"line Projection Matrix: "<<result<<endl;

	return result;
}
inline
void changeSigns(Vector3d &vec)
{
	vec[0] = -1 * vec[0];
	vec[1] = -1 * vec[1];
	vec[2] = -1 * vec[2];
}

template <class NumTy>
inline bool matchingSigns(NumTy a, NumTy b) {
    return a < (NumTy)0 == b < (NumTy)0;
}

inline Vector2d
computeIntersectionOfTwoLines(Vector3d l1, Vector3d l2)
{
	//a1*x+b1*y+c1=0, a2*x+b2*y+c2=0

	//x=(b2c1-b1c2)/(a2b1-a1b2) y = (a1 c2-a2 c1)/(a2 b1-a1 b2)
	Vector2d intersection;
	intersection[0] = (l2[1]*l1[2]-l1[1]*l2[2])/(l2[0]*l1[1]-l1[0]*l2[1]);
	intersection[1] = (l1[0]*l2[2]-l2[0]*l1[2])/(l2[0]*l1[1]-l1[0]*l2[1]);
	return intersection;
}
//angles are returned in degrees
inline void cartesianToPolar(Vector3d line, double &mag, double &angle)
{
	/*
	 * ax+by+c=0
	 * y=-(a/b)x-c/b
	 */

	double a = line[0];
	double b = line[1];
	double c = line[2];

	//if horizontal line
	if (a == (double) 0.0)
	{
		mag = -(c / b);
		//if in the last two quadrants
		if (mag < (double) 0.0)
		{
			mag=-mag;
		}
		angle = 90.0;
	}
	//if vertical line
	else if (b == (double) 0.0)
	{
		mag = -(c / a);
		angle = 0.0;

	}
	else if(c == (double) 0.0)
	{
		mag=0.0;
		angle=90+atan2(-a,b);
	}
	else
	{
		/*the perpendicular line that passes through (0,0) has -1/m slope => b/a
		 * y-y0 = -1/m(x-x0)
		 * y-0= b/a (x-0)
		 * y=(b/a)x
		 * the intersection of this perpendicular line with the original line is found
		 * by solving (b/a)x=-(a/b)x-c/b => x= -ca/(b*b+a*a)  and y= -bc/(b*b+a*a)
		 */
		double intersec_x =  -c*a/(b*b+a*a);
		double intersec_y = -b*c/(b*b+a*a);
		mag=sqrt(pow(intersec_x,2)+pow(intersec_y,2));
		angle=atan2(intersec_y,intersec_x)* (180 / CV_PI);
	}
	//if the intersection was in quadrants three or four, change signs and angle
	if(angle<(double) 0.0)
	{
		mag=-mag;
		angle=angle+180.0;
	}
	assert(angle>=(double)0.0 && angle<=(double)180.0);
	cout<<"radius: "<<mag<<" angle: "<<angle<<endl;
}

inline double
computePolarDistance(const double projectedLineMag,const double projectedLineAng, const double currentLineMag, const double currentLineAng, const double weight)
{
	double distance = sqrt((projectedLineMag-currentLineMag)*(projectedLineMag-currentLineMag) + weight*((projectedLineAng-currentLineAng)*(projectedLineAng-currentLineAng)));
	//cout<<"distance: "<<distance<<" diff mag: "<<projectedLineMag-currentLineMag<<" diff angle: "<<projectedLineAng-currentLineAng<<endl;
	return distance;
}

inline bool
isThereAnIntersectionBetweenLines(Vector3d l1, Vector3d l2)
{
	//lines intersect each other if a1b2 ≠ a2b1
	/// if they are equal, they are parallel lines
	//cout<<"var1: "<<l1[0]*l2[1]<<" var2: "<<l2[0]*l1[1]<<endl;
	return (l1[0]*l2[1]!=l2[0]*l1[1]);
}
inline vector<Vector3d>
calculateEdges()
{
	Vector2d topLeft(0,0);
	Vector2d topRight(640,0);
	Vector2d bottomRight(640,480);
	Vector2d bottomLeft(0,480);

	Vector3d topLine = calculateLinearForm(topLeft[0], topLeft[1], topRight[0], topRight[1]);
	Vector3d rightEdge = calculateLinearForm(topRight[0], topRight[1], bottomRight[0], bottomRight[1]);
	Vector3d bottomEdge = calculateLinearForm(bottomLeft[0],bottomLeft[1], bottomRight[0], bottomRight[1]);
	Vector3d leftEdge = calculateLinearForm(topLeft[0], topLeft[1],bottomLeft[0],bottomLeft[1]);

	vector<Vector3d> edges {topLine, rightEdge, bottomEdge, leftEdge};
	return edges;
}


inline bool
lineIsInsideFrame(const Vector3d line,const vector<Vector3d> & edges, Vector2d &intersec1, Vector2d &intersec2, bool debug)
{
//line is in Ax+By+c=0 form
//we can consider the edges of the 640x480 image also as lines, compute the intersection for each edge and see if the point lies inside the image
//if the line is inside frame, we also compute the two intersection points of the projection with the frame edges


	vector<string> names {"topLine", "rightEdge", "bottomEdge", "leftEdge"};
	bool first=true;
	bool lineInsideFrame=false;
	if(debug) cout<<"--------------------------------------------------------------"<<endl;
	int i=0;
	for (vector<Vector3d>::const_iterator it = edges.begin(); it != edges.end(); it++)
	{
		if (isThereAnIntersectionBetweenLines(line, *it))
		{


			Vector2d intersect = computeIntersectionOfTwoLines(line, *it);
			if(debug) cout<<"intersect: "<<intersect<<endl;
			if (((int)intersect[0] <= 640 && (int)intersect[1] <= 480) && ((int)intersect[0] >= 0 && (int)intersect[1] >= 0))
			{
				if(debug)cout << "intersect in frame with " << names.at(i) << "at: "<< intersect<< endl;
				if(first)
				{
					intersec1=intersect;
					first=false;
				}
				else //todo:check if it sometimes returns only one intersection
				{
					intersec2=intersect;
					lineInsideFrame = true;
				}
			}
			else
			{
				if(debug) cout << "doesn't intersect in frame with " << names.at(i) << endl;
			}
		}
		else
		{
			if(debug)	cout << "doesn't intersect with " << names.at(i) << endl;
		//	return false;
		}
	i++;
	}
	return lineInsideFrame;
}

inline bool
equalVectors3d(Vector3d a, Vector3d b)
{
	return ((a[0]==b[0] && a[1]==b[1]) && a[2]==b[2]);
}

inline Matrix<double,2,3>
d_proj_d_y(const double & f, const Vector3d & xyz)
{
  double z_sq = xyz[2]*xyz[2];
  Matrix<double,2,3> J;
  J << f/xyz[2], 0,           -(f*xyz[0])/z_sq,
      0,           f/xyz[2], -(f*xyz[1])/z_sq;
  return J;
}

inline Matrix3d
d_stereoproj_d_y(const double & f, double b, const Vector3d & xyz)
{
  double z_sq = xyz[2]*xyz[2];
  Matrix3d J;
  J << f/xyz[2], 0,           -(f*xyz[0])/z_sq,
      0,           f/xyz[2], -(f*xyz[1])/z_sq,
      f/xyz[2], 0,           -(f*(xyz[0]-b))/z_sq;
  return J;
}

inline Matrix<double,3,6>
d_expy_d_y(const Vector3d & y)
{
  Matrix<double,3,6> J;
  J.topLeftCorner<3,3>().setIdentity();
  J.bottomRightCorner<3,3>() = -SO3::hat(y);
  return J;
}

inline Matrix3d
d_Tinvpsi_d_psi(const SE3 & T, const Vector3d & psi)
{
  Matrix3d R = T.rotation_matrix();
  Vector3d x = invert_depth(psi);
  Vector3d r1 = R.col(0);
  Vector3d r2 = R.col(1);
  Matrix3d J;
  J.col(0) = r1;
  J.col(1) = r2;
  J.col(2) = -R*x;
  J*=1./psi.z();
  return J;
}

inline void
point_jac_xyz2uv(const Vector3d & xyz,
                 const Matrix3d & R,
                 const double & focal_length,
                 Matrix<double,2,3> & point_jac)
{
  double x = xyz[0];
  double y = xyz[1];
  double z = xyz[2];
  Matrix<double,2,3> tmp;
  tmp(0,0) = focal_length;
  tmp(0,1) = 0;
  tmp(0,2) = -x/z*focal_length;
  tmp(1,0) = 0;
  tmp(1,1) = focal_length;
  tmp(1,2) = -y/z*focal_length;
  point_jac =  -1./z * tmp * R;
}

inline void
frame_jac_xyz2uv(const Vector3d & xyz,
                 const double & focal_length,
                 Matrix<double,2,6> & frame_jac)
{
  double x = xyz[0];
  double y = xyz[1];
  double z = xyz[2];
  double z_2 = z*z;

  frame_jac(0,0) = -1./z *focal_length;
  frame_jac(0,1) = 0;
  frame_jac(0,2) = x/z_2 *focal_length;
  frame_jac(0,3) =  x*y/z_2 * focal_length;
  frame_jac(0,4) = -(1+(x*x/z_2)) *focal_length;
  frame_jac(0,5) = y/z *focal_length;

  frame_jac(1,0) = 0;
  frame_jac(1,1) = -1./z *focal_length;
  frame_jac(1,2) = y/z_2 *focal_length;
  frame_jac(1,3) = (1+y*y/z_2) *focal_length;
  frame_jac(1,4) = -x*y/z_2 *focal_length;
  frame_jac(1,5) = -x/z *focal_length;
}

inline void
frame_jac_xyz2uvu(const Vector3d & xyz,
                  const Vector2d & focal_length,
                  Matrix<double,3,6> & frame_jac)
{
  double x = xyz[0];
  double y = xyz[1];
  double z = xyz[2];
  double z_2 = z*z;

  frame_jac(0,0) = -1./z *focal_length(0);
  frame_jac(0,1) = 0;
  frame_jac(0,2) = x/z_2 *focal_length(0);
  frame_jac(0,3) =  x*y/z_2 * focal_length(0);
  frame_jac(0,4) = -(1+(x*x/z_2)) *focal_length(0);
  frame_jac(0,5) = y/z *focal_length(0);

  frame_jac(1,0) = 0;
  frame_jac(1,1) = -1./z *focal_length(1);
  frame_jac(1,2) = y/z_2 *focal_length(1);
  frame_jac(1,3) = (1+y*y/z_2) *focal_length(1);
  frame_jac(1,4) = -x*y/z_2 *focal_length(1);
  frame_jac(1,5) = -x/z *focal_length(1);
}

//  /**
//   * Abstract prediction class
//   * Frame: How is the frame/pose represented? (e.g. SE3)
//   * FrameDoF: How many DoF has the pose/frame? (e.g. 6 DoF, that is
//   *           3 DoF translation, 3 DoF rotation)
//   * PointParNum: number of parameters to represent a point
//   *              (4 for a 3D homogenious point)
//   * PointDoF: DoF of a point (3 DoF for a 3D homogenious point)
//   * ObsDim: dimensions of observation (2 dim for (u,v) image
//   *         measurement)
//   */
template <typename Frame,
          int FrameDoF,
          typename Point,
          int PointDoF,
          int ObsDim>
class AbstractPrediction
{
public:

  /** Map a world point x into the camera/sensor coordinate frame T
       * and create an observation*/
  virtual Matrix<double,ObsDim,1>
  map                        (const Frame & T,
                              const Point & x) const = 0;

  virtual Matrix<double,ObsDim,1>
  map_n_bothJac              (const Frame & T,
                              const Point & x,
                              Matrix<double,ObsDim,FrameDoF> & frame_jac,
                              Matrix<double,ObsDim,PointDoF> & point_jac) const
  {
    frame_jac = frameJac(T,x);
    point_jac = pointJac(T,x);
    return map(T,x);
  }

  virtual Matrix<double,ObsDim,1>
  map_n_frameJac             (const Frame & T,
                              const Point & x,
                              Matrix<double,ObsDim,FrameDoF> & frame_jac) const
  {
    frame_jac = frameJac(T,x);
    return map(T,x);
  }

  virtual Matrix<double,ObsDim,1>
  map_n_pointJac             (const Frame & T,
                              const Point & x,
                              Matrix<double,ObsDim,PointDoF> & point_jac) const
  {
    point_jac = pointJac(T,x);
    return map(T,x);
  }


  /** Jacobian wrt. frame: use numerical Jacobian as default */
  virtual Matrix<double,ObsDim,FrameDoF>
  frameJac                   (const Frame & T,
                              const Point & x) const
  {
    double h = 0.000000000001;
    Matrix<double,ObsDim,FrameDoF> J_pose
        = Matrix<double,ObsDim,FrameDoF>::Zero();

    Matrix<double,ObsDim,1>  fun = -map(T,x);
    for (unsigned int i=0; i<FrameDoF; ++i)
    {
      Matrix<double,FrameDoF,1> eps
          = Matrix<double,FrameDoF,1>::Zero();
      eps[i] = h;

      J_pose.col(i) = (-map(add(T,eps),x) -fun)/h ;
    }
    return J_pose;
  }

  /** Jacobian wrt. point: use numerical Jacobian as default */
  virtual Matrix<double,ObsDim,PointDoF>
  pointJac                   (const Frame & T,
                              const Point & x) const
  {
    double h = 0.000000000001;
    Matrix<double,ObsDim,PointDoF> J_x
        = Matrix<double,ObsDim,PointDoF>::Zero();
    Matrix<double,ObsDim,1> fun = -map(T,x);
    for (unsigned int i=0; i<PointDoF; ++i)
    {
      Matrix<double,PointDoF,1> eps
          = Matrix<double,PointDoF,1>::Zero();
      eps[i] = h;

      J_x.col(i) = (-map(T,add(x,eps)) -fun)/h ;

    }
    return J_x;
  }

  /** Add an incermental update delta to pose/frame T*/
  virtual Frame
  add                        (const Frame & T,
                              const Matrix<double,FrameDoF,1> & delta) const = 0;

  /** Add an incremental update delta to point x*/
  virtual Point
  add                        (const Point & x,
                              const Matrix<double,PointDoF,1> & delta) const = 0;
};


template <typename Frame,
          int FrameDoF,
          typename Point,
          int PointDoF,
          int ObsDim>
class AbstractAnchoredPrediction
{
public:

  /** Map a world point x into the camera/sensor coordinate frame T
       * and create an observation*/
  virtual Matrix<double,ObsDim,1>
  map                        (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a) const = 0;

  virtual Matrix<double,ObsDim,1>
  map_n_bothJac              (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a,
                              Matrix<double,ObsDim,FrameDoF> & frame_jac,
                              Matrix<double,ObsDim,PointDoF> & point_jac) const
  {
    frame_jac = frameJac(T_cw,A_wa,x_a);
    point_jac = pointJac(T_cw,A_wa,x_a);
    return map(T_cw,A_wa,x_a);
  }

  virtual Matrix<double,ObsDim,1>
  map_n_allJac               (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a,
                              Matrix<double,ObsDim,FrameDoF> & frame_jac,
                              Matrix<double,ObsDim,FrameDoF> & anchor_jac,
                              Matrix<double,ObsDim,PointDoF> & point_jac) const
  {
    frame_jac = frameJac(T_cw,A_wa,x_a);
    anchor_jac = anchorJac(T_cw,A_wa,x_a);
    point_jac = pointJac(T_cw,A_wa,x_a);
    return map(T_cw,A_wa,x_a);
  }


  /** Jacobian wrt. frame: use numerical Jacobian as default */
  virtual Matrix<double,ObsDim,FrameDoF>
  frameJac                   (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a) const
  {
    double h = 0.000000000001;
    Matrix<double,ObsDim,FrameDoF> J_pose
        = Matrix<double,ObsDim,FrameDoF>::Zero();

    Matrix<double,ObsDim,1>  fun = -map(T_cw,A_wa,x_a);
    for (unsigned int i=0; i<FrameDoF; ++i)
    {
      Matrix<double,FrameDoF,1> eps
          = Matrix<double,FrameDoF,1>::Zero();
      eps[i] = h;

      J_pose.col(i) = (-map(add(T_cw,eps),A_wa,x_a) -fun)/h ;
    }
    return J_pose;
  }

  /** Jacobian wrt. anchor: use numerical Jacobian as default */
  virtual Matrix<double,ObsDim,FrameDoF>
  anchorJac                  (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a) const
  {
    double h = 0.000000000001;
    Matrix<double,ObsDim,FrameDoF> J_pose
        = Matrix<double,ObsDim,FrameDoF>::Zero();

    Matrix<double,ObsDim,1>  fun = -map(T_cw,A_wa,x_a);
    for (unsigned int i=0; i<FrameDoF; ++i)
    {
      Matrix<double,FrameDoF,1> eps
          = Matrix<double,FrameDoF,1>::Zero();
      eps[i] = h;

      J_pose.col(i) = (-map(T_cw,add(A_wa,eps),x_a) -fun)/h ;
    }
    return J_pose;
  }

  /** Jacobian wrt. point: use numerical Jacobian as default */
  virtual Matrix<double,ObsDim,PointDoF>
  pointJac                   (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a) const
  {
    double h = 0.000000000001;
    Matrix<double,ObsDim,PointDoF> J_x
        = Matrix<double,ObsDim,PointDoF>::Zero();
    Matrix<double,ObsDim,1> fun = -map(T_cw,A_wa,x_a);
    for (unsigned int i=0; i<PointDoF; ++i)
    {
      Matrix<double,PointDoF,1> eps
          = Matrix<double,PointDoF,1>::Zero();
      eps[i] = h;

      J_x.col(i) = (-map(T_cw,A_wa,add(x_a,eps)) -fun)/h ;

    }
    return J_x;
  }

  /** Add an incermental update delta to pose/frame T*/
  virtual Frame
  add                        (const Frame & T,
                              const Matrix<double,FrameDoF,1> & delta
                              ) const = 0;

  /** Add an incremental update delta to point x*/
  virtual Point
  add                        (const Point & x,
                              const Matrix<double,PointDoF,1> & delta
                              ) const = 0;
};



/** abstract prediction class dependig on
     * 3D rigid body transformations SE3 */
template <int PointParNum, int PointDoF, int ObsDim>
class SE3_AbstractPoint
    : public AbstractPrediction
    <SE3,6,Matrix<double, PointParNum,1>,PointDoF,ObsDim>
{
public:
  SE3 add(const SE3 &T, const Matrix<double,6,1> & delta) const
  {
    return SE3::exp(delta)*T;
  }
};

class SE3XYZ_STEREO: public SE3_AbstractPoint<3, 3, 3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SE3XYZ_STEREO              (const StereoCamera & cam)
    : _cam(cam)
  {
  }

  Matrix<double,3,6>
  frameJac(const SE3 & se3,
           const Vector3d & xyz)const
  {
    const Vector3d & xyz_trans = se3*xyz;
    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    double f = _cam.focal_length();

    double one_b_z = 1./z;
    double one_b_z_sq = 1./(z*z);
    double A = -f*one_b_z;
    double B = -f*one_b_z;
    double C = f*x*one_b_z_sq;
    double D = f*y*one_b_z_sq;
    double E = f*(x-_cam.baseline())*one_b_z_sq;

    Matrix<double, 3, 6> jac;
    jac <<  A, 0, C, y*C,     z*A-x*C, -y*A,
        0, B, D,-z*B+y*D, -x*D,     x*B,
        A, 0, E, y*E,     z*A-x*E, -y*A;
    return jac;
  }

  Vector3d map(const SE3 & T,
               const Vector3d& xyz) const
  {
    return _cam.map_uvu(T*xyz);
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }


private:
  StereoCamera _cam;
};

#ifdef MONO
class Sim3XYZ : public AbstractPrediction<Sim3,6,Vector3d,3,2>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Sim3XYZ(const LinearCamera & cam)
  {
    this->cam = cam;
  }

  inline Vector2d map(const Sim3 & T,
                      const Vector3d& x) const
  {
    return cam.map(project2d(T*x));
  }


  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

  Sim3 add(const Sim3 &T, const Matrix<double,6,1> & delta) const
  {
    Matrix<double,7,1> delta7;
    delta7.head<6>() = delta;
    delta7[6] = 0;
    return Sim3::exp(delta7)*T;
  }

private:
  LinearCamera  cam;

};

class Sim3XYZ_STEREO : public AbstractPrediction<Sim3,7,Vector3d,3,3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Sim3XYZ_STEREO(const StereoCamera & cam)
  {
    this->cam = cam;
  }

  inline Vector3d map(const Sim3 & T,
                      const Vector3d& x) const
  {
    return cam.map_uvu(T*x);
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

  Sim3 add(const Sim3 &T, const Matrix<double,7,1> & delta) const
  {
    return Sim3::exp(delta)*T;
  }

private:
  StereoCamera  cam;

};

class AbsoluteOrient : public AbstractPrediction<Sim3,7,Vector3d,3,3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AbsoluteOrient()
  {
  }

  inline Vector3d map(const Sim3 & T,
                      const Vector3d& x) const
  {
    return T*x;
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

  Sim3 add(const Sim3 &T, const Matrix<double,7,1> & delta) const
  {
    return Sim3::exp(delta)*T;
  }
};
#endif



/** 3D Euclidean point class */
class SE3XYZ: public SE3_AbstractPoint<3, 3, 2>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SE3XYZ(const LinearCamera & cam)
  {
    this->cam = cam;
  }

  inline Vector2d map(const SE3 & T,
                      const Vector3d& x) const
  {
    return cam.map(project2d(T*x));
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

private:
  LinearCamera  cam;

};

/** 3D inverse depth point class*/
class SE3UVQ : public SE3_AbstractPoint<3, 3, 2>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3UVQ ()
  {
  }

  SE3UVQ (const LinearCamera & cam_pars)
  {
    this->cam = cam_pars;
  }

  inline Vector2d map(const SE3 & T,
                      const Vector3d& uvq_w) const
  {
    Vector3d xyz_w = invert_depth(uvq_w);
    return cam.map(project2d(T*xyz_w));
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

private:
  LinearCamera  cam;
};

/** 3D inverse depth point class*/
class SE3AnchordUVQ : public AbstractAnchoredPrediction<SE3,6,Vector3d,3,2>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3AnchordUVQ ()
  {
  }

  SE3AnchordUVQ (const LinearCamera & cam_pars)
  {
    this->cam = cam_pars;
  }

  inline Vector2d map(const SE3 & T_cw,
                      const SE3 & A_aw,
                      const Vector3d& uvq_a) const
  {
    Vector3d xyz_w = A_aw.inverse()*invert_depth(uvq_a);
    return cam.map(project2d(T_cw*xyz_w));
  }

  Vector3d add(const Vector3d & point,
               const Vector3d & delta) const
  {
    return point+delta;
  }

  Matrix<double,2,3>
  pointJac(const SE3 & T_cw,
           const SE3 & A_aw,
           const Vector3d & psi_a) const
  {
    SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d y = T_ca*invert_depth(psi_a);
    Matrix<double,2,3> J1
        = d_proj_d_y(cam.focal_length(),y);

    Matrix3d J2 = d_Tinvpsi_d_psi(T_ca,  psi_a);
    return -J1*J2;

  }

  Matrix<double,2,6>
  frameJac(const SE3 & T_cw,
           const SE3 & A_aw,
           const Vector3d & psi_a) const
  {
      SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d y = T_ca*invert_depth(psi_a);
    Matrix<double,2,3> J1 = d_proj_d_y(cam.focal_length(),y);
    Matrix<double,3,6> J2 = d_expy_d_y(y);
    return -J1*J2;
  }

  Matrix<double,2,6>
  anchorJac(const SE3 & T_cw,
            const SE3 & A_aw,
            const Vector3d & psi_a) const
  {
    SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d x = invert_depth(psi_a);
    Vector3d y = T_ca*x;
     Matrix<double,2,3> J1
        = d_proj_d_y(cam.focal_length(),y);
    Matrix<double,3,6> d_invexpx_dx
        = -d_expy_d_y(x);
    return -J1*T_ca.rotation_matrix()*d_invexpx_dx;
  }

  SE3 add(const SE3 &T, const Matrix<double,6,1> & delta) const
  {
    return SE3::exp(delta)*T;
  }

private:
  LinearCamera  cam;
};


/** 3D inverse depth point class*/
class SE3NormUVQ : public AbstractPrediction<SE3,5,Vector3d,3,2>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3NormUVQ ()
  {
  }

  SE3NormUVQ (const LinearCamera & cam_pars)
  {
    this->cam = cam_pars;
  }

  inline Vector2d map(const SE3 & T_cw,
                      const Vector3d& uvq_w) const
  {
    Vector3d xyz_w = invert_depth(uvq_w);
    return cam.map(project2d(T_cw*xyz_w));
  }

  Vector3d add(const Vector3d & point,
               const Vector3d & delta) const
  {
    return point+delta;
  }

  SE3 add(const SE3 &T, const Matrix<double,5,1> & delta) const
  {
    Vector6d delta6;
    delta6[0] = delta[0];
    delta6[1] = delta[1];
    delta6[2] = 0;
    delta6.tail<3>() = delta.tail<3>();

    SE3 new_T = SE3::exp(delta6)*T;
    double length = new_T.translation().norm();
    assert(fabs(length)>0.00001);

    new_T.translation() *= 1./length;
    assert(fabs(new_T.translation().norm()-1) < 0.00001);

    return new_T;
  }


private:
  LinearCamera  cam;
};

/** 3D inverse depth point class*/
class SE3AnchordUVQ_STEREO
    : public AbstractAnchoredPrediction<SE3,6,Vector3d,3,3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3AnchordUVQ_STEREO ()
  {
  }

  SE3AnchordUVQ_STEREO (const StereoCamera & cam_pars)
  {
    this->cam = cam_pars;
  }

  inline Vector3d map(const SE3 & T_cw,
                      const SE3 & A_aw,
                      const Vector3d& uvq_a) const
  {
    Vector3d xyz_w = A_aw.inverse()*invert_depth(uvq_a);
    return cam.map_uvu(T_cw*xyz_w);
  }

  Matrix3d
  pointJac(const SE3 & T_cw,
           const SE3 & A_aw,
           const Vector3d & psi_a) const
  {
    SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d y = T_ca*invert_depth(psi_a);
    Matrix3d J1
        = d_stereoproj_d_y(cam.focal_length(),
                           cam.baseline(),
                           y);
    Matrix3d J2
        = d_Tinvpsi_d_psi(T_ca,
                          psi_a);
    return -J1*J2;
  }

  Matrix<double,3,6>
  frameJac(const SE3 & T_cw,
           const SE3 & A_aw,
           const Vector3d & psi_a) const
  {
    SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d y = T_ca*invert_depth(psi_a);
    Matrix3d J1
        = d_stereoproj_d_y(cam.focal_length(),
                           cam.baseline(),
                           y);
    Matrix<double,3,6> J2
        = d_expy_d_y(y);
    return -J1*J2;
  }


  Matrix<double,3,6>
  anchorJac(const SE3 & T_cw,
            const SE3 & A_aw,
            const Vector3d & psi_a) const
  {
    SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d x = invert_depth(psi_a);
    Vector3d y = T_ca*x;
    Matrix3d J1
        = d_stereoproj_d_y(cam.focal_length(),
                           cam.baseline(),
                           y);
    Matrix<double,3,6> d_invexpx_dx
        = -d_expy_d_y(x);
    return -J1*T_ca.rotation_matrix()*d_invexpx_dx;
  }

  Vector3d add(const Vector3d & point,
               const Vector3d & delta) const
  {
    return point+delta;
  }

  SE3 add(const SE3 &T, const Matrix<double,6,1> & delta) const
  {
    return SE3::exp(delta)*T;
  }

private:
  StereoCamera  cam;
};
/** 3D inverse depth point class*/
class SE3UVU_STEREO : public SE3_AbstractPoint<3, 3, 3>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3UVU_STEREO ()
  {
  }

  SE3UVU_STEREO (const StereoCamera & cam)
  {
    this->cam = cam;
  }

  inline Vector3d map(const SE3 & T,
                      const Vector3d& uvu) const
  {
    Vector3d x = cam.unmap_uvu(uvu);
    return cam.map_uvu(T*x);
  }


  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

private:
  StereoCamera  cam;
};

/** 3D inverse depth point class*/
class SE3UVQ_STEREO : public SE3_AbstractPoint<3, 3, 3>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3UVQ_STEREO ()
  {
  }

  SE3UVQ_STEREO (const StereoCamera & cam)
  {
    this->cam = cam;
  }

  inline Vector3d map(const SE3 & T,
                      const Vector3d& uvq) const
  {
    Vector3d x = invert_depth(uvq);
    return cam.map_uvu(T*x);
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

private:
  StereoCamera  cam;
};


/** observation class */
template <int ObsDim>
class IdObs
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IdObs(){}
  IdObs(int point_id, int frame_id, const Matrix<double,ObsDim,1> & obs)
    : frame_id(frame_id), point_id(point_id), obs(obs)
  {
  }

  int frame_id;
  int point_id;
  Matrix<double,ObsDim,1> obs;
};


/** observation class with inverse uncertainty*/
template <int ObsDim>
class IdObsLambda : public IdObs<ObsDim>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IdObsLambda(){}
  IdObsLambda(int point_id,
              int frame_id,
              const Matrix<double,ObsDim,1> & obs,
              const Matrix<double,ObsDim,ObsDim> & lambda)
    : IdObs<ObsDim>(point_id, frame_id,  obs) , lambda(lambda)
  {
  }
  Matrix<double,ObsDim,ObsDim> lambda;
};

}


#endif
