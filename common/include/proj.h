/*
 * proj.h
 *
 *  Created on: Nov 21, 2013
 *      Author: rmb-am
 */

#ifndef PROJ_H_
#define PROJ_H_



#include <list>
#include <sophus/se3.h>
#ifdef MONO
#include <sophus/sim3.h>
#endif
#include <visiontools/linear_camera.h>
#include "maths_utils.h"
#include "stereo_camera.h"


using namespace Eigen;
using namespace Sophus;
using namespace VisionTools;


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


inline  Matrix <double, 3, 6>
computeLineProjectionMatrix(Matrix<double,3,4> projectionsMatrix)
{
	//Line Projection Matrix
	//P = (p2 ∧ p3)
	//	  (p3 ∧ p1)
	//    (p1 ∧ p2)

	//Plücker Dual Coordinates
	//L* = P(Q)T - Q(P)T

	//cout << "Projection matrix : " << endl << projectionsMatrix << endl;

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

	//cout << "New Projection matrix : " << endl << result << endl;

	return result;
}

#endif /* PROJ_H_ */
