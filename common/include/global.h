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

#ifndef GLOBAL_H
#define GLOBAL_H

#include <list>
#include <map>
#include <vector>


#include <tr1/unordered_set>
#include <tr1/unordered_map>
#include <tr1/memory>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>



#include <glew.h>
#include <GL/freeglut.h>


// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  void operator=(const TypeName&);

namespace std
{
namespace tr1
{
template<class T>
class hash<shared_ptr<T> > {
public:
  size_t operator()(const shared_ptr<T>& key) const
  {
    return (size_t)key.get();
  }
};
}
}

namespace VisionTools
{
}

namespace ScaViSLAM
{
using namespace Eigen;
using namespace std;
using namespace VisionTools;

typedef Matrix< double, 5, 5 > Matrix5d;
typedef Matrix< float, 5, 5 > Matrix5f;

typedef Matrix< double, 6, 6 > Matrix6d;
typedef Matrix< float, 6, 6 > Matrix6f;

typedef Matrix< double, 7, 7 > Matrix7d;
typedef Matrix< float, 7, 7 > Matrix7f;

typedef Matrix< double, 5, 1 > Vector5d;
typedef Matrix< float, 5, 1 > Vector5f;

typedef Matrix< double, 6, 1 > Vector6d;
typedef Matrix< float, 6, 1 > Vector6f;

typedef Matrix< double, 7, 1 > Vector7d;
typedef Matrix< float, 7, 1 > Vector7f;

typedef Matrix< double, 1, 5 > 	RowVector5d;
typedef Matrix< float, 1, 5 > 	RowVector5f;

typedef Matrix< double, 1, 6 > 	RowVector6d;
typedef Matrix< float, 1, 6 > 	RowVector6f;

typedef Matrix< double, 1, 7 > 	RowVector7d;
typedef Matrix< float, 1, 7 > 	RowVector7f;

typedef struct Line
{
	int global_id;
	int anchor_id;
	Vector6d pluckerLinesObservation;
	Vector6d optimizedPluckerLines;
	std::vector<int> descriptor;
	Vector3d linearForm; //ax+by+c=0
	cv::Point startingPoint2d;
	cv::Point endPoint2d;
	Vector3d startingPoint3d;
	Vector3d endPoint3d;
	int count;

	Line(){}

	Line(int id,int anch_id,Vector3d linForm,std::vector<int> desc,Vector6d pluckerLines,int c,cv::Point startingPoint, cv::Point endPoint, Vector3d startingP3d, Vector3d endingP3d ){
		global_id=id;
		anchor_id=anch_id;
		linearForm = linForm;
		descriptor = desc;
		pluckerLinesObservation = pluckerLines;
		optimizedPluckerLines = pluckerLines;
		count = c;
		startingPoint2d = startingPoint;
		endPoint2d = endPoint;
		startingPoint3d = startingP3d;
		endPoint3d = endingP3d;
	}

	//typedef typename ALIGNED<Line>::int_hash_map LTable; //line feature table

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <int T>
struct VECTOR
{
  typedef Matrix<double,T,1> col;
  typedef Matrix<double,1,T> row;
};

typedef cv::Rect_<double> Rectangle;

const double EPS = 0.0000000001;
const int NUM_PYR_LEVELS  = 3;

class IntPairHash
{
public:
  size_t operator ()(const pair<int,int> & p) const
  {
    return tr1::hash<int>()(p.first) + tr1::hash<int>()(100000*p.second);
  }
};

// Defining a templated-struced is necessary since templated typedefs are only
// allowed in the upcoming C++0x standard!
template <class T>
struct ALIGNED
{
  typedef std::vector<T, Eigen::aligned_allocator<T> > vector;
  typedef std::list<T, Eigen::aligned_allocator<T> > list;
  typedef std::map<int,T, less<int>,

  Eigen::aligned_allocator<pair<const int, T> > >
  int_map;
  typedef tr1::unordered_map<int,T, tr1::hash<int>,  equal_to<int>,
  Eigen::aligned_allocator<pair<const int, T> > >
  int_hash_map;
  typedef tr1::unordered_map<pair<int,int>, T, IntPairHash,
  equal_to<pair<int, int> >,
  Eigen::aligned_allocator<pair<const pair<int,int>, T> > >
  intpair_hash_map;
};

typedef tr1::unordered_map<int,int> IntTable;
}

#endif
