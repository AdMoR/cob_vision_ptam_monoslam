cmake_minimum_required(VERSION 2.8)

include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

project(cob_vision_ptam_monoslam)

rosbuild_init()
set(CMAKE_VERBOSE_MAKEFILE TRUE)
#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

rosbuild_add_boost_directories()

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/CMakeModules/")



set(EXT_LIBS)


find_package(PCL  REQUIRED)
#if(OPENNI_FOUND)
#    MESSAGE(STATUS "found openni")
    include_directories(/usr/include/openni2) #NEW!!!!!!!
#    include_directories(/opt/ros/fuerte/stacks/perception_pcl/pcl_ros/lib) 
#    include_directories(/opt/ros/fuerte/lib)
     include_directories(/usr/lib)
    include_directories(/opt/ros/groovy/lib/)
    #include_directories(/opt/ros/groovy/lib/)
    add_definitions(-DSCAVISLAM_PCL_SUPPORT)
    include_directories(${PCL_INCLUDE_DIRS})
    MESSAGE(STATUS "pcl_include: " ${PCL_INCLUDE_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    link_directories(${PCL_LIBRARY_DIRS})
    list(APPEND EXT_LIBS ${PCL_LIBRARIES})
#else()
#    MESSAGE(STATUS "openni not found")
#endif()



INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/common/include)
# OpenCV should automatically work because of ROS, but on some Ubuntu configurations
# it seems necessary to set OpenCV_DIR correctly. The next statement will try to
# automatically guess this seting for ROS fuerte.
#if(NOT OpenCV_DIR)
#find_path(OpenCV_DIR "OpenCVConfig.cmake" DOC "Root directory of OpenCV" HINTS "/opt/ros/fuerte/share/OpenCV/")
#endif(NOT OpenCV_DIR)
# Finally, try finding the package


#FIND_PACKAGE(OpenCV REQUIRED)
#FIND_PACKAGE(opencv REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND EXT_LIBS ${OpenCV_LIBS})

find_package(SuiteSparse  REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})
list(APPEND EXT_LIBS ${CSPARSE_LIBRARY})

find_package(Eigen3  REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenGL  REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
list(APPEND EXT_LIBS ${OPENGL_LIBRARIES})

find_package(GLUT  REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})
list(APPEND EXT_LIBS ${GLUT_LIBRARIES})

#find_package(Sophus  REQUIRED)
#include_directories(${Sophus_INCLUDE_DIR})
#list(APPEND EXT_LIBS ${Sophus_LIBRARIES})

#find_package(Pangolin  REQUIRED) #PATHS /home/rmb-am/git/cob_object_perception_intern/)
#include_directories(${Pangolin_INCLUDE_DIR})
#list(APPEND EXT_LIBS ${Pangolin_LIBRARIES})

find_package(Boost COMPONENTS thread filesystem system regex  REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
list(APPEND EXT_LIBS ${Boost_LIBRARIES})

find_package(BLAS  REQUIRED)
include_directories(${BLAS_INCLUDE_DIR})
list(APPEND EXT_LIBS ${BLAS_LIBRARIES})

find_package(LAPACK  REQUIRED)
include_directories(${LAPACK_INCLUDE_DIR})
list(APPEND EXT_LIBS ${LAPACK_LIBRARIES})


#LIST(APPEND CMAKE_LIBRARY_PATH /home/rmb-am/git/cob_object_perception_intern/)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/rmb-am/git/cob_object_perception_intern/libg2o/g2o_src/cmake_modules)
#find_package(G2O  REQUIRED)
#IF(NOT G2O_FOUND)
# MESSAGE(STATUS "hola G2O not found")
#ENDIF(NOT G2O_FOUND)
#IF(G2O_FOUND)
#  MESSAGE(STATUS "aurkitua: " ${G2O_INCLUDE_DIR})
#  include_directories(${G2O_INCLUDE_DIRS})
#  list(APPEND EXT_LIBS ${G2O_LIBRARIES})
#ENDIF(G2O_FOUND)



#get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
#message("inc_dirs = ${inc_dirs}")


#FIND_LIBRARY(LIB_VisionTools VisionTools)
 # IF (LIB_VisionTools)
  #  MESSAGE(STATUS "found library VisionTools: LIB_VisionTools")
  #ELSE (LIB_VisionTools)
  #  MESSAGE(FATAL_ERROR "cannot find library VisionTools - this will not work ...")
  #ENDIF (LIB_VisionTools)

#MESSAGE(STATUS "ttt" ${CMAKE_MODULE_PATH})
#find_package(VisionTools  REQUIRED PATHS /home/rmb-am/git/cob_object_perception_intern/)
#include_directories(${VisionTools_INCLUDE_DIRS})
#list(APPEND EXT_LIBS ${VisionTools_LIBRARIES})

#Add class XYZ here which consits of two files: XYZ.cpp and XYZ.h
set(CLASSES placerecognizer
              maths_utils
              ransac_models
              homography
              anchored_points
              stereo_camera
              backend
              fast_grid
              stereo_frontend
	      mono_frontend
              dense_tracking
              filegrabber
	      rgbd_grabber)


set(TEMPLATE ransac
              frame_grabber
              matcher
              slam_graph )

set(SOURCES ${PROJECT_SOURCE_DIR}/common/include/draw_items.h
             ${PROJECT_SOURCE_DIR}/common/include/global.h
             ${PROJECT_SOURCE_DIR}/common/include/keyframes.h
             ${PROJECT_SOURCE_DIR}/common/include/quadtree.h
             ${PROJECT_SOURCE_DIR}/common/include/transformations.h
             ${PROJECT_SOURCE_DIR}/common/include/data_structures.h
             ${PROJECT_SOURCE_DIR}/common/include/pose_optimizer.h
             ${PROJECT_SOURCE_DIR}/common/include/errorf.h)

#set(PTAM_HEADERS     
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/CameraCalibrator.h   
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/KeyFrame.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/MapViewer.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/Relocaliser.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/TrackerData.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/ATANCamera.h
 # ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/LevelHelpers.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/MEstimator.h
 # ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/ShiTomasi.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/Tracker.h
 # ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/Bundle.h
 # ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/GLWindow2.h
 # ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/Map.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/MiniPatch.h
# ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/SmallBlurryImage.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/CalibCornerPatch.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/GLWindowMenu.h
 # ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/MapMaker.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/OpenGL.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/SmallMatrixOpts.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/CalibImage.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/HomographyInit.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/MapPoint.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/PatchFinder.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/Params.h
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/PtamParamsConfig.h
  #${PROJECT_SOURCE_DIR}/thirdparty/PTAM/System.h
  #${PROJECT_SOURCE_DIR}/thirdparty/PTAM/VideoSource.h
#)

#set(PTAM_SRC
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/CameraCalibrator.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/KeyFrame.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/MapViewer.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/Relocaliser.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/Tracker.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/ATANCamera.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/ShiTomasi.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/Bundle.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/GLWindow2.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/Map.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/MiniPatch.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/SmallBlurryImage.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/CalibCornerPatch.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/GLWindowMenu.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/MapMaker.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/CalibImage.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/HomographyInit.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/MapPoint.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/PatchFinder.cc
  #${PROJECT_SOURCE_DIR}/thirdparty/PTAM/System.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/ARDriver.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/EyeGame.cc
#  ${PROJECT_SOURCE_DIR}/thirdparty/PTAM/Params.cpp
  #${PROJECT_SOURCE_DIR}/thirdparty/PTAM/VideoSource_Linux_V4L.cc
#)


foreach(class ${CLASSES})
    if(EXISTS ${PROJECT_SOURCE_DIR}/common/src/${class}.cpp)
       list(APPEND SOURCES ${PROJECT_SOURCE_DIR}/common/src/${class}.cpp ${PROJECT_SOURCE_DIR}/common/include/${class}.h)
       MESSAGE(STATUS ${PROJECT_SOURCE_DIR}/common/src/${class}.cpp " exists ")
    else()
       MESSAGE(STATUS ${PROJECT_SOURCE_DIR}/common/src/${class}.cpp " doesn't exist")
    endif()
endforeach()


#list(APPEND SOURCES ${PTAM_SRC} ${PTAM_HEADERS})



foreach(template ${TEMPLATE})
    list(APPEND SOURCES ${PROJECT_SOURCE_DIR}/common/src/${template}.cpp
                        ${PROJECT_SOURCE_DIR}/common/include/${template}.hpp
                        ${PROJECT_SOURCE_DIR}/common/src/${template}-impl.cpp)
endforeach()

rosbuild_add_executable(point_cloud_server ${PROJECT_SOURCE_DIR}/common/src/point_cloud_server.cpp)
target_link_libraries(point_cloud_server ${PCL_LIBRARIES})

rosbuild_add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${EXT_LIBS})
rosbuild_add_executable(stereo_slam ${PROJECT_SOURCE_DIR}/common/src/stereo_slam.cpp)
target_link_libraries(stereo_slam ${PROJECT_NAME})

rosbuild_add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${EXT_LIBS})
rosbuild_add_executable(mono_slam ${PROJECT_SOURCE_DIR}/common/src/mono_slam.cpp)
target_link_libraries(mono_slam ${PROJECT_NAME})


rosbuild_add_executable(errorf ${PROJECT_SOURCE_DIR}/common/src/errorf.cpp)
target_link_libraries(errorf ${PROJECT_NAME})

rosbuild_add_executable(error_rec ${PROJECT_SOURCE_DIR}/common/src/error_record.cpp)
target_link_libraries(error_rec ${PROJECT_NAME})

rosbuild_add_executable(ftest ${PROJECT_SOURCE_DIR}/common/src/test.cpp)
target_link_libraries(ftest ${PROJECT_NAME})


#rosbuild_link_boost(cob_vision_ptam_monoslam thread filesystem system regex)
#rosbuild_add_compile_flags(${PROJECT_NAME} -DMONO)
rosbuild_add_compile_flags(${PROJECT_NAME} -D_LINUX)
add_definitions(-std=c++0x)

rosbuild_add_executable(create_dictionary ${PROJECT_SOURCE_DIR}/common/src/create_dictionary.cpp)
target_link_libraries(create_dictionary ${PROJECT_NAME})

rosbuild_add_executable(dump_rgbd ${PROJECT_SOURCE_DIR}/common/src/dump_rgbd.cpp)
target_link_libraries(dump_rgbd ${PROJECT_NAME})
target_link_libraries(mono_slam OpenNI)

