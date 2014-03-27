// BoFSIFT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

#define DICTIONARY_BUILD 0 // set DICTIONARY_BUILD 1 to do Step 1, otherwise it goes to step 2



void test_img(int number,bool train=false){
	
	//Step 2 - Obtain the BoF descriptor for given image/video frame. 

    //prepare BOW descriptor extractor from the dictionary    
	Mat dictionary; 
	FileStorage fs("/home/rmb-am/Slam_datafiles/dictionary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();
    
	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create Sift feature point extracter
	Ptr<FeatureDetector> detector(new SiftFeatureDetector());
	//create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);	
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor,matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	//To store the image file name
	char * filename = new char[100];
	//To store the image tag name - only for save the descriptor in a file


	
	//the image file with the location. change it according to your image file location
	if(!train){
		sprintf(filename,"/home/rmb-am/Slam_datafiles/validation_img/rgb_%i.png",number);
		Mat img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
		vector<KeyPoint> keypoints;
		detector->detect(img,keypoints);
		Mat bowDescriptor;
		bowDE.compute(img,keypoints,bowDescriptor);
	}
	else{
		FileStorage fs("/home/rmb-am/Slam_datafiles/descriptor.yml", FileStorage::WRITE);
		Mat mean=Mat::zeros(1,200,CV_32F);
		for(int i=0;i<50;i++){
			char * imageTag = new char[10];
			sprintf(filename,"/home/rmb-am/Slam_datafiles/training_img_biiig/rgb_%i.png",i);
			//read the image
			Mat img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
			//To store the keypoints that will be extracted by SIFT
			vector<KeyPoint> keypoints;
			//Detect SIFT keypoints (or feature points)
			detector->detect(img,keypoints);
			//To store the BoW (or BoF) representation of the image
			Mat bowDescriptor;
			//extract BoW (or BoF) descriptor from given image
			bowDE.compute(img,keypoints,bowDescriptor);
			mean+=bowDescriptor;
			sprintf(imageTag,"img%i",i);
			fs << imageTag << bowDescriptor;
		}
		double tot = sum(mean).val[0];
		mean/=(tot);
		cout << "ahh" << endl;
		Mat out = 1/mean;
		fs << "inv_freq" << out;
		fs.release();
	}


	if(!train){

		FileStorage fs("/home/rmb-am/Slam_datafiles/descriptor.yml", FileStorage::READ);
		double best_score=0;
		int best_match=-1;
		sprintf(filename,"/home/rmb-am/Slam_datafiles/testing_img/rgb_%i.png",number);


		//read the image
		Mat img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
		//To store the keypoints that will be extracted by SIFT
		vector<KeyPoint> keypoints;
		//Detect SIFT keypoints (or feature points)
		detector->detect(img,keypoints);
		//To store the BoW (or BoF) representation of the image
		Mat bowDescriptor;
		//extract BoW (or BoF) descriptor from given image
		bowDE.compute(img,keypoints,bowDescriptor);


		for(int i=0;i<50;i++){
			Mat descriptor,ifreq;
			char * imageTag = new char[10];
			sprintf(imageTag,"img%i",i);
			fs[imageTag] >> descriptor;
			fs["inv_freq"] >> ifreq;
			double score = (ifreq.mul(ifreq)).dot(bowDescriptor.mul(descriptor));
			//cout << "score with img " << i << " is " << score << endl;

			if(score > best_score){
				best_score=score;
				best_match=i;
			}
		}
		fs.release();

		cout << "BEST MATCH OF " << number <<" = " << best_match << endl;
	}


	//You may use this descriptor for classifying the image.
			


}




int main(int argc, char* argv[] )
{
#if DICTIONARY_BUILD == 1

	//Step 1 - Obtain the set of bags of features.

	//to store the input file names
	char * filename = new char[100];
	//to store the current input image
	Mat input;

	//To store the keypoints that will be extracted by SIFT
	vector<KeyPoint> keypoints;
	//To store the SIFT descriptor of current image
	Mat descriptor;
	//To store all the descriptors that are extracted from all the images.
	Mat featuresUnclustered;
	//The SIFT feature extractor and descriptor
	SiftDescriptorExtractor detector;

	//I select 20 (1000/50) images from 1000 images to extract feature descriptors and build the vocabulary
	for(int f=0;f<50;f++){
		//create the file name of an image
		sprintf(filename,"/home/rmb-am/Slam_datafiles/training_img_biiig/rgb_%i.png",f);
		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale
		//detect feature points
		detector.detect(input, keypoints);
		//compute the descriptors for each keypoint
		detector.compute(input, keypoints,descriptor);
		//put the all feature descriptors in a single Mat object
		featuresUnclustered.push_back(descriptor);
		//print the percentage
		printf("%i percent done\n",2*f);
	}


	//Construct BOWKMeansTrainer
	//the number of bags
	int dictionarySize=200;
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	//retries number
	int retries=1;
	//necessary flags
	int flags=KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
	//cluster the feature vectors
	Mat dictionary=bowTrainer.cluster(featuresUnclustered);
	//store the vocabulary
	FileStorage fs("/home/rmb-am/Slam_datafiles/dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

#else
	for(int j=0;j<70;j++)
		test_img(j,false);
//	else
//		test_img(50,false);
#endif
	printf("\ndone\n");	
    return 0;
}
