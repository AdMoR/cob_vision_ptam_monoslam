/*
 * mixture_recovery.cpp
 *
 *  Created on: Jan 28, 2014
 *      Author: rmb-am
 */
#include "rgbd_line.cpp"
#include "random_tree_rgbd.cpp"
#include <global.h>
#include <transformations.h>




void findNearestLinesOnCurrentFrameOrderedByDistance(Vector3d line, std::vector<Line> &linesOnCurrentFrame,float err,map<float,Line>& candidates)
{
	map<double,Line> nearestLines;
	double projectedtLineMag=0;
	double projectedLineAng=0;
	cartesianToPolar(line, projectedtLineMag,projectedLineAng);
	Vector2d found_point;
	double angle_weight=5;


	for (auto it = linesOnCurrentFrame.begin(); it != linesOnCurrentFrame.end(); ++it)
	{
		double currentLineMag=0;
		double currentLineAng=0;
		double dist=1000000;
		cartesianToPolar((*it).linearForm, currentLineMag, currentLineAng);
		double dang=min(abs(currentLineAng-projectedLineAng),(double)abs(int(currentLineAng+10)%180-int(projectedLineAng+10)%180));

		dist = computePolarDistance(projectedtLineMag, projectedLineAng, currentLineMag, currentLineAng, angle_weight);

		if(dist <= (double)err)
		{
			candidates.insert( std::pair<float,Line>(dist, (*it))); //should be ordered
		}

	}

}

void quickMatching(vector<pair<int,pair<int,float> > >& matches, vector<Line>& tracked_lines, vector<Line>& newLines, SE3& transform){

	for(auto ptr = tracked_lines.begin(); ptr!=tracked_lines.end();ptr++){
		 Matrix<double,3,3> camera_matrix;
		 camera_matrix << 530,0,240,
				 	 	 0,530,320,
				 	 	  0,0,1;

		 Matrix<double, 3, 4> projectionsMatrix = computeProjectionMatrix(camera_matrix, transform.matrix() );
		 Vector3d projectedHomogeneousLine = computeLineProjectionMatrix2(projectionsMatrix, toPlueckerMatrix(ptr->optimizedPluckerLines));
		 map<float,Line> candidates;
		 bool loop=true;
		 projectedHomogeneousLine.normalize();
		 if (projectedHomogeneousLine[2] < 0.0)
		{
			changeSigns(projectedHomogeneousLine);
		}



		findNearestLinesOnCurrentFrameOrderedByDistance(projectedHomogeneousLine,newLines,100,candidates);
		for(auto it = candidates.begin(); it!=candidates.end();it++)
			cout << it->first << " " ;
		cout << endl;

		for(auto pointer=candidates.rbegin();pointer!=candidates.rend() && loop;pointer++){
			if(score(ptr->d_descriptor,pointer->second.d_descriptor)<1.3){

				matches.push_back(make_pair((*ptr).global_id,make_pair( (*pointer).second.global_id,(*pointer).first) ));
				loop=false;
				}
		 }
		if(loop){
			matches.push_back(make_pair((*ptr).global_id,make_pair(-1,10000)));
		}

	}


}



//Function used on every loop of the particle filter to update the prior
void pose_prior_modification(vector<pair<SE3,float> >& pose_prior, vector<CameraPoseHypothesis>& pose_hyp, int nb_prior,float max_err_l, float max_err_rt, float weight=140000, float lambda=10000){

	begin:
	pose_prior.clear();
	map<float,CameraPoseHypothesis> scores;

	for(auto ptr = pose_hyp.begin();ptr != pose_hyp.end();ptr++){
		scores.insert(make_pair(exp(-(ptr->error+weight*ptr->error_line)/lambda),(*ptr)));
	}

	float max_err=0;
	int i=0;
	for(auto ptr = scores.rbegin();ptr!=scores.rend();ptr++){
		max_err+=ptr->first;
		i++;
		if(i==nb_prior)
			break;
	}

	i=0;
	float cumulative_prob=0;
	if(max_err!=0){
		for(auto ptr = scores.rbegin();ptr!=scores.rend();ptr++){
			if(i!=nb_prior-1){
				//assert(max_err==0);
				pose_prior.push_back((make_pair(ptr->second.pose,cumulative_prob+ptr->first/max_err)));
				cumulative_prob+=ptr->first/max_err;
				i++;}
			else{
				pose_prior.push_back((make_pair(ptr->second.pose,1)));
				return;
			}
		}
	}
	else{
		lambda=lambda*5;
		cout << "USE GOTO" << endl;
		goto begin;
	}

}


//A quick modification of the Pluecker of a line
void transform_line(vector<Line>& lineOnFrame, SE3& tf){

	for(auto line_p=lineOnFrame.begin();line_p!=lineOnFrame.end();line_p++){
		Matrix<double,4,4> inv_tf = tf.matrix().inverse();
		line_p->optimizedPluckerLines=toPlueckerVec(inv_tf*toPlueckerMatrix(line_p->GTPlucker)*inv_tf.transpose());
		line_p->optimizedPluckerLines.normalize();
	}

}

//Relative mean square error
double RMSE(SE3& GTn,SE3& GTnm1,SE3& Pn,SE3& Pnm1){

	SE3 first_m = GTnm1.inverse()*GTn, second_m=Pnm1.inverse()*Pn;
	SE3 out = first_m.inverse()*second_m;

	return out.translation().norm();

}


int main(int argc, char* argv[]){

	const bool particle_filter_mode = true;
	//The particle filter is tested in this function, the test is similar to the one in the regression forst but the score is modified by the info of lines

	if(particle_filter_mode){


	const int nb_random_param_per_node=5000;//5000;
	const int nb_of_trees=5;
	const int max_depth=17;


	const bool training = false;

	const int nb_pose_hyp = 16;
	const int nb_prior = 2;

	ros::init(argc, argv,"mixture_recovery");


	vector<SplitNode*> forest;
	DatasetBuilder empty(0);
	vector<Point> zero=vector<Point>();
	vector<vector<cv::Mat> > channel_v;
	vector<Vector3d> nothing;

	for(auto ptr=empty.rgb_img.begin();ptr!=empty.rgb_img.end();ptr++){
		vector<Mat> channels;
		split((*ptr),channels);
		channel_v.push_back(channels);
	}

	//Create the root
	//cout << "root created " << root->isLeaf << endl;
	//Read the parameters
	for(int i = 0; i<nb_of_trees;i++){
		stringstream ss;
		ss << "/home/rmb-am/Slam_datafiles/read_a_forest_" << i <<".rt";
		forest.push_back( new SplitNode(zero,nothing,channel_v,empty.d_img,0,nb_random_param_per_node,max_depth,training,false) );
		readTree(forest[i],ss.str().c_str());
	}

	//Load data
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img_biiig/","/home/rmb-am/Slam_datafiles/frame_label_biiig.txt");
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img_long/","/home/rmb-am/Slam_datafiles/frame_labels.txt");
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img_3/","/home/rmb-am/Slam_datafiles/labels20.txt");
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img_4/","/home/rmb-am/Slam_datafiles/labels_test.txt");
	empty.load_from_training("/home/rmb-am/Slam_datafiles/validation_img/","/home/rmb-am/Slam_datafiles/frame_labels_validation.txt");
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/testing_img/","/home/rmb-am/Slam_datafiles/frame_labels_testing.txt");
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/testing_img_bis/","/home/rmb-am/Slam_datafiles/labels_testing_bis");


	//See the prediction on a training set image
	vector<Point> pxSet;
	vector<Vector3d> local_labels;
	Vector3d unknown_position = Vector3d(0,0,0);
	SE3 null_tf=SE3(),last_GT;
	vector<vector<CameraPoseHypothesis> > pose_models;



	vector<pair<SE3,float> > pose_prior;
	//pose_prior.push_back(make_pair(previous_transform,1));

	for(int i=0;i<10;i++){
		double current_RMSE=-1;
		pose_prior.clear();
		SE3 previous_transform=empty.pose_vector[0];
		pose_prior.push_back(make_pair(previous_transform,1));

		for(int le_numero_gagnant = 1; le_numero_gagnant<empty.rgb_img.size();le_numero_gagnant++){

			float  max_error_line=0,max_error_rt=0;
			vector<CameraPoseHypothesis> pose_hyp;
			cv::Mat hsv,hsv2;
			vector<cv::Mat> hsv_channel,hsv_channel2;
			vector<Line> lines_on_first,lines_on_second;

			double  durationA = cv::getTickCount();

			previous_transform=empty.pose_vector[le_numero_gagnant];//empty.pose_vector[le_numero_gagnant-1];

			refine_phase(forest,empty.rgb_img[le_numero_gagnant],empty.d_img[le_numero_gagnant],10,1000,nb_pose_hyp,pose_hyp,pose_prior,true,0.1);
			pose_models.push_back(pose_hyp);

			cv::cvtColor(empty.rgb_img[le_numero_gagnant], hsv, CV_BGR2HSV);
			cv::cvtColor(empty.rgb_img[le_numero_gagnant-1], hsv2, CV_BGR2HSV);
			split(hsv,hsv_channel);
			split(hsv2,hsv_channel2);



			Mat rgb = empty.rgb_img[le_numero_gagnant].clone(), d = empty.d_img[le_numero_gagnant].clone();
			build_normal_map(d,rgb,hsv_channel[1],lines_on_first,null_tf,45,60,8);

			//TODO implement a correct way to extract learnt lines
			//It should find the frame with the closest pose or a set of closest frame
			rgb = empty.rgb_img[le_numero_gagnant].clone();
			d = empty.d_img[le_numero_gagnant].clone();
			build_normal_map(d,rgb,hsv_channel2[1],lines_on_second,empty.pose_vector[le_numero_gagnant],45,60,8);


			SE3 null_transform=SE3(),tf=previous_transform;


			for(auto pose_ptr = pose_hyp.begin();pose_ptr!=pose_hyp.end();pose_ptr++){
				//For every hypothesis

				float error = 0 ;
				//We get the line coordinates
				transform_line(lines_on_first,pose_ptr->pose);
				map<int,Line> already_matched;


				for(int i = 0 ; i<lines_on_second.size();i++){


					Vector6d vec1;
					vec1=lines_on_second[i].optimizedPluckerLines;
					vec1.normalize();

					//Look for a match from the previous frame
					Line l=find_closest(lines_on_second[i],lines_on_first,already_matched);

					if(l.global_id!=-1)
						already_matched.insert(make_pair(l.global_id,l));
					Vector6d vec2 = l.optimizedPluckerLines;
					vec2.normalize();
					float line_error;
					if(l.global_id==-1)
						line_error=2;
					else
						line_error=plucker_similarity(vec1,vec2)+score(lines_on_second[i].d_descriptor,l.d_descriptor);
					//And add the error
					error+=line_error;

				}

				if(lines_on_second.size()!=0)
					pose_ptr->error_line=(error/lines_on_second.size());
				else
					pose_ptr->error_line=0;




				max_error_line+=pose_ptr->error_line;
				max_error_rt+=pose_ptr->error;

				//Debug to file, gives the score of each step for each hypotheses, useful to know if the system is doing its job
				Matrix<double,4,4> diff = pose_ptr->pose.matrix(), GT=previous_transform.matrix();
				ofstream os;
				os.open("/home/rmb-am/Slam_datafiles/mixture_data_bad_lines.txt",ios::app);
				os <<le_numero_gagnant<<" " << pose_ptr->error << " " << pose_ptr->error_line << " " << myMatrixNorm(GT,diff) << " " <<  current_RMSE << " " << se3_metric(previous_transform,pose_ptr->pose,0.7,10) <<endl;
				os.close();


			cout << "duration : "<<(getTickCount()-durationA)/getTickFrequency() << endl;

			//After the match phase, we modify the prior according to the hypotheses scores
			pose_prior_modification(pose_prior,pose_hyp,nb_prior,max_error_line,max_error_rt,140000,8000);


			//Another debug, it gives the score of the prior on different metrics
			for(int i=0;i<pose_prior.size();i++){
				Matrix<double,4,4> diff = pose_prior[i].first.matrix(), GT=previous_transform.matrix();
			//	cout << "number " << i << " does an error of " << myMatrixNorm(GT,diff)<<endl;
				ofstream os;
				os.open("/home/rmb-am/Slam_datafiles/inlier_thr_effect_bad_lines_3.txt",ios::app);
				os <<le_numero_gagnant<<" "  << myMatrixNorm(GT,diff) << " " << pose_prior[i].second  << " " << se3_metric(previous_transform,pose_prior[i].first,0.7,10)  << " " << se3_metric(previous_transform,pose_prior[i].first,0.6,5)  << " " << se3_metric(previous_transform,pose_prior[i].first,0.5,5) <<endl;
				os.close();
			}

			}
		}
	}

	//Destroy the forst to clean the memory
	for(int i = 0; i<nb_of_trees;i++)
		destroyTree(forest[i]);


	}

	else{


	//This has been developed to test the lines only as it appeared to be challenging too
	double line_match_thres=0.6;
	int line_vote  = 80;
	int inter_line=8;


	ros::init(argc, argv,"line");


	DatasetBuilder empty(0);

	//Load data
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img_biiig/","/home/rmb-am/Slam_datafiles/frame_label_biiig.txt");
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img_long/","/home/rmb-am/Slam_datafiles/frame_labels.txt");
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img_3/","/home/rmb-am/Slam_datafiles/labels20.txt");
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/training_img_4/","/home/rmb-am/Slam_datafiles/labels_test.txt");
	empty.load_from_training("/home/rmb-am/Slam_datafiles/validation_img/","/home/rmb-am/Slam_datafiles/frame_labels_validation.txt");
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/testing_img/","/home/rmb-am/Slam_datafiles/frame_labels_testing.txt");
	//empty.load_from_training("/home/rmb-am/Slam_datafiles/testing_img_bis/","/home/rmb-am/Slam_datafiles/labels_testing_bis");


	//See the prediction on a training set image
	for(int i=40 ; i<100 ; i=i+10){
		line_vote=i;
		for(int j=2 ; j<10 ; j++){
			inter_line=j;
			for(double p=0.5 ; p<1.5 ; p=p+0.1){
				line_match_thres=p;

				for(int le_numero_gagnant = 1; le_numero_gagnant<empty.rgb_img.size();le_numero_gagnant++){

					cv::Mat hsv,hsv2;
					vector<cv::Mat> hsv_channel,hsv_channel2;
					vector<Line> lines_on_current,lines_on_previous;



					cv::cvtColor(empty.rgb_img[le_numero_gagnant], hsv, CV_BGR2HSV);
					cv::cvtColor(empty.rgb_img[le_numero_gagnant-1], hsv2, CV_BGR2HSV);
					split(hsv,hsv_channel);
					split(hsv2,hsv_channel2);


					Mat rgb = empty.rgb_img[le_numero_gagnant].clone(), d = empty.d_img[le_numero_gagnant].clone();
					build_normal_map(d,rgb,hsv_channel[1],lines_on_current,empty.pose_vector[le_numero_gagnant],45,line_vote,inter_line);

					rgb = empty.rgb_img[le_numero_gagnant-1].clone();
					d = empty.d_img[le_numero_gagnant-1].clone();
					build_normal_map(d,rgb,hsv_channel2[1],lines_on_previous,empty.pose_vector[le_numero_gagnant-1],45,line_vote,inter_line);




						float error = 0 ;

						map<int,Line> already_matched;

						for(int i = 0 ; i<lines_on_previous.size();i++){
							Vector6d vec1;
							vec1=lines_on_previous[i].optimizedPluckerLines;
							vec1.normalize();
							Line l=find_closest(lines_on_previous[i],lines_on_current,already_matched,line_match_thres);
							if(l.global_id!=-1)
								already_matched.insert(make_pair(l.global_id,l));
							Vector6d vec2 = l.optimizedPluckerLines;
							vec2.normalize();
							float line_error;
							if(l.global_id==-1)
								line_error=2;
							else
								line_error=2*plucker_similarity(vec1,vec2);//+0.5*score(lines_on_previous[i].d_descriptor,l.d_descriptor);
							error+=line_error;
						}

						ofstream os;
						os.open("/home/rmb-am/Slam_datafiles/line_match_6.txt",ios::app);
						os << le_numero_gagnant << " " << line_vote << " " << inter_line << " " << line_match_thres << " : " << error << " " << already_matched.size() << " " << lines_on_current.size() << " " << lines_on_previous.size() <<endl;
						os.close();



				}
			}
			}
		}





	}
}

