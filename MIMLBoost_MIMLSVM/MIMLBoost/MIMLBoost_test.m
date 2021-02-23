function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels,te_time]=MIMLBoost_test(test_bags,test_target,classifiers,c_values,Iter)
%MIMLBoost_test implements the testing procedure of MIMLBOOST as shown in [1].
%
%N.B.: MIMLBoost employs the Matlab version of Libsvm [2] (available at http://sourceforge.net/projects/svm/) as the base learners
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels,te_time]=MIMLBoost_test(test_bags,test_target,classifiers,c_values,Iter)
%
%    Description
%
%       MIMLBoost_test takes,
%           test_bags        - An Mx1 cell, the jth instance of the ith test bag is stored in test_bags{i,1}(j,:)
%           test_target      - A QxM array, if the ith test bag belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           classifiers      - An Iterx1 structure, where parameters of the ith trained base learner is stored in classifiers(i)
%           c_values         - A 1xIter array, where the ith c value is stored in c_value(i)
%           Iter             - The number of trained base learners
%      and returns,
%           HammingLoss      - The hamming loss on testing data as described in [3]
%           RankingLoss      - The ranking loss on testing data as described in [3]
%           OneError         - The one-error on testing data as described in [3]
%           Coverage         - The coverage on testing data as described in [3]
%           Average_Precision- The average precision on testing data as described in [3]
%           Outputs          - A QxM array, the output of the ith testing instance on the jth class is stored in Outputs(j,i)
%           Pre_Labels       - A QxM array, if the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1
%           te_time          - The testing time
%
% [1] Z.-H. Zhou and M.-L. Zhang. Multi-instance multi-label learning with application to scene classification. In: Advances in Neural Information Processing Systems 19 (NIPS'06) (Vancouver, Canada), B. Sch??lkopf, J. Platt, and T. Hofmann, eds. Cambridge, MA: MIT Press, 2007.
% [2] C.-C. Chang and C.-J. Lin. Libsvm: a library for support vector machines, Department of Computer Science and Information Engineering, National Taiwan University, Taipei, Taiwan, Technical Report, 2001.
% [3] Schapire R. E., Singer Y. BoosTexter: a boosting based system for text categorization. Machine Learning, 39(2/3): 135-168, 2000.

     start_time=cputime;
     
     [num_class,num_testing]=size(test_target);
     
     Dim=length(test_bags{1,1}(1,:))+1;
     
     if(Dim<=50)
         ER=1;
     else
         ER=log(Dim)/log(50);
     end
     
     Outputs=zeros(num_class,num_testing);
     Pre_Labels=zeros(num_class,num_testing);
     
     for i=1:num_testing
         temp_bag=test_bags{i,1};
         tempsize=size(temp_bag,1);
         for j=1:num_class
             temp_output=0;
             Samples=[temp_bag,ER*(j-1)/(num_class-1)*ones(tempsize,1)]';
             if(test_target(j,i)==1)
                 Labels=ones(1,tempsize);
             else
                 Labels=-ones(1,tempsize);
             end
             for round=1:Iter
                 model=classifiers(round).model;
                 c_value=c_values(round);
                 
                 [predict_label, accuracy, dec_values] = svmpredict(Labels', Samples', model);
                 temp_output=temp_output+sum(c_value*predict_label);
             end
             Outputs(j,i)=temp_output;
         end
     end
     
     for i=1:num_testing
         for j=1:num_class
             if(Outputs(j,i)>=0)
                 Pre_Labels(j,i)=1;
             else
                 Pre_Labels(j,i)=-1;
             end
         end
     end
     HammingLoss=Hamming_loss(Pre_Labels,test_target);
    
     RankingLoss=Ranking_loss(Outputs,test_target);
     OneError=One_error(Outputs,test_target);
     Coverage=coverage(Outputs,test_target);
     Average_Precision=Average_precision(Outputs,test_target);     
     
     te_time=cputime-start_time;