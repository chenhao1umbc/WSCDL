------------------------------------------------------------------------------------------
                   Readme for the MIMLBOOST & MIMLSVM Package at
http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/annex/MIMLBoost&MIMLSVM.htm
	 		       version Jun. 26, 2008
------------------------------------------------------------------------------------------

The package includes the MATLAB code of algorithms MIMLBOOST and MIMLSVM, both which are designed to deal with multi-instance multi-label learning [1]. It is in particular useful when a real-world object is associated with multiple instances as well as multiple labels simultaneously.

[1] Z.-H. Zhou and M.-L. Zhang. Multi-instance multi-label learning with application to scene classification. In: Advances in Neural Information Processing Systems 19 (NIPS'06) (Vancouver, Canada), B. Scholkopf, J. Platt, and T. Hofmann, eds. Cambridge, MA: MIT Press, 2007, pp.1609-1616.

Note that the Matlab version of Libsvm [2] (available at http://www.csie.ntu.edu.tw/~cjlin/libsvm/) is also included here to facilitate the implementation of both MIMLBoost and MIMLSVM. To use this toolbox, please unzip the file "libsvm-mat-2.86-1.rar" to the working directory of Matlab.

[2] C.-C. Chang and C.-J. Lin. Libsvm: a library for support vector machines, Department of Computer Science and Information Engineering, National Taiwan University, Taipei, Taiwan, Technical Report, 2001.


As for MIMLBoost, first use the "MIMLBoost_train" function to train a multi-instance multi-lable prediction model, and then pass it to the "MIMLBoost_test" function to perform testing. As for MIMLSVM, directly use the "MIMLSVM" function to perform training and testing. Note that normalization of the experimental data is necessary before using these functions.

Please type "help MIMLBoost_train", "help MIMLBoost_test" and "help MIMLSVM" to get detailed information about how to use these functions. Furthermore, type "help normalize_bags" to see usage details of the normalization function.

In order to help you understand how to use the package, we have included a sample file sample.m. Also, we have included the partition file 10CV.mat which was used in our experiments reported in our NIPS paper. The index for the ith partition can be found at the index (200*(i-1)+1:200*i). In our experiments we have not finely tune the parameters of the MIML algorithms. For MIMLBoost and MIMLSVM, we use SVM with Gaussian kernel as the base learner. The "cost" parameter is set to the default value 1. For MIMLBoost, $\gamma$ of the Gaussian kernel is set to 1; for MIMLSVM, $\gamma$ is set to 0.2 and $k$ is set to be 20% of the number of training images. We have normalized the data into 0-1. Our normalization files are also included in the package. 


ATTN: 
- This package is free for academic usage. You can run it at your own risk. For other
  purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn).

- This package was developed by Mr. Min-Ling Zhang (zhangml@hhu.edu.cn). For any
  problem concerning the code, please feel free to contact Mr. Zhang.

------------------------------------------------------------------------------------------