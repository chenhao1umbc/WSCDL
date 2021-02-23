#include "mex.h"
#include <math.h>
#include <stdlib.h>


void printf_vec(double *arr_double_ind, int n)
 {
     int i;
     for (i=0;i <n; ++i)
     {
         mexPrintf("%f\n",arr_double_ind[i]);
         mexEvalString("drawnow");
     }    
 }

//Note that in C, index= no_of_rows * col_id + row_id
void printf_mat(double *arr_double_ind, int m, int n)
 {
     int i,j;
     for (i=0;i <m; ++i)
     {
         for (j=0; j<n; j++)
         {
            mexPrintf("%f, ",arr_double_ind[j* m+ i]);
         }
         mexPrintf("\n");
     }    
 }

void printf_mat_colseg(double *arr_double_ind, int m, int startn,int endn)
 {
     int i,j;
     for (i=0;i <m; ++i)
     {
         for (j=startn; j<endn; j++)
         {
            mexPrintf("%f, ",arr_double_ind[j* m+ i]);
         }
         mexPrintf("\n");
     }    
 }

double sum_r(double *arr, int row, int n_row, int n_col)
{
    int j;
    double sum_=0;
    for (j=0; j<n_col; j++)
    {       
        sum_=sum_+arr[j* n_row + row];
    }
    return sum_;
}

int find_arr(double *arr, int row, int n_row, int n_col, double val)
{
    int j;
    for (j=0; j<n_col; j++)
    {
        if(arr[j * n_row + row]==val)
        {
            return j;
        }
    }
    return -1;
}


int * find_arr_list(double *arr, int row, int n_row, int n_col, int list_count, double val)
{
    int *list= (int*)mxCalloc((mwSize)list_count,(mwSize)sizeof(int));
    
    int j=0;
    
    int count=0;
    
    for (j=0; j<n_col; j++)
    {
        if(arr[j * n_row + row]==val)
        {
            list[count]=j;
            count=count+1;
        }
    }
    return list;
}

double * project_row(double *arr, double *result, int row, int no_of_rows, int no_of_cols)
{
//     double *result= (double*)mxCalloc((mwSize)no_of_cols,(mwSize)sizeof(double));
    
    int i;
    
    for (i=0; i<no_of_cols; i++)
    {
        result[i]=arr[i*no_of_rows+row];
    }
    return result;
}

int ceil_log2(unsigned long long x)
{
  static const unsigned long long t[6] = {
    0xFFFFFFFF00000000ull,
    0x00000000FFFF0000ull,
    0x000000000000FF00ull,
    0x00000000000000F0ull,
    0x000000000000000Cull,
    0x0000000000000002ull
  };

  int y = (((x & (x - 1)) == 0) ? 0 : 1);
  int j = 32;
  int i;

  for (i = 0; i < 6; i++) {
    int k = (((x & t[i]) == 0) ? 0 : j);
    y += k;
    x >>= k;
    j >>= 1;
  }

  return y;
}

//compute convolution
double * convolution_limit(double *arr1, int size1, double *arr2, int size2, int limit, double *result)
{
    int length=0;
    int n;
//     if (strcmp(str, "full")==0)
//     {
//     double *result= (double*)mxCalloc((mwSize)(size1+size2-1),(mwSize)sizeof(double));
    int kmin;
    int kmax;     
    int k;
    length=(limit<(size1+size2-1))?limit:(size1+size2-1);
//     }else
//     {
//         length=size1>size2? size1-size2+1:size2-size1+1;
//         double *result= (double*)mxCalloc((mwSize)length,(mwSize)sizeof(double));
//     }
        
    for (n=0; n<length; n++)
    {
        kmin = (n >= size2-1)? n-(size2-1) : 0;
        kmax = (n < size1-1)? n : size1-1;
        for ( k = kmin; k <= kmax; k++)
        {
            result[n]+=arr1[k]*arr2[n-k];
        }
    }
    return result;
}

//compute convolution valid
double * xcorr_limit(double *arr1, int size1, double *arr2, int size2, int limit, double *result)
{
    int n, k;
//     double *result= (double*)mxCalloc((mwSize)(size1-size2+1),(mwSize)sizeof(double));
    for (n=0; n<limit; n++)
    {
        for ( k = 0; k < size2; k++)
        {
            if (n+k<size1)
            {
                result[n]+=arr1[n+k]*arr2[k];
            }
        }
    }
    return result;
}

//Note that in C, index= no_of_rows * col_id + row_id
double *normalize_mat(double *arr_double_ind, int m, int n, int col_id)
 {
    double sum_=0;
    int i;
    for (i=0; i<m; i++)
    {       
        sum_=sum_+arr_double_ind[m* col_id + i];
    }
    for (i=0; i<m; i++)
    {       
        arr_double_ind[m* col_id + i]=arr_double_ind[m* col_id + i]/sum_;
    }
    return arr_double_ind;
    
 }


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    //define an 2-d Cell
    typedef struct{
        int col_size;
        double * stateprob;
        double * constmat;
    } stprobclass;
 
    //Input variables
    double *Nbmax;
    double *prior;
    double *str;
    double *shift_table;
    //Internal variables
    stprobclass *stateprior, *statepost;
    double *priorn_unnormalized;
    double *priorn;
    double *condp, *postp;
    int class_count;
    int *class_list;
    size_t powerC;     
    size_t C;
    size_t no_ins;
    //Output variables
    double *outMatrix;
    double *post;
    //Temp variables
    int i, j, k, k1, num, h, a, n, booltmp;
    int l, L;
    int tmpidx, tmpidx2, size1, size2, size3;
    double temp1, temp2, temp3, temp4, tempsum;
    double *sig1, *sig2, *sig3, *cons;
    double lastconst, c1, c2, lc1, lc2, lc3;
    double *v1, *v2, *v3;
    
    Nbmax = mxGetPr(prhs[0]);
    prior = mxGetPr(prhs[1]);
    str = mxGetPr(prhs[2]);
    shift_table = mxGetPr(prhs[3]);
    
    powerC = mxGetM(prhs[2]);
    C = mxGetN(prhs[2]);
    no_ins = mxGetN(prhs[1]);
    L = ceil_log2((unsigned long long) no_ins);
//     mexPrintf("%d, %d\n",(int)no_ins, L);
    
    stateprior = (stprobclass*)mxCalloc((mwSize)(L+1)*(int)no_ins,(mwSize)sizeof(stprobclass));
    //Initialize the first level value
    for (j=0; j<(int) no_ins; j++)
    {
        stateprior[j*(L+1)].col_size = 2;
        stateprior[j*(L+1)].stateprob = (double*)mxCalloc((mwSize)((int)powerC*2),(mwSize)sizeof(double));
        stateprior[j*(L+1)].constmat = (double*)mxCalloc((mwSize)((int)powerC*1),(mwSize)sizeof(double));
         for (i=0; i <(int)powerC; i++)
        {
            a=sum_r(str,i,(int)powerC,(int)C);
            if((int)a==1)
            {
                l=find_arr(str,i,(int)powerC,(int)C,1);
                stateprior[j*(L+1)].constmat[i] = log(prior[j*(int)C+l]);
                if(l == (int)C-1)
                {
                    stateprior[j*(L+1)].stateprob[i]= 1;
                }else {
                    stateprior[j*(L+1)].stateprob[1*(int)powerC+i] = 1;
                }
            } else
            {
                stateprior[j*(L+1)].constmat[i] = 0;
                stateprior[j*(L+1)].stateprob[i] = 0;
                stateprior[j*(L+1)].stateprob[1*(int)powerC+i] = 0;
            }
        }
//         printf_mat(stateprior[j*(L+1)].stateprob, (int)powerC, 2);
//         mexPrintf("\n");
//         printf_mat(stateprior[j*(L+1)].constmat, (int)powerC, 1);
//         mexPrintf("\n");
    }
   
    //forward algorithm
    num = (int) no_ins;
    booltmp=0;
    class_count=sum_r(str,powerC-1,powerC,C);
    class_list=find_arr_list(str, powerC-1, powerC, C, class_count, 1);
    
    for (k=0; k<L; k++)
    {
//         mexPrintf("%d\n", num);
        if (num > 1 && num%2 == 1)
        {
//             mexPrintf("%d\n", num/2);
            stateprior[(int)(num/2)*(L+1) + (k+1)].col_size = stateprior[(num-1)*(L+1) + k].col_size;
            stateprior[(int)(num/2)*(L+1) + (k+1)].stateprob = stateprior[(num-1)*(L+1) + k].stateprob;
            stateprior[(int)(num/2)*(L+1) + (k+1)].constmat = stateprior[(num-1)*(L+1) + k].constmat;
//             printf_mat(stateprior[(int)(num/2)*(L+1) + (k+1)].stateprob, (int)powerC, stateprior[(int)(num/2)*(L+1) + (k+1)].col_size);
//             mexPrintf("\n");
            num = num-1;
            booltmp = 1;
        }
        num=num/2;
        for (k1=0; k1<num; k1++)
        {
            size1 = stateprior[(2*k1)*(L+1) + k].col_size;
            size2 = stateprior[(2*k1+1)*(L+1) + k].col_size;
            size3 = (((int)(*Nbmax)+1) < (size1 + size2 -1))? (int)(*Nbmax)+1 : size1 + size2 -1;
            sig1 = (double*)mxCalloc((mwSize)size1,(mwSize)sizeof(double));
            sig2 = (double*)mxCalloc((mwSize)size2,(mwSize)sizeof(double));
            sig3 = (double*)mxCalloc((mwSize)size3,(mwSize)sizeof(double));
//             mexPrintf("%d, %d, %d\n",size1, size2, size3);
            
            stateprior[(k1)*(L+1) + (k+1)].col_size = size3;
            stateprior[(k1)*(L+1) + (k+1)].stateprob = (double*)mxCalloc((mwSize)((int)powerC*size3),(mwSize)sizeof(double));
            stateprior[(k1)*(L+1) + (k+1)].constmat = (double*)mxCalloc((mwSize)((int)powerC*1),(mwSize)sizeof(double));
            for (i=0; i<(int)powerC; i++)
            {
                sig1=project_row(stateprior[(2*k1)*(L+1) + k].stateprob, sig1, i, (int)powerC, size1);
                temp1=sum_r(sig1, 0, 1, size1);
                for (j=0; j<(int)powerC; j++)
                {
                    sig2=project_row(stateprior[(2*k1+1)*(L+1) + k].stateprob, sig2, j, (int)powerC, size2);
                    temp2=sum_r(sig2, 0, 1, size2);
                        //mexPrintf("%d, %d, %d, %d\n", k, k1,i, j);
                        if(temp1 != 0 && temp2 != 0)
                        {
                            for (h=0; h<size3; h++)
                            {
                                sig3[h]=0;
                            }
                           sig3=convolution_limit(sig1, size1, sig2, size2,(int)(*Nbmax)+1,sig3);
                           //adding normalization part
                           lc1=stateprior[k1*(L+1)+ (k+1)].constmat[(int) shift_table[i*(int)powerC+j]-1];
                           lc2=stateprior[(2*k1)*(L+1) + k].constmat[i] + stateprior[(2*k1+1)*(L+1) + k].constmat[j];
                           if (lc1==0)
                           {
                               for (h=0; h<size3; h++)
                               {
                                   stateprior[k1*(L+1)+ (k+1)].stateprob[h*(int)powerC+ (int) shift_table[i*(int)powerC+j]-1] = sig3[h]; 
                               }                               
                               stateprior[k1*(L+1)+ (k+1)].constmat[(int) shift_table[i*(int)powerC+j]-1] = lc2;
                           } else 
                           {
                               if (lc1 >= lc2)
                               {
                                  stateprior[k1*(L+1)+ (k+1)].constmat[(int) shift_table[i*(int)powerC+j]-1] = lc1+log(1+exp(lc2-lc1));
                                  for (h=0; h<size3; h++)
                                  {
                                      stateprior[k1*(L+1)+ (k+1)].stateprob[h*(int)powerC+ (int) shift_table[i*(int)powerC+j]-1] = 1/(1+exp(lc2-lc1))*stateprior[k1*(L+1)+ (k+1)].stateprob[h*(int)powerC+ (int) shift_table[i*(int)powerC+j]-1] + exp(lc2-lc1)/(1+exp(lc2-lc1))*sig3[h]; 
                                  }  
                               } else
                               {
                                  stateprior[k1*(L+1)+ (k+1)].constmat[(int) shift_table[i*(int)powerC+j]-1] = lc2+log(1+exp(lc1-lc2));
                                  for (h=0; h<size3; h++)
                                  {
                                      stateprior[k1*(L+1)+ (k+1)].stateprob[h*(int)powerC+ (int) shift_table[i*(int)powerC+j]-1] = exp(lc1-lc2)/(1+exp(lc1-lc2))*stateprior[k1*(L+1)+ (k+1)].stateprob[h*(int)powerC+ (int) shift_table[i*(int)powerC+j]-1] + 1/(1+exp(lc1-lc2))*sig3[h]; 
                                  } 
                               }
                           }
                        }
                    }
                }
            mxFree(sig1);
            mxFree(sig2);
            mxFree(sig3);
//                 printf_mat(stateprior[k1*(L+1)+ (k+1)].stateprob, (int)powerC, size3);
//                 mexPrintf("\n");  
            }
            if (booltmp == 1)
            {
               num = num + 1;
               booltmp = 0;
            }
        }
    
    //calculate bag label probability
    priorn_unnormalized = (double*)mxCalloc((mwSize)(int)(*Nbmax)+1,(mwSize)sizeof(double));
    priorn = (double*)mxCalloc((mwSize)(int)(*Nbmax)+1,(mwSize)sizeof(double));
    
    for (k=0; k<=(int)(*Nbmax); k++)
    {
        priorn_unnormalized[k] = stateprior[L].stateprob[k*(int)powerC+(int)powerC-1]+stateprior[L].stateprob[k*(int)powerC+(int)powerC-2];
    }
//     printf_vec(priorn_unnormalized, (int)(*Nbmax)+1);
//     mexPrintf("\n");
    tempsum=sum_r(priorn_unnormalized, 0, 1, (*Nbmax)+1);
//     mexPrintf("%f\n",tempsum);
    for (k=0; k<=(int)(*Nbmax); k++)
    {
        priorn[k]=priorn_unnormalized[k]/tempsum;
    }
//     printf_vec(priorn, (int)(*Nbmax)+1);
    //Backward algorithm
    statepost = (stprobclass*)mxCalloc((mwSize)(L+1)*(int)no_ins,(mwSize)sizeof(stprobclass));
    post = (double*)mxCalloc((mwSize)(int)C*(int)no_ins,(mwSize)sizeof(double));
    condp = (double*)mxCalloc((mwSize)(int)C*(int)no_ins,(mwSize)sizeof(double));
    for (i=0; i<(int) C; i++)
    {
      for (j=0; j<(int) (no_ins); j++) 
      {
          post[j*(int)C+i]=0;
          condp[j*(int)C+i]=0;
      }
    }
    // Each Nb
    statepost[0].stateprob = (double*)mxCalloc((mwSize)((int)powerC*((int) no_ins+1)),(mwSize)sizeof(double));
    statepost[0].constmat = (double*)mxCalloc((mwSize)(int)powerC,(mwSize)sizeof(double));

    //initialize state posterior probability
    for (i=0; i<(int)powerC; i++)
    {
        statepost[0].constmat[i] = 0;
        for(j=0; j<=(int)no_ins; j++)
        {
            statepost[0].stateprob[j*(int)powerC + i] = 0;
        }
    }

    statepost[0].col_size = (int) (*Nbmax) + 1;
    for (h=sum_r(str,(int)powerC-2,(int)powerC,(int)C); h<=(int)(*Nbmax); h++)
    {
        statepost[0].stateprob[h*(int)powerC + (int)powerC-1]=1;
        statepost[0].stateprob[h*(int)powerC + (int)powerC-2]=1;
    }
//         printf_mat(statepost[0].stateprob, (int)powerC, (int) no_ins +1);
//         mexPrintf("\n");
    //main
    for (k=0; k<L; k++)
    {
        k1=0;
        while (k1 < (int) no_ins && stateprior[k1*(L+1)+L-1-k].col_size != 0)
        {
//             mexPrintf("%d, %d\n", k+1, k1);
            tmpidx = (k1 % 2 == 0)? k1 + 1 : k1 - 1;
            tmpidx2 = (k1 % 2 == 0)? tmpidx - 1 : tmpidx + 1;
            statepost[(k1)*(L+1) + (k+1)].col_size = stateprior[k1*(L+1)+L-1-k].col_size;
            statepost[(k1)*(L+1) + (k+1)].stateprob = (double*)mxCalloc((mwSize)((int)powerC*statepost[(k1)*(L+1) + (k+1)].col_size),(mwSize)sizeof(double));
            statepost[(k1)*(L+1) + (k+1)].constmat = (double*)mxCalloc((mwSize)(int)powerC,(mwSize)sizeof(double));
            if (stateprior[tmpidx*(L+1)+L-1-k].col_size == 0)
            {
                statepost[(k1)*(L+1) + (k+1)].col_size = statepost[(k1/2)*(L+1) + k].col_size;
                statepost[(k1)*(L+1) + (k+1)].stateprob = statepost[(k1/2)*(L+1) + k].stateprob;
                statepost[(k1)*(L+1) + (k+1)].constmat = statepost[(k1/2)*(L+1) + k].constmat;
//                     mexPrintf("%d, %d\n", k1/2, statepost[(k1/2)*(L+1) + k].col_size);
//                     printf_mat(statepost[(k1/2)*(L+1) + k].stateprob, (int)powerC, statepost[(k1/2)*(L+1) + k].col_size);
//                     mexPrintf("\n");
            } else{
                    size1 = statepost[(k1/2)*(L+1) + k].col_size;
                    size2 = stateprior[tmpidx*(L+1)+L-1-k].col_size;
                    size3 = stateprior[tmpidx2*(L+1)+L-1-k].col_size;
                    statepost[(k1)*(L+1) + (k+1)].col_size = size3;
                    sig1 = (double*)mxCalloc((mwSize)size1,(mwSize)sizeof(double));
                    sig2 = (double*)mxCalloc((mwSize)size2,(mwSize)sizeof(double));
                    v1 = (double*)mxCalloc((mwSize)size3,(mwSize)sizeof(double));
                    v2 = (double*)mxCalloc((mwSize)size3,(mwSize)sizeof(double));
                    for (i=0; i<(int)powerC; i++)
                    {
                        statepost[(k1)*(L+1) + (k+1)].constmat[i] = 0;
                        for(j=0; j<statepost[(k1)*(L+1) + (k+1)].col_size; j++)
                        {
                            statepost[(k1)*(L+1) + (k+1)].stateprob[j*(int)powerC + i] = 0;
                        }
                    }
                    for (i=0; i<(int)powerC; i++)
                    {
                        lastconst=0;
                        for (j=0; j<(int)powerC; j++)
                        {
                            sig1 = project_row(statepost[(k1/2)*(L+1) + k].stateprob, sig1, (int) shift_table[i*(int)powerC+j]-1, (int)powerC, size1);
                            c1 = statepost[(k1/2)*(L+1) + k].constmat[(int) shift_table[i*(int)powerC+j]-1];
                            temp1 = sum_r(sig1, 0, 1, size1);
                            sig2 = project_row(stateprior[tmpidx*(L+1)+L-1-k].stateprob, sig2, j, (int)powerC, size2);
                            //c2 = sum_r(sig2, 0, 1, size2);
                            c2 = stateprior[tmpidx*(L+1)+L-1-k].constmat[j];
                            if(temp1 != 0 && c2 != 0)
                            {
                                lc1 = c1 + c2;
                                for(n=0; n<size3; n++)
                                {
                                    v1[n] = 0; 
                                }
                                v1 = xcorr_limit(sig1, size1, sig2, size2, size3, v1);
                                v2 = project_row(statepost[k1*(L+1)+ (k+1)].stateprob, v2, i, (int)powerC, size3);
                                lc2 = lastconst;
                                if (lc2 == 0)
                                {
                                    lc3 = lc1;
                                    for (n=0; n<size3; n++)
                                    {
                                        statepost[k1*(L+1)+ (k+1)].stateprob[n*(int)powerC+ i]=v1[n]; 
                                    }
                                } else
                                {
                                    if (lc1 >= lc2)
                                    {
                                        lc3 = lc1+log(1+exp(lc2-lc1));
                                        for (n=0; n<size3; n++)
                                        {
                                            statepost[k1*(L+1)+ (k+1)].stateprob[n*(int)powerC+ i]=1/(1+exp(lc2-lc1))*v1[n] + exp(lc2-lc1)/(1+exp(lc2-lc1))*v2[n]; 
                                        }
                                    } else {
                                        lc3 = lc2+log(1+exp(lc1-lc2));
                                        for (n=0; n<size3; n++)
                                        {
                                            statepost[k1*(L+1)+ (k+1)].stateprob[n*(int)powerC+ i]=exp(lc1-lc2)/(1+exp(lc1-lc2))*v1[n] + 1/(1+exp(lc1-lc2))*v2[n]; 
                                        }
                                    }
                                }
                                lastconst = lc3;
//                                    mexPrintf("%d, %d, %d, %d\n", k, k1,i, j);
//                                    printf_vec(sig1,size1);
//                                      mexPrintf("\n");
//                                    printf_vec(sig2,size2);
//                                      mexPrintf("\n");
//                                    sig3=xcorr(sig1, size1, sig2, size2);
//                                    printf_vec(sig3,size3);
//                                      mexPrintf("\n");
//                                    for (n=0; n<size3; n++)
//                                    {
//                                        statepost[k1*(L+1)+ (k+1)].stateprob[n*(int)powerC+ i]=statepost[k1*(L+1)+ (k+1)].stateprob[n*(int)powerC+ i] + sig3[n]; 
//                                    }
                            }
                        }
                        statepost[(k1)*(L+1) + (k+1)].constmat[i] = lastconst;
                    }  
                    mxFree(sig1);
                    mxFree(sig2);
                    mxFree(v1);
                    mxFree(v2);
                  }
//                       printf_mat(statepost[k1*(L+1)+ (k+1)].stateprob, (int)powerC, statepost[(k1)*(L+1) + (k+1)].col_size);
//                       mexPrintf("\n");
                  k1++;
              }
        }
    //record posterior probability from state zero
    for (n=0; n<(int) no_ins; n++)
    {
        for (i=0; i <(int)powerC; i++)
        {
            a=sum_r(str,i,(int)powerC,(int)C);
            if((int)a==1)
            {
                j=find_arr(str,i,(int)powerC,(int)C,1);
                if(j==(int)C-1)
                {
                    condp[n*(int)C+j]=statepost[n*(L+1)+L].stateprob[i];
                }else {
                    condp[n*(int)C+j]=statepost[n*(L+1)+L].stateprob[(int)powerC+i];
                }
                post[n*(int)C+j]=condp[n*(int)C+j]*prior[n*(int)C+j];
            }
        }
    }
//     printf_mat(condp,(int)C,(int)no_ins);
//     mexPrintf("\n");
    for (j=0; j<(int)no_ins; j++)
    {
        post=normalize_mat(post, C, no_ins, j);
    }
//     printf_mat(post,(int)C,(int)no_ins);
    
    plhs[0] = mxCreateDoubleMatrix((mwSize)C,(mwSize)no_ins,mxREAL);
    plhs[1] =  mxCreateDoubleScalar(tempsum);
    outMatrix = mxGetPr(plhs[0]);
//       
//     outMatrix = project_col(arr,(*allsample)-1,(int)powerC, (*allsample));
    memcpy(outMatrix, post, (int)C*(int)no_ins*sizeof(double));
//     printf_mat(outMatrix,(int)C,(int)no_ins);
    }