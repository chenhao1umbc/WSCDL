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

double sum_mat(double *arr, int n_row, int n_col)
{
    int i,j;
    double sum_=0;
    for (i=0; i<n_row; i++)
    {
        for (j=0; j<n_col; j++)
        {       
            sum_=sum_+arr[j* n_row + i];
        }
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

double * project_col(double *arr, double *result, int col, int no_of_rows, int no_of_cols)
{
    int i;
    
    for (i=0; i<no_of_rows; i++)
    {
        result[i]=arr[col*no_of_rows+i];
    }
    return result;
}

double * project_col_seg(double *arr, double *result, int startcol, int endcol, int rowid, int no_of_rows)
{
    int i=0;
    int j;    
    for (j=startcol; j<endcol; j++)
    {
        result[i]=arr[j*no_of_rows+rowid];
        i++;
    }
    return result;
}

double *normalize_prob(double *arr_double_ind, int m, int n, int col_id)
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

double *normalize_mat(double *arr_double_ind, int m, int n)
 {
    double sum_=0;
    int i,j;
    for (i=0; i<m; i++)
    {   
        for (j=0; j<n; j++)
        {
            sum_=sum_+arr_double_ind[m*j+i];
        }
    }
    for (i=0; i<m; i++)
    {   
        for (j=0; j<n; j++)
        {
            arr_double_ind[m*j+i]=arr_double_ind[m*j+i]/sum_;
        }
    }
    return arr_double_ind;
    
 }


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    
    typedef struct{
        double * stateprob;
    } stprobclass;
    
   
    double *Nbmax;
    double *prior;
    double *str;
    double *shift_table;
    double *shift_table_add;

    stprobclass *stateprior, *statepost;
    double *priorn_unnormalized;
    double *priorn;
    double *condp, *postp;

    int class_count;
    int *class_list;
    size_t powerC;     
    size_t C;
    size_t no_ins;

    double *outMatrix;
    double *post;

    int i, j, k, h, a, n;
    double l,tempsum,logconst;
    double *sig1, *sig2;
    int size, size1, size2;
    
    Nbmax = mxGetPr(prhs[0]);
    prior = mxGetPr(prhs[1]);
    str = mxGetPr(prhs[2]);
    shift_table_add = mxGetPr(prhs[3]);
    
    powerC = mxGetM(prhs[2]);
    C = mxGetM(prhs[1]);
    no_ins = mxGetN(prhs[1]);

    stateprior = (stprobclass*)mxCalloc((mwSize)(int)no_ins,(mwSize)sizeof(stprobclass));
   
    stateprior[0].stateprob = (double*)mxCalloc((mwSize)((int)powerC*2),(mwSize)sizeof(double));

    for (i=0; i<(int)powerC; i++)
    {
        for (j=0; j<2; j++)
        {
            stateprior[0].stateprob[j*(int)powerC+i]=0;
        }
    } 

    logconst=0;
    stateprior[0].stateprob[0]=prior[(int)C-1];
    for (i=0; i <(int)powerC; i++)
    {
        a=sum_r(str,i,(int)powerC,(int)C-1);
        if((int)a==1)
        {
            j=find_arr(str,i,(int)powerC,(int)C-1,1);

            stateprior[0].stateprob[1*(int)powerC+i] = prior[j];
        }
    }
    logconst=logconst+log(sum_mat(stateprior[0].stateprob, (int)powerC, 2));
    stateprior[0].stateprob=normalize_mat(stateprior[0].stateprob,(int)powerC, 2);

    
    class_count=sum_r(str,powerC-1,powerC,C-1);
    class_list=find_arr_list(str, powerC-1, powerC, C-1, class_count, 1);
    
    for (k=1; k<(int)no_ins; k++)
    {
        size = (k<=(int)(*Nbmax)-1)? k+2:(int)(*Nbmax)+1;
        stateprior[k].stateprob = (double*)mxCalloc((mwSize)((int)powerC*size),(mwSize)sizeof(double));
   
        for (i=0; i<(int)powerC; i++)
        {
            for (j=0; j<size; j++)
            {
                stateprior[k].stateprob[j*(int)powerC+i]=0;
            }
        }
        size1= (k<(int)(*Nbmax))?size-1:size;
        for (i=0; i<(int)powerC; i++)
        {
            for (h=0;h<size1;h++)
            {
                stateprior[k].stateprob[h*(int)powerC+i]=stateprior[k].stateprob[h*(int)powerC+i] + stateprior[k-1].stateprob[h*(int)powerC+i]*prior[k*(int)C+(int)C-1];
            }

            for (j=0; j<class_count; j++)
            {
                l=shift_table_add[class_list[j]*(int)powerC+i];      
                if(l!=-1)
                {
                    l=l-1;

                       for (h=1;h<size;h++)
                       {
                           stateprior[k].stateprob[h*(int)powerC+(int)l]=stateprior[k].stateprob[h*(int)powerC+(int)l] + stateprior[k-1].stateprob[(h-1)*(int)powerC+i]*prior[k*(int)C+(class_list[j])];
                       }  
                }
            } 
        }
        logconst=logconst+log(sum_mat(stateprior[k].stateprob, (int)powerC, size));
        stateprior[k].stateprob=normalize_mat(stateprior[k].stateprob,(int)powerC, size);

    }
    

    priorn_unnormalized = (double*)mxCalloc((mwSize)(int)(*Nbmax)+1,(mwSize)sizeof(double));
    priorn = (double*)mxCalloc((mwSize)(int)(*Nbmax)+1,(mwSize)sizeof(double));
    
    for (k=0; k<=(int)(*Nbmax); k++)
    {
        priorn_unnormalized[k] = stateprior[(int)no_ins-1].stateprob[k*(int)powerC+(int)powerC-1];
    }

    tempsum=log(sum_r(priorn_unnormalized, 0, 1, (int)(*Nbmax)+1))+logconst;
    for (k=0; k<=(int)(*Nbmax); k++)
    {
        priorn[k]=priorn_unnormalized[k]/tempsum;
    }

    
    statepost = (stprobclass*)mxCalloc((mwSize)(int)no_ins,(mwSize)sizeof(stprobclass));
    post = (double*)mxCalloc((mwSize)(int)C*(int)no_ins,(mwSize)sizeof(double));
    condp = (double*)mxCalloc((mwSize)(int)C*(int)no_ins,(mwSize)sizeof(double));

    for (i=0; i<(int) C; i++)
    {
      for (j=0; j<(int) no_ins; j++) 
      {
          post[j*(int)C+i]=0;
          condp[j*(int)C+i]=0;
      }
    }

    statepost[(int)no_ins-1].stateprob=(double*)mxCalloc((mwSize)(int)powerC*((int)(*Nbmax)+1),(mwSize)sizeof(double));
    for (i=0; i <(int)powerC; i++)
    {
        for (j=0; j<=(int)(*Nbmax);j++)
        {
            statepost[(int)no_ins-1].stateprob[j*(int)powerC+i]=0;
        }
    }

    for (h=class_count; h<=(int)(*Nbmax); h++)
    {
        statepost[(int)no_ins-1].stateprob[h*(int)powerC+(int)powerC-1]=1;
    }
    statepost[(int)no_ins-1].stateprob=normalize_mat(statepost[(int)no_ins-1].stateprob, (int)powerC, (int)(*Nbmax)+1);

    for (k=(int)no_ins-2; k>=0; k--)
    {
        size=(k<(int)(*Nbmax))? k+2:(int)(*Nbmax)+1;
        size2=(k>=(int)(*Nbmax)-1)? size-1:size;

        statepost[k].stateprob = (double*)mxCalloc((mwSize)((int)powerC*size),(mwSize)sizeof(double));
        
        for (i=0; i<(int)powerC; i++)
        {
            for (j=0; j<size; j++)
            {
                statepost[k].stateprob[j*(int)powerC+i]=0;
            }
        }
        for (i=0; i<(int)powerC; i++)
        {
            for(h=0; h<size; h++)
            {
              statepost[k].stateprob[h*(int)powerC+i]=statepost[k].stateprob[h*(int)powerC+i]+statepost[k+1].stateprob[h*(int)powerC+i]*prior[(k+1)*(int)C+(int)C-1];  
          
              condp[(k+1)*(int)C+(int)C-1]=condp[(k+1)*(int)C+(int)C-1]+statepost[k+1].stateprob[h*(int)powerC+i]*stateprior[k].stateprob[h*(int)powerC+i];
            }
            for (j=0; j<class_count; j++) 
            {
                l=shift_table_add[class_list[j]*(int)powerC+i];
                if(l!=-1)
                {
                    l=l-1;

                    for(h=0; h<size2; h++)
                    {
                      statepost[k].stateprob[h*(int)powerC+i]=statepost[k].stateprob[h*(int)powerC+i]+statepost[k+1].stateprob[(h+1)*(int)powerC+(int)l]*prior[(k+1)*(int)C+(class_list[j])];  
            
                      condp[(k+1)*(int)C+(class_list[j])]=condp[(k+1)*(int)C+(class_list[j])]+statepost[k+1].stateprob[(h+1)*(int)powerC+(int)l]*stateprior[k].stateprob[h*(int)powerC+i];
                    } 
                }
            }               
        }
        statepost[k].stateprob=normalize_mat(statepost[k].stateprob, (int)powerC, size);


        for (j=0; j<class_count; j++)
        {
            post[(k+1)*(int)C+(class_list[j])]=condp[(k+1)*(int)C+(class_list[j])]*prior[(k+1)*(int)C+(class_list[j])];
        }
        post[(k+1)*(int)C+(int)C-1]=condp[(k+1)*(int)C+(int)C-1]*prior[(k+1)*(int)C+(int)C-1];
    }
 
    condp[(int)C-1]=statepost[0].stateprob[0];
    post[(int)C-1]=condp[(int)C-1]*prior[(int)C-1];
    for (i=0; i <(int)powerC; i++)
    {
        a=sum_r(str,i,(int)powerC,(int)C-1);
        if((int)a==1)
        {
            j=find_arr(str,i,(int)powerC,(int)C-1,1);
            condp[j]=statepost[0].stateprob[(int)powerC+i];
            post[j]=condp[j]*prior[j];
        }
    }

    for (j=0; j<(int)no_ins; j++)
    {
        post=normalize_prob(post, C, no_ins, j);
    }
    
    plhs[0] = mxCreateDoubleMatrix((mwSize)C,(mwSize)no_ins,mxREAL);
    plhs[1] =  mxCreateDoubleScalar(tempsum);
    outMatrix = mxGetPr(plhs[0]);
   
    memcpy(outMatrix, post, (int)C*(int)no_ins*sizeof(double));

}