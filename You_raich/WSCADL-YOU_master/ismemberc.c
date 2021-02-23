#include "mex.h"
#include <math.h>
#include <stdlib.h>

double * project_row(double *arr, int row, int no_of_rows, int no_of_cols)
{
    double *result= (double*)mxCalloc((mwSize)no_of_cols,(mwSize)sizeof(double));
    
    int i;
    
    for (i=0; i<no_of_cols; i++)
    {
        result[i]=arr[i*no_of_rows+row];
    }
    return result;
}

double check_equal(double *arr1, double *arr2, int no_of_cols)
{
    double ok;
    int i;
    ok=1;
    for (i=0; i<no_of_cols; i++)
    {
        if(arr1[i]!=arr2[i])
        {
            ok=0;
            break;
        }
        
    }
    return ok;    
}


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *matrix_inp;
    double *arr1;
    double *arr2;
    double *result;
    double ok;
    int i;
    double index_val;
    size_t no_of_rows;     
    size_t no_of_cols;
    double *outvalue;  
  
    matrix_inp = mxGetPr(prhs[0]);
    arr2 = mxGetPr(prhs[1]); 
    
    no_of_rows = mxGetM(prhs[0]);
    no_of_cols = mxGetN(prhs[0]);
    
    for (i=0; i<no_of_rows;i++)
    {
    arr1=project_row(matrix_inp, i, (int)no_of_rows, (int)no_of_cols);
    ok=check_equal(arr1, arr2, no_of_cols);
        if(ok==1)
        {
            break;
        }
    }
    
    if(ok!=0)
    {
        index_val=(double)(i+1);
    }
    else
    {
        index_val=-1;
    }
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    outvalue = mxGetPr(plhs[0]);
    memcpy(outvalue,&index_val, 1*sizeof(double));
    }
    
     
    
