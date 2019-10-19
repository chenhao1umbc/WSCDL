#include "mex.h"
#include <math.h>
#include <stdlib.h>

void printf_complex_vec(double *real, double *image, int n)
 {
     int i;
     for (i=0;i <n; ++i)
     {
         mexPrintf("%f + i %f\n",real[i], image[i]);
         mexEvalString("drawnow");
     }    
 }

void printf_vec(double *arr_double_ind, int n)
 {
     int i;
     for (i=0;i <n; ++i)
     {
         mexPrintf("%f\n",arr_double_ind[i]);
         mexEvalString("drawnow");
     }    
 }

double * project_row(double *arr, double *result, int row, int b, int no_of_rows, int no_of_cols)
{
    int i;
    
    for (i=0; i<no_of_cols; i++)
    {
        result[i]=arr[row+i*no_of_rows+b*(no_of_rows*no_of_cols)];
    }
    return result;
}

double * project_row_seg(double *arr, double *result, int f, int c, int k, int size, int m, int n)
{
    int startidx = f*size;
    int endidx = (f+1)*size;
    int i=0,j;    
    for (j=startidx; j<endidx; j++)
    {
        result[i]=arr[j+c*m+k*m*n];
        i++;
    }
    return result;
}

double * convolution(double *arr1, int size1, double *arr2, int size2, double *result)
{
    int length=0;
    int n;
    int kmin;
    int kmax;     
    int k;
    length=size1+size2-1;
        
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

void FFT(short int dir,long m,double *x,double *y)
{
   long n,i,i1,j,k,i2,l,l1,l2;
   double c1,c2,tx,ty,t1,t2,u1,u2,z;

   n = 1;
   for (i=0;i<m;i++) 
      n *= 2;

   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
      if (i < j) {
         tx = x[i];
         ty = y[i];
         x[i] = x[j];
         y[i] = y[j];
         x[j] = tx;
         y[j] = ty;
      }
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
   }

   c1 = -1.0; 
   c2 = 0.0;
   l2 = 1;
   for (l=0;l<m;l++) {
      l1 = l2;
      l2 <<= 1;
      u1 = 1.0; 
      u2 = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {
            i1 = i + l1;
            t1 = u1 * x[i1] - u2 * y[i1];
            t2 = u1 * y[i1] + u2 * x[i1];
            x[i1] = x[i] - t1; 
            y[i1] = y[i] - t2;
            x[i] += t1;
            y[i] += t2;
         }
         z =  u1 * c1 - u2 * c2;
         u2 = u1 * c2 + u2 * c1;
         u1 = z;
      }
      c2 = sqrt((1.0 - c1) / 2.0);
      if (dir == 1) 
         c2 = -c2;
      c1 = sqrt((1.0 + c1) / 2.0);
   }

   if (dir == -1) {
      for (i=0;i<n;i++) {
         x[i] /= n;
         y[i] /= n;
      }
   }
   
}


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *arr1;
    double *arr2;
    double *sig1, *sig2, *tempsig, *result;
    double *fftrealsig1, *fftrealsig2, *ifftrealsig3;
    double *fftimgsig1, *fftimgsig2, *ifftimgsig3;
    int i,f,c,k,b, length;
    int bool_addone;
    int bool_conv;
    int NFFT;
    int nDimNum, nDimNum2, nDimNumX;
    const int *pDims;
    const int *pDimsX;
    int *pDims2;
    mxArray *Data;
    size_t size1;     
    size_t size2;
    size_t size3;
    size_t F,C,K,B;
    double *outarray;  
  
    arr1 = mxGetPr(prhs[0]);
    arr2 = mxGetPr(prhs[1]); 
    bool_addone = mxGetScalar(prhs[2]);
    bool_conv = mxGetScalar(prhs[3]);
    nDimNumX = mxGetNumberOfDimensions(prhs[0]);
    nDimNum = mxGetNumberOfDimensions(prhs[1]);
    pDimsX = mxGetDimensions(prhs[0]);
    F = pDimsX[0];
    size1 = pDimsX[1];
    B = (nDimNumX==2)? 1: pDimsX[2];
    pDims = mxGetDimensions(prhs[1]);
    if (bool_addone==1)
    {
        size2=(pDims[0]-1)/F;
    }else {
        size2=pDims[0]/F;
    }
    size3 = size1 + size2 - 1;
    NFFT = (int)pow(2.0, ceil(log((double)size3)/log(2.0)));
    C = pDims[1];   
    K = (nDimNum==2)? 1: pDims[2];
    nDimNum2 = (nDimNum==2)? nDimNum+1: nDimNum;
    pDims2 = (int*)mxCalloc((mwSize) nDimNum2,(mwSize)sizeof(int));
    pDims2[0]=size3;
    pDims2[1]=C*K;
    pDims2[2]=B;
    Data = mxCreateNumericArray(nDimNum2, pDims2, mxDOUBLE_CLASS, mxREAL);
    result = (double *) mxGetPr(Data);
    for (b=0; b<B; b++) {
        for (k=0; k<K; k++){
            for (c=0; c<C; c++){
                for (i=0; i<size3; i++){
                    result[i+(C*k+c)*size3+b*size3*(C*K)]=0;
                }
            }
        }
    }
    
    sig1 = (double*)mxCalloc((mwSize)size1,(mwSize)sizeof(double));
    sig2 = (double*)mxCalloc((mwSize)size2,(mwSize)sizeof(double));
    fftrealsig1 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    fftimgsig1 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    fftrealsig2 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    fftimgsig2 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    ifftrealsig3 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    ifftimgsig3 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    tempsig = (double*)mxCalloc((mwSize)size3,(mwSize)sizeof(double));
    for (b=0; b<B; b++) {
        for (k=0; k<K; k++){
            for (c=0; c<C; c++){
                for (f=0; f<F; f++){
                    for (i=0; i<size1; i++){
                        sig1[i]=0;
                    }
                    sig1 = project_row(arr1, sig1, f, b, F, size1);
                    for (i=0; i<size2; i++){
                        sig2[i]=0;
                    }                          
                    sig2 = project_row_seg(arr2, sig2, f, c, k, size2, pDims[0], C);
                    if (bool_conv) {
                        for (i=0; i<size3; i++)
                        {
                            tempsig[i] = 0;
                        }
                        tempsig = convolution(sig1, size1, sig2, size2, tempsig);
                    } else {   
                    
                        for (i=0; i<NFFT; i++) {
                            fftrealsig1[i]=0;
                            fftimgsig1[i]=0;
                            fftrealsig2[i]=0;
                            fftimgsig2[i]=0;
                        }

                        for (i=0; i<size1; i++) {
                            fftrealsig1[i]=sig1[i];
                        }

                        FFT(1,(int)(log((double)NFFT)/log(2.0)),fftrealsig1, fftimgsig1);

                        for (i=0; i<size2; i++) {
                            fftrealsig2[i]=sig2[i];
                        }

                        FFT(1,(int)(log((double)NFFT)/log(2.0)),fftrealsig2, fftimgsig2);

                        for (i=0; i<NFFT; i++) {

                            ifftrealsig3[i] = fftrealsig1[i] * fftrealsig2[i] - fftimgsig1[i] * fftimgsig2[i];
                            ifftimgsig3[i] = fftrealsig1[i] * fftimgsig2[i] + fftrealsig2[i]* fftimgsig1[i] ;
                        }
                        FFT(-1,(int)(log((double)NFFT)/log(2.0)),ifftrealsig3, ifftimgsig3);
                        for (i=0; i<size3; i++)
                        {
                            tempsig[i] = ifftrealsig3[i];
                        }
                    }

                    for (i=0; i<size3; i++){
                        if(bool_addone==1)
                        {
                            result[i+(C*k+c)*size3+b*size3*(C*K)] = result[i+(C*k+c)*size3+b*size3*(C*K)] + tempsig[i] + arr2[pDims[0]-1+c*pDims[0]+k*pDims[0]*C]/F;
                        }else {
                            result[i+(C*k+c)*size3+b*size3*(C*K)] = result[i+(C*k+c)*size3+b*size3*(C*K)] + tempsig[i];
                        }
                    }
                }
            }
        }
    }
    mxFree(sig1);
    mxFree(sig2);
    mxFree(tempsig);
    mxFree(fftrealsig1);
    mxFree(fftrealsig2);
    mxFree(ifftrealsig3);
    mxFree(fftimgsig1);
    mxFree(fftimgsig2);
    mxFree(ifftimgsig3);
    length = (int) size3*C*K*B;
    plhs[0] = mxCreateNumericArray(nDimNum2, pDims2, mxDOUBLE_CLASS, mxREAL);
    outarray = (double *)mxGetPr(plhs[0]);
    memcpy(outarray,result, length*sizeof(double));
    
    }
    
     
    
