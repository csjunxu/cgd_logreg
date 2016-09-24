/************************************************************************
 mexdiaghess: compute 
 norm(y.*X(:,j).*X(:,j))^2 for j = 1:n

 diagH = mexdiaghess(X,y)
************************************************************************/

#include "mex.h"
#include <math.h>
#include <matrix.h>

#if !defined(MX_API_VER) || ( MX_API_VER < 0x07030000 )
typedef int mwIndex;
typedef int mwSize;
#endif

/********************************************************************
  PROCEDURE mexFunction - Entry for Matlab
*********************************************************************/
void mexFunction(const int nlhs, mxArray *plhs[],
                 const int nrhs, const mxArray *prhs[])
{
  double   *X, *y, *ytmp, *diagH;
  mwIndex  *irX, *jcX, *iry, *jcy; 
  int       m, n, jm, isspX, isspy, j, jn, k, kstart, kend, r; 
  double    tmp;

  if(nrhs != 2)
    mexErrMsgTxt("mexdiaghess: requires 2 input arguments.");
  if(nlhs > 1)
    mexErrMsgTxt("mexdiaghess: requires 1 output argument.");

  X = mxGetPr(prhs[0]);
  isspX = mxIsSparse(prhs[0]); 
  if (isspX) {
     irX = mxGetIr(prhs[0]);
     jcX = mxGetJc(prhs[0]);
  }
  m = mxGetM(prhs[0]); 
  n = mxGetN(prhs[0]); 
  isspy = mxIsSparse(prhs[1]);
  y = mxGetPr(prhs[1]);  
  if (isspy) {
     iry = mxGetIr(prhs[1]);
     jcy = mxGetJc(prhs[1]);
  }
  if (mxGetM(prhs[1]) != m) {
     mexErrMsgTxt("mexdiaghess: size of 2nd input not compatiable.");
  }
  ytmp = mxCalloc(m,sizeof(double)); 
  if (isspy) { 
     for (k=0; k<jcy[1]; ++k) { r=iry[k]; ytmp[r]=y[k]; }  
  } else {
     for (k=0; k<m; ++k) { ytmp[k]=y[k]; }  
  }
  plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);      
  diagH = mxGetPr(plhs[0]);  
  /********************************************************/

     if (isspX) {
        for (j=0; j<n; j++) {
           kstart = jcX[j]; kend = jcX[j+1]; 
           tmp = 0; 
           for (k=kstart; k<kend; k++) { 
	      r = irX[k];
              tmp += ytmp[r]*X[k]*X[k]; 
	   }
           diagH[j] = tmp; 
	}
     } else { 
        for (j=0; j<n; j++) {
           tmp = 0; 
           jm = j*m; 
           for (k=0; k<m; k++) { 
              tmp += ytmp[k]*X[jm+k]*X[jm+k]; 
	   }
           diagH[j] = tmp;         
	}
     }
  mxFree(ytmp); 
return;
}
/************************************************************************/
