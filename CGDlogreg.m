%%************************************************************************
%% December 21, 2007
%%
%% This is a Matlab implementation of a coordinate gradient descent method
%% for solving the L_1 regularized logistic regression problem
%%
%%  min_{w,v}  l_{avg}(w,v)(=:(1/m)*sum_{i=1}^m f(w^Ta_i+vb_i)) 
%%            + lambda*||w||_1 ,
%% where f(z)=log(1+exp(-z)) and a_i=b_i*x_i.
%%
%% The method, for a given (w^T,v)^T, computes a direction d 
%% as the solution of the subproblem
%%
%%   min    g'*d + d'*H*d/2 + lambda*norm(w+d,1)   
%%   s.t.   d(j)=0 for j not in J,
%% where g = (\nabla_w l_{avg}(w,v)^T,\nabla_v l_{avg}(w,v))^T, 
%% H is a positive definite diagonal matrix,
%% and J is a nonempty subset of {1,...,n,n+1}.  
%% Then (w^T,v)^T is updated by (w^T,v)^T = (w^T,v)^T + step*d,
%% and this is repeated until a termination criterion is met.
%% The index subset J is chosen by a Gauss-southwell type rule,
%% and step is chosen by an Armijo stepsize rule.
%%
%%===== Required inputs =============
%%
%%  X: mxn matrix (where n is the dimension of explanatory or feature
%%     variables and m the number of observed or training examples).
%%  b: a vector in R^m and its each component is -1 or 1
%%  lambda: usually, a non-negative real parameter of the objective
%%       function (see above).
%%
%%===== Optional inputs =============
%%
%%  'StopCriterion' = type of stopping criterion to use
%%                    0 = stop when the relative
%%                       change in the objective function
%%                       falls below 'Tol'
%%                    1 = stop when max_j |d_j(x;N)|
%%                       falls below 'Tol'
%%                    2 = stop when the objective function
%%                        becomes equal or less than Tol.
%%                    Default = 1.
%%
%%  'ChooseH' = choose a matrix H for directional subproblem
%%                    0 = H is the identity matrix.
%%                    1 = positive approximation of Hessian diagonal
%%                    2 = tau*I where tau is a constant
%%                    Default = 1.
%%
%%  'Tol' = stopping threshold; Default = 0.001
%%
%%  'Maxiter' = maximum number of iterations allowed in the algorithm.
%%              Default = 10000
%%
%%  'Initialization' must be one of {0,1,array}
%%               0 -> Initialization at zero.
%%               1 -> initialization provided by the user.
%%               Default = 0;
%%
%%  'Verbose'  = work silently (0) or verbosely (1)
%%
%%===== Outputs =============
%%
%%   [w^T;v] = solution of the main algorithm
%%   objective = sequence of values of the objective function
%%   times = CPU time after each iteration
%%************************************************************************

   function [w,v,objective,times] = CGD_logreg(X,b,lambda,par)

   if (nargin < 3)
      error('Need at least 3 inputs');
   end
   dir = 'sd';
   stopCriterion = 1;
   chooseH = 1;
   tol     = 1e-6;
   maxiter = 100000;
   init    = 0;
   verbose = 1;
   randnstate = randn('state');
   randn('state',0);
   if isfield(par,'Direction'); dir = par.Direction; end
   if isfield(par,'StopCriterion'); stopCriterion = par.StopCriterion; end
   if isfield(par,'ChooseH'); chooseH = par.ChooseH; end
   if isfield(par,'Tolerance'); tol = par.Tolerance; end
   if isfield(par,'Maxiter'); maxiter = par.Maxiter; end
   if isfield(par,'init'); init = par.init; end
   if isfield(par,'w0'); w0 = par.w0; end
   if isfield(par,'Verbose'); verbose = par.Verbose; end
%%
%% Initialization
%%
   [m,n] = size(X); 
   switch init
      case 0   % initialize at zero, using AT to find the size of x
        w = zeros(n,1);
        v = 0;
      case 1   % initial [w;v] was given by user
        w = w0(1:end-1);
        v = w0(end);
      otherwise
        error(['Unknown ''Initialization'' option']);
   end
   A = spdiags(b,0,length(b),length(b))*X;
   At = A'; 
   %%  
   %% scale b 
   %%
   colnormA = mexdiaghess(A,ones(m,1)); 
   vscale = sqrt(mean(sqrt(colnormA))/norm(b));
   b = vscale*b;
%%
%% Parameters for Armijo stepsize rule
%%
   sigma = 0.1; 
   beta  = 0.5;
%%
%% initial stepsize for coord. gradient iterations
%% initial threshold for choosing J.
%%
   pexp = exp(-(A*sparse(w)+v*b));
   plog = 1+pexp;
   l1norm = norm(w,1);
   f = sum(log(plog))/m + lambda*l1norm;
   t0 = cputime;
   times(1) = 0;
   objective(1) = f;
   step = 1;
   maxd = 1;
   if strcmp(dir,'sq')
      ups = 0.90;
   elseif strcmp(dir,'sd')
      ups = 0.90; 
   end
%%
%% Compute the diagonal of Hessian
%%
  if (chooseH == 0)
     hw = 0.001*ones(n,1); 
     hv = 0.001;  
  elseif (chooseH == 2)
     xrand = rand(n,1); xrand = xrand/norm(xrand);
     Axrand = A*xrand;
     tau = norm(Axrand)^2/(8*m);
     hw  = tau*ones(n,1);
     plogexp = pexp./(plog.*plog);
     hv = min(max((b'*(plogexp.*b))/m,1e-10),1e10);
  end
  if (verbose)
     fprintf('\n iter  objective    step   ups  nnzd   maxd');
     fprintf('\n %3.0f  %10.6e  %3.2f  %3.2f',...
     0.0,f,step,ups);
  end
%%
  for iter = 1:maxiter  
     plogexp = pexp./(plog.*plog);
     if ((chooseH == 1) & (mod(iter,1)==0 | iter==1)) 
        hw = (1/m)*mexdiaghess(A,plogexp); 
        hv = (1/m)*(b'*(plogexp.*b));
        hw = min(max(hw,1e-10),1e10);
        hv = min(max(hv,1e-10),1e10);
     elseif (chooseH == 2)
        hv = (1/m)*(b'*(plogexp.*b));
        hv = min(max(hv,1e-10),1e10);
     end
     %% compute gradient
     gw = At*(pexp./plog); gw = (-1/m)*gw; 
     gv = b'*(pexp./plog); gv = (-1/m)*gv;  
  
     %% compute cgd direction
     dv = -gv/hv; 
     if strcmp(dir,'sd')
        [maxdw,dw,nnzd] = dirsd(lambda,w,gw,hw,ups);
     elseif strcmp(dir,'sq')
        [maxdw,dw,nnzd] = dirsq(lambda,w,gw,hw,ups);
     end
     %%
     %% Check for termination
     maxd = max(maxdw,abs(gv)); 
     if (stopCriterion == 1)
        if (maxd <= tol & iter > 10); break; end
     end
     %%
     %% Armijo stepsize rule
     %%
     dirderiv = gw'*dw+gv*dv+lambda*(norm(w+dw,1)-l1norm); 
     dz   = A*sparse(dw)+dv*b;
     step = min(step/beta^5,1);
     for k = 1:20
        sw = step*dw;
        sv = step*dv;
        pexp_new = pexp.*exp(-step*dz);
        plog_new = 1+pexp_new;
        l1norm_new = norm(w+sw,1);
        fnew = sum(log(plog_new))/m + lambda*l1norm_new;
        if (fnew-f < sigma*step*dirderiv) 
           break; 
        end
        step = step*beta; 
        if (step<1e-4)
           fprintf('stepsize too small! Check roundoff error in fnew-f.\n');
           break;
        end
     end
     fold = f;
     f = fnew;  
     w = w + sw;
     v = v + sv;
     pexp = pexp_new;
     plog = plog_new;
     l1norm = l1norm_new;
     %%
     %% Update the threshold for choosing J
     opt = 1;
     if (opt==1) %% better 
        if (rem(iter,20)==0 | iter < 10); ups = max(0.05,0.95*ups); end
     else
        if (step > 0.5) 
           ups = max(0.05,0.95*ups);
        elseif (step <= 0.01)
           ups = min(0.5,5*ups);
           step = 0.01;
           fprintf('**');
        end
     end
%%
     objective(iter+1) = f;
     times(iter+1) = cputime-t0;
     if (verbose)
        fprintf('\n %3.0d  %10.6e  %3.2f  %3.2f %5.0d  %3.2e',...
        iter,f,step,ups,length(nnzd),maxd);
     end      
     if (stopCriterion == 0)
        criterionObjective = abs(f-fold)/(1+abs(fold));
        if (criterionObjective < tol); break; end
     elseif (stopCriterion == 2)
        if (f < tol); break; end
     end
  end 
  v = vscale*v; 
%%
  if (verbose)
     fprintf(1,'\nFinished the main algorithm!\nResults:\n')
     fprintf(1,'Objective function = %10.3e\n',f);
     fprintf(1,'CPU time so far = %10.3e\n', times(end));
     fprintf(1,'max residual = %3.2e\n',maxd);
     fprintf(1,'\n');
  end
%%************************************************************************
