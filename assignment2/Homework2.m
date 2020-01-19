%Decision making with Business Analytics | Homework 2 
%Linear programming & Sensitivity analysis
%Dafni Lappa & Epameinondas-Spyridon Themelis


%QUESTION 1
%for the purposes of this question we consider a linear model in the normal
%form, i.e. a maximization problem with <= constraints, and all variables 
%being non negative.

%In order to visualize the results of this program the red dog toy example
%is used.

%The initial form of the problem is:
%(Version I)
%max 2x1+x2
%s.t 3x1+2x2 <=80
%    3x1+x2<= 50
%    x1+2x2<=60
%    x1,x2 >= 0

%However, in this file user should transform the problem in a form that the
%problem is a minimization one, and all the slack variables are also
%shown in the respected constraints.

%Hence, user should provide the data coming from a linear model of the 
%following form, where the objective function is still to be maximized but
%all constraint inequalities have been transformed to equality using slack
%variables, which are also shown as x(). Thus x3,x4,x5 are slack variables.
%The non-negativity restriction remains.

%Version(II - initial form)
%max 2x1+x2
%s.t 3x1+2x2 + x3 = 80
%    3x1+x2      +x4 = 50
%    x1+2x2          +x5= 60
%    x1,x2,x3,x4,x5 >= 0

%in this problem there are m=5 variables and n=3 constraints plus the 
%constraint requiring non negative variables

%Provide the required data in matrices

%Aeq is a n*m matrix whose Aij element is the coefficient of variable j in
%constraint equality i. Of course, each slack variable has a unit coefficient
%only in the constraint that it was used to transform it into equality.
Aeq=[3 2 1 0 0
    3 1 0 1 0
    1 2 0 0 1];

%beq is a vector of size n, listing the right hand side of all constraints.
%But, the non-negativity constraint should not be included in the beq vector.
beq=[80 50 60]';

%lb is a zero vector of size m. This vector is generally used to form the
%non negativity constraint. 
lb=zeros(5,1);

%f is the vector of size 1*m including the coefficients of the objective
%function. Slack variables are included in the objective functions well as 
%in this vector with zero coefficient. As we chose cplex as our solver the
%objective function should demonstrate the minimization problem, and this is the
%explains the negative sign in the initial coefficients. 
f=[-2 -1 0 0 0 ]';


%Cplexlp is used to solve this problem 
[y,fval]=cplexlp(f,[],[],Aeq,beq,lb,[],[]);


%Displaying the value of the objective function after solving the linear
%program and the corresponding values of each variable
disp('the value of the objective function of the problem is:')
disp(-fval)
disp('the value of each variables of the problem (slack variables included) is:')
disp(y)




%Define variabes based on the solution provided.

%based on the lenght of y, the total number of variables can be verified
%slack variables are also included
[sol, numColumns]=size(y);
%sol is the total number of variables
sol;

%%Calculate the number of basic and non-basic variables 

%By checking the optimal value of each variable we can determine whether each 
%variable is a basci or a non basic one. Variables with final-optimal value 
%equal to zero are non basic variables while variables with value greater
%than 0 are considered to be basic.

cal_nb=0;
cal_b=0;  
for i=1:sol
    if y(i,1)==0
        cal_nb=cal_nb+1; 
    else
        cal_b=cal_b+1; 
    end
    
end
%    disp('number of basic variables in the problem')
%    disp(cal_b) 


%Before we calculated the number of basic and non basic variables of the
%problem, while now we cal identify the location each basic and non basic
%variable. This is basically a step in order to help in future
%calculations.

%%identify the location of basic variables
   loc_b=find (y ~= 0); %==0
%    disp('basic variables are in the following positions of y vector')
%    disp(loc_b) 

% %identify the location of non-basic variables
   loc_nb=find (y == 0); %~0
%    disp('non basic variables are in the following positions of y vector')
%    disp(loc_nb)
  

%%Print basic and non-basic variables
disp('The basic variables of the linear problem are:')
for i=1:cal_b
   X_bv(i,:) = ['x' num2str(loc_b(i))];
end
disp(X_bv)

disp('The non-basic variables of the linear problem (including the slack variables) are:')
for i=1:cal_nb
   X_nbv(i,:) = ['x' num2str(loc_nb(i))] ;
end
disp(X_nbv);
 

%Creating the vectors C_bv and C_nbv, which list the coefficients of the
%basic and non basic variables (according to the optimal tableau) in the 
%initial objective function respectively. 

%c_bv and c_nbv [row vector of the initial objective coefficients for the optimal tableau 's basic variable]
loc_nbtr= loc_nb';
supnb=size(loc_nbtr,2);
loc_btr= loc_b';
supb=size(loc_btr,2);

for i=1:supb
    h=loc_b(i);
    c_bv(1,i)=f(h); 
end
    c_bv=-(c_bv);


for i=1:supnb
    h=loc_nb(i);
    c_nbv(i)=f(h);
end
    c_nbv= -(c_nbv);
    
%  disp('The coefficients of the basic variables in the objective function are:')
%  disp(c_bv)
%  disp('The coefficients of the non basic variables in the objective function are:')
%  disp(c_nbv)
 
%Define matrices B and N.
%BV is a n*n matrix whose j column is the column for the constraints of basic 
%variable j in the initial tableau. 
%NBV matrix is n*n-basic whose j column is the constraints column for the 
%non basic variables in the initial form.
  
 BV = Aeq(:,loc_b);
 NBV = Aeq(:,loc_nb);
 
  
  %################################################################################################%
  %Question 1a
  
  %%Optimality check
  
  %In order to check whether the obtained solution is optimal we need to
  %check if the following criteria hold for this problem.
  
  % -each constraint has a non negative right hand side in the optimal tableau
  %AND
  % -each variable in row 0 has a nonnegative coefficient 
  
  
  %%Identify the optimal tableau
  %computing the constraints' coefficients for the optimal tableau, Xj
 
  %calculate the number of constraints in the Linear problem
  constraints_number=length(beq);   
  %calculate the inverse table of BV
  Binv=inv(BV);
  %Binv_reorder=Binv([3,1,2],:)

  %define matrix X. Table X=B^(-1)*aj, where aj is the column vector of
  %table Aeq for variable j. Each element of X(a,b) is the constraints 
  %coefficinet for the b-th variable in the a-th constraint in the optimal
  %tableau.
  X=zeros(constraints_number,sol);
  X = Binv*Aeq;                                                               %[variables' order as in Binv]
 
 
 %disp('The constraint coefficients for the optimal tableau for each variable x:')
 %disp(X)
 %X(a,b) is the matrix indicating the 
 
 
 %Compute the right hand side of optimal's tableau constraints|RHS=B^(-1)*b
 RHS=Binv*beq;                                                                  %[x1,x2,x3]
 
 
 %Compute coefficients of slack variables in row 0 in the optimal tableau
 c_slack=c_bv*Binv;
  
 %Compute RHS of optimal row 0
 rhs_0=c_bv*Binv*beq;
 
 %Define the c - coefficinets of all variables based on the initial LP
 %this step is required because the minimization coefficients were provided as
 %input to the solver
for i=1:sol
    c(i)=-f(i);
end

%compute reduced costs, C_bar using the formula C_bar(j)=C*B^(-1)*aj-cj
for i=1:sol
    c_bar(i,1)=c_bv*Binv*Aeq(:,i)-c(i);
end


%checking if optimality holds

%setting two optimality flags
optimal=1;
optimal_cb=1;
 
%For every constraint in the optimal tableau we check the first criterion
%mentioned at the begining of the section. If a negative right hand side
%for even one constraint set optimality flag=0

 for i=1:constraints_number
     if RHS(i) < 0 
         optimal=0 ;
     end
 end
 
 %For each variable in row 0 of the optimal tableau check if it has a 
 %negative coefficient. If a negative coefficient is found, set optimality
 %flag=0. 
 %DISCLAIMER: MATLAB changes the layout of the inverse table. Thus,
  %variables may appear to be in different column order that in the original
  %BV table. Also, in order to avoid very small values that are practically 
  %zero but MATLAB does not set them equal to zero 
  %(for example [-2.77555756156289e-17]) we manually used a threshold setting 
  %all values less than 1e-3, be equal to zero. By this way, optimality is not
  %affected. 
 
 for i=1:sol
     
     if c_bar(loc_b) ~= 0
         
         for i=1:constraints_number
            for j=1:constraints_number
                if Binv(i,j)> 1e-3
                     optimal_cb =0;
                end
            end
         end
     elseif c_bar(loc_nb) < 0 & c_bar> 1e-3
                 optimal_cb=0;
     end
 end
 

 %Get the optimality response
if optimal_cb == 1 && optimal==1
     disp('Solution is optimal')
end

%##########################################################################################%
%Question 1b
%Sensitivity analysis
%Change constraints' rhs - what shoud e be in order to mantain the optimal
%basis of question 1a

%For this part user has to manually define the constraint which right hand
%side he wishes to change. Specifically, variable c_change is used for this
%reason. In the reddog example, if user select c_change=1; this means that 
%the constraint 1,3x1+2x2+x3=80 will now become 3x1+2x2+x3=80+e. In this
%part of the homework we seek to identify e so that the optimal base found 
%in question 1a does not change.

%please choose the constraint which right hand side changes
c_change=3;

%symbolic maths are used in order to solve the problem containing e variable
b_zeros=sym(zeros(constraints_number,1));
syms e

%formulating the vector zero vector in a way that includes e variable in
%the right position
b_zeros(c_change,1)=e;
b_sens=sym(b_zeros);

%defining the new vector listing the right hand side of constraints in the 
%initial LP, including e in the chosen position. For example if constraint 
%1 is chosen to be affected, b_eqsens=[80+e;50;60]
b_eqsens=beq+b_sens;


%formulating the inequalities to be solved using the existing matrices
new_bas= RHS + Binv*b_sens;

%initializing the vectors that will include all solutions
positives=zeros(constraints_number,1);
negatives=zeros(constraints_number,1);

%initialize large hypotheticaly values for solving convenience
%values can be adjusted based on the problem's data
for i=1:constraints_number
    
 negatives(i)=-8000000;
 positives(i)=8000000;
 
end
 
%solving the inequalities using symbolic maths
%the loop soolves consequently each inequality resulting from the linear
%operations in the new_bas.
for i=1:constraints_number                      %apo edo xekinaei h diadikasia gia kathe anisothta

    ineq(1,i)=solve(new_bas(i)>=0,e);
    
    if ineq(1,i)<0
        negatives(i)=ineq(i);
    else
        positives(i)=ineq(i);
    end
end

%find the maximum negative and minimum positive value in order to use them 
%as range extreme values
    max_negatives=max(negatives);
    min_positives=min(positives);
    
%adding the extreme values in the initial right hand side of the changed constraint    
    rhs_upper_range=beq(c_change)+min_positives;
    rhs_lower_range=beq(c_change)+max_negatives;
    
%including the case where increasing the right hand side of the constraint 
%does not affect the optimal basis    
    if rhs_upper_range==0
        rhs_upper_range= Inf
    end
    
%displaying the results of sensitivity analysis regarding the right hand side of a constraint   
    print=['The range for the right hand side of constraint ', num2str(c_change), '  is:']
    disp(rhs_lower_range)
    disp(rhs_upper_range)
 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Question 1c
%Sensitivity analysis
%Change coefficients of variables in the objective function - what shoud d 
%be in order to mantain the optimal basis of question 1a


%Change coefficient of a non basic variable
%This is because if xj is a nonbasic variable and we let
%objective-function coefficient cj be changed by an amount 1cj with all
%other data held fixed, then the current solution remains unchanged so long 
%as the new reduced cost c_barj remains nonnegative.


%For this part user has to manually define the variable whose coefficient 
%wishes to change. Specifically, variable pivot_nv is used for this reason.
%DISCLAIMER: when choosing the non basic variable which coefficient is changed
%always refer to the position of the variable in the loc_nb vector, i.e the
%vector listing the non basic variables. For example in the reddog case, if 
%user wishes to change the coefficient of slack variable s2, that is x4 should
%insert pivot_nv=1, which is the position of x4 in the vector listing the non
%basic variables loc_nb

coefficients_non_basic_initial=c(loc_nb);
%please choose the non basic variable whose coefficient is changed
pivot_nv=1;

%use symbolic maths to perform operations concerning the variable k, that
%is the amount that a coefficient of a non basic variable is changed
c_pivot=sym(zeros(1,1));
syms k

%formulating the vector zero vector in a way that includes k variable in
%the right position
c_pivot=sym(k);
%add the vector including the parameter k to the vector listing the initial 
%objective coefficient 
new_c_nbv= sym (coefficients_non_basic_initial(pivot_nv)+c_pivot);
X_pivots=X(:,loc_nb);
X_nbvpivot=X_pivots(:,pivot_nv);

%formulate the inequality to be solved>=0
non_basic_inequality= sym(c_bv*X_nbvpivot-new_c_nbv);

%solve the inequality
non_basic_value_sens=solve(non_basic_inequality>=0,k);
%calculate the deviation from the original value of the non basic variable
actual_nbv_range=(coefficients_non_basic_initial(pivot_nv)+non_basic_value_sens);


%if coefficient was initially zero (slack variable) and the deviation is
%changing the coefficient will not make a difference
if coefficients_non_basic_initial(pivot_nv)==0
    
    if actual_nbv_range <0  
    disp('Coefficient should remain 0')
    else 
% deviation is positive, the variable can be included in the objective function 
%without resulting to changes in the optimal tableau      
    print=['Optimal tableau is not changed as long as coefficient for the variable x' num2str(loc_nb(pivot_nv))', ...
        ' is less than ']
    disp(actual_nbv_range)
    end
else
    print=['Optimal tableau is not changed as long as coefficient for the variable x' num2str(loc_nb(pivot_nv))', ...
        ' is less than ']
    disp(actual_nbv_range)
end

%Change the coefficient of a basic variable
%For this part user has to manually define the variable whose coefficient 
%wishes to change. Specifically, variable c_cob is used for this
%reason. In the reddog example, if user select c_ob=1; this means that 
%the coefficient of variable x1 in the objective function will now become 
% (2+d)x1+x2+0x3+0x4+0x5. In this part of the homework we seek to identify 
%d so that the optimal base found in question 1a does not change.
%Change coefficient of a basic variable
%please choose the basic variable which coefficient changes
c_ob=2;   

%symbolic maths are used in order to solve the problem containing d variable
syms d

%initialize the objective's coefficient in order to address the
%real maximization problem and not the minimization used of cplex
initial_obj_coef= -f(loc_b).';
coef_obj=sym(zeros(1,cal_b));
coef_obj(1,c_ob)=d; 
%formulating the c_bnew vector listing the new coefficients of the basic
%variables. In the reddog example, assuming that user seeks to find the range for 
%coefficient of x1, i.e. c_ob=1, the c_bnew vector is c_bnew=[(2+d),1,0]
c_bnew=sym(c_bv+coef_obj);
 

%initialize the vectors that will list the solutions provided after solving
%the inequalities. Values can be adjusted based on each problem's data if
%required
for i=1:constraints_number
    
 negative_solutions(i)=-8000000;
 positive_solutions(i)=8000000;
 
end

%formulating the problem to be solved using linear algebra based on the existing
%tables. Once again, symbolic maths are used to solve this problem.tremo1
%is a vector of size 1*(number of non basic variables)
 tremo1=sym(c_bnew*Binv*Aeq(:,loc_nb))-c(loc_nb);
 
 
%solving the inequalities using symbolic maths
%the loop soolves consequently each inequality resulting from the linear
%operations in the tremo1.
 for j=1:cal_nb
     
     d_ineq(j) = solve(tremo1(j)>=0,d) ;
     
     if d_ineq(j)<0
         negative_solutions(j,1)=d_ineq(j);
     else
         positive_solutions(j,1)=d_ineq(j);
     end
 end
 
 %find the maximum negative and minimum positive values resluted from
 %solving the inequalities in order to use them as pivot elements during
 %the identification of the coefficient's range
 max_negative_solutions=max(negative_solutions);
 min_positive_solutions=min(positive_solutions);
 

 %Defining the values for the lower and upper bound
 %All cases are considered even if not realistic???.
 if max_negative_solutions == -8000000
     for i=1:cal_nb
         sorted_positive_solutions=sort(positive_solutions);
     end
     coef_lower_bound=min(sorted_positive_solutions);
     coef_upper_bound=sorted_positive_solutions(2);
     
 elseif min_positive_solutions == 8000000
     coef_lower_bound=max_negative_solutions;
     coef_upper_bound=Inf;
 else
     coef_lower_bound=max_negative_solutions;
     coef_upper_bound=min_positive_solutions;
 end
 


%Calculate the actual values that the selected coefficient can take, by
%adding the extreme values identified before in the existing coefficient value 
%of the named variable
coef_upper_value =(c_bv(c_ob)+ coef_upper_bound);
coef_lower_value =(c_bv(c_ob)+ coef_lower_bound);


%printing out the results of sensitivity analysis regarding the coefficient
%change in one of the basic variables
print=['Coefficient for the chosen variable x', num2str(c_ob), ' can take values from: ',num2str(coef_lower_value),...
    ' to ', num2str(coef_upper_value) ]
disp(print)
disp(coef_lower_value)
disp(coef_upper_value)

 

