#i need to use the logistic function(http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.expit.html) but the python version does not have have the asymptotic property that the analytic version has. expit(10)==1 is true, which becomes a problem when i use it in log(1-expit(10)), since log(0) is not defined. anyone have a clue how to mitigate this?
#hence version 2

##start with plotting the data to get an idea.
#modify the datafile to name the columns
#plotting is easy with pandas!
#use more separate functions

import pandas as pd
import matplotlib.pyplot as plt
import math as math
import numpy as np
from scipy import optimize
df=pd.read_csv('ex2data1.txt')

####PLOT#################
not_admitted_df=df[df.y==0]
admitted_df=df[df.y==1]
column_names=df.columns.tolist()
plt.scatter(admitted_df[column_names[0]], admitted_df[column_names[1]], label="admitted", color="blue")
plt.scatter(not_admitted_df[column_names[0]], not_admitted_df[column_names[1]], label="not admitted", color="red")
plt.xlabel(column_names[0])
plt.ylabel(column_names[1])
plt.title('visualization of the dataset we \n are going to use for logistic regression')
plt.legend()
plt.show()
##################END PLOT

#map dataframe to X-matrix and Y-vector,    initialize theta array, fill X with 1 intercepts
def df_to_matrix(df,Y_name='y', theta_array=[]):
    column_names=df.columns.tolist()
    column_names.remove(Y_name)

    X_df=df[column_names]
    X_df['intercept']=1
    X_column_names=['intercept']+ column_names
    X=X_df[X_column_names].as_matrix()

    Y=df[Y_name].as_matrix()

    if len(theta_array)==0:
        for i in range(0,len(X_column_names)):
            theta_array.append(0)


    return X, Y, np.array(theta_array)




#1.2.1 implement sigmoid function

#def sigmoid_R(x):
    #return 1/(1+math.exp(-x))

from scipy.special import expit as sigmoid
    #bonus: vectorized
#sigmoid=np.vectorize(sigmoid_R)

###there is a little problem with above, it does not have asymptotic behavior! math.exp(-1000)==0 is true according to python. this throws an error
#later when it is used. so i will use the built in one from scipy
#math.log=np.vectorize(math.log)


#1.2.2 implement the cost function and gradient for logistic regression.
def cost_function(theta_array,X,Y):
    cost_list=[]

    m=len(Y)
    for i in range(0,len(Y)):
        h_theta=sigmoid(np.dot(theta_array,X[i]))
        term=(-Y[i]*np.log(h_theta)-(1-Y[i])*np.log(1-h_theta))/m
        cost_list.append(term)


    cost=sum(cost_list)

    return float(cost)

#second approach for computing the cost function. they both work.
def cost_function2(theta_array, X,Y):
    m=len(Y)
    h_x=sigmoid(np.dot(X,theta_array.T))

    term1=np.dot(np.log(h_x),-Y.T)
    term2=np.dot(np.log(1-h_x),(1-Y).T)

    return (term1-term2)/m

def gradient_function(theta_array,X,Y):

    common_term_for_derivatives_list=[]
    m=len(Y)
    for i in range(0,len(Y)):
        h_theta=sigmoid(np.dot(theta_array,X[i]))

        #do some work on the gradient
        common_term_for_derivatives=(h_theta-Y[i])/m
        common_term_for_derivatives_list.append(common_term_for_derivatives)
        #

    #gradient
    gradient=[]
    #datastructure change so that we can use dot prodcut
    common_term_for_derivatives_array=np.array(common_term_for_derivatives_list)
    for j in range(0, len(theta_array)):
        derivative=np.dot(common_term_for_derivatives_array,X[:,j])
        gradient.append(derivative)

    gradient=np.array(gradient)

    return gradient

#cost,gradient=cost_and_gradient(X,Y,theta_array)
#initial check works out

#one can use fmin from scipy import optimize
#but i want to create a general gradient descent.
#so we need to have the costfunction convex.
#this is the case and it can easily be shown analytically.

#but in the gradient_descent function i have two whileloops just to make it stop
#incase the costfunction is not convex => no convergence

def optimal_thetas(theta_array,X,Y):
    result = optimize.fmin(cost_function, x0=theta_array, args=(X, Y), maxiter=400, full_output=True)
    return result[0], result[1]


df=pd.read_csv('ex2data1.txt')
X,Y, theta_array=df_to_matrix(df)

theta_array_opti, minimal_cost=optimal_thetas(theta_array,X,Y)

##it works!
#Optimization terminated successfully.
#         Current function value: 0.203498
#         Iterations: 157
#         Function evaluations: 287

#as expected

#another test For a student with an Exam 1 score of 45 and an Exam 2 score of 85, you should expect to see an admission probability of 0.776.
sigmoid(np.dot(theta_array,np.array([1,45,85])))
#outputs 0.77629159041124107, booya.

#plot the decisionboundary###
def DB_function_outer(theta_array_opti):

    def DB_function_inner(x):
        return (-theta_array_opti[0]-theta_array_opti[1]*x)/theta_array_opti[2]

    return DB_function_inner

DB_function_inner=DB_function_outer(theta_array_opti)
x_range=list(range(0,100))
y_range=list(map(DB_function_inner, x_range))

not_admitted_df=df[df.y==0]
admitted_df=df[df.y==1]
plt.plot(x_range,y_range, label="decisionboundary")
plt.scatter(admitted_df[column_names[0]], admitted_df[column_names[1]], label="admitted", color="blue")
plt.scatter(not_admitted_df[column_names[0]], not_admitted_df[column_names[1]], label="not admitted", color="red")
plt.xlabel(column_names[0])
plt.ylabel(column_names[1])
plt.title('visualization of the dataset we \n are going to use for logistic regression')
#plt.legend()
#plt.show()
