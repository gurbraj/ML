##start with plotting the data to get an idea.
#modify the datafile to name the columns
#plotting is easy with pandas!
#use more separate functions

import pandas as pd
import matplotlib.pyplot as plt
import math as math
import numpy as np

df=pd.read_csv('ex2data1.txt')

####PLOT#################
not_admitted_df=df[df.y==0]
admitted_df=df[df.y==1]

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

def sigmoid(x):
    return 1/(1+math.exp(-x))

    #bonus: vectorized
    #sigmoid_vectorized=np.vectorize(sigmoid)
###there is a little problem with above, it does not have asymptotic behavior! math.exp(-1000)==0 is true according to python. this throws an error
#later when it is used. so i will use the built in one from scipy

#from scipy.special import expit as sigmoid




#1.2.2 implement the cost function and gradient for logistic regression.
def cost_and_gradient(X,Y,theta_array):
    cost_list=[]
    common_term_for_derivatives_list=[]
    m=len(Y)
    for i in range(0,len(Y)):
        h_theta=sigmoid(np.dot(theta_array,X[i]))

        if h_theta==1:
            print(h_theta)
        else:
            term=(-Y[i]*math.log(h_theta)-(1-Y[i])*math.log(1-h_theta))/m
            cost_list.append(term)


        #do some work on the gradient
        common_term_for_derivatives=(h_theta-Y[i])/m
        common_term_for_derivatives_list.append(common_term_for_derivatives)
        #

    cost=sum(cost_list)

    #gradient
    gradient=[]
    #datastructure change so that we can use dot prodcut
    common_term_for_derivatives_array=np.array(common_term_for_derivatives_list)
    for j in range(0, len(theta_array)):
        derivative=np.dot(common_term_for_derivatives_array,X[:,j])
        gradient.append(derivative)

    gradient=np.array(gradient)

    return cost, gradient

#cost,gradient=cost_and_gradient(X,Y,theta_array)
#initial check works out

#one can use fmin from scipy import optimize
#but i want to create a general gradient descent.
#so we need to have the costfunction convex.
#this is the case and it can easily be shown analytically.

#but in the gradient_descent function i have two whileloops just to make it stop
#incase the costfunction is not convex => no convergence

def gradient_descent(X,Y,theta_array,alpha=0.01,counter_limit=1500,treshold=0.01):

    counter=0
    #just initialize cost to something
    cost=0
    theta_list=[]
    while counter < counter_limit:

        counter=counter+1
        cost_old=cost
        cost,gradient=cost_and_gradient(X,Y,theta_array)
        print(cost)

        #while abs(cost-cost_old)>treshold:
        for i in range(0,len(theta_array)):

            #theta_array[i]=theta_array[i]-alpha*gradient[i]
            #above updating rule do not work for some odd reason. updating array problem?
            theta=theta_array[i]-alpha*gradient[i]
            theta_list.append(theta)


        theta_array=np.array(theta_list)
        theta_list=[]



    return theta_array


df=pd.read_csv('ex2data1.txt')
X,Y, theta_array=df_to_matrix(df)
gradient_descent_theta_array=gradient_descent(X,Y,theta_array)
