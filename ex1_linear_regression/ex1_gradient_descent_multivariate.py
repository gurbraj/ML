import pandas as pd
import numpy as np


df1=pd.read_csv('ex1data2.txt')


#do i modify df1 here? should i care?
def df_to_matrix(df,Y_name='y', theta_array=None):
    column_names=df.columns.tolist()
    column_names.remove(Y_name)
    X_df=df[column_names]
    X_df['intercept']=1
    X_column_names=['intercept']+ column_names
    X=X_df[X_column_names].as_matrix()

    Y=df[Y_name].as_matrix()

    if theta_array==None:
        theta_array=[]
        for i in range(0,len(X_column_names)):
            theta_array.append(0)


    return X, Y, np.array(theta_array)

X,Y,theta_array=df_to_matrix(df1)

def feature_normalization(X):
    #here i just want to modify X, instead of returning a new X. i reason like this: if I modify X i can easily get it back by running df_to_matrix(df1). naturally it does have a cost associated with it but i think it is small?
    row_length=len(X[0:1][0])

    for i in range(0, row_length):
        if not X[:,i].std()==0:
            X[:,i]=(X[:,i]-X[:,i].mean())/X[:,i].std()

feature_normalization(X)




def cost_function_lin_reg(theta_array, X,Y):
    m=len(Y)



    cost=sum((np.dot(X,theta_array)-Y)**2)/(m*2)

    return cost




#Note that you should not regularize the parameter theta_array[0]!! hence the if-else clausul. i use regularization for classification problems, so ignore it for now.
def gradient_function(theta_array,X,Y, l=0):

    common_term_for_derivatives_list=[]
    m=len(Y)
    for i in range(0,len(Y)):
        h_theta=np.dot(theta_array,X[i])

        #do some work on the gradient
        common_term_for_derivatives=(h_theta-Y[i])/m
        common_term_for_derivatives_list.append(common_term_for_derivatives)
        #

    #gradient
    gradient=[]
    #datastructure change so that we can use dot prodcut
    common_term_for_derivatives_array=np.array(common_term_for_derivatives_list)
    for j in range(0, len(theta_array)):
        #Note that you should not regularize the parameter theta_array[0]!! hence the if-else clausul below
        if j==0:
            derivative=np.dot(common_term_for_derivatives_array,X[:,j])
        else:
            derivative=np.dot(common_term_for_derivatives_array,X[:,j]) +(l/m)*theta_array[j]

        gradient.append(derivative)

    gradient=np.array(gradient)

    return gradient


    def gradient_descent(theta_array, X, Y,treshold=10):

        cost=cost_function_lin_reg(theta_array,X,Y)
        counter=0
        cost_old=treshold+1
        cost_list=[]
        while abs(cost-cost_old)>treshold and counter <10000:
            counter=counter+1
            cost_list.append(cost)
            cost_old=cost

            theta_array=gradient_function(theta_array,X,Y)
            cost=cost_function_lin_reg(theta_array, X, Y)

        print('it took {counter} iterations'.format(counter=counter))

        return theta_array, cost, cost_list

#one can use fmin from scipy import optimize
#but i want to create a general gradient descent.
#so we need to have the costfunction convex.
#this is the case and it can easily be shown analytically.

#but in the gradient_descent function i have two whileloops just to make it stop
#incase the costfunction is not convex => no convergence

def optimal_thetas(theta_array,X,Y,iterations=400):
    result = optimize.fmin(cost_function, x0=theta_array, args=(X, Y), maxiter=iterations, full_output=True)
    return result[0], result[1]
