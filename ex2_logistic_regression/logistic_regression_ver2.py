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
#regularize by setting l=0 in the argument
#Note that you should not regularize the parameter theta_array[0]!! i,e use theta_array[1:]
def cost_function(theta_array,X,Y, l=0):
    cost_list=[]

    m=len(Y)
    for i in range(0,len(Y)):
        h_theta=sigmoid(np.dot(theta_array,X[i]))
        term=(-Y[i]*np.log(h_theta)-(1-Y[i])*np.log(1-h_theta))/m
        cost_list.append(term)


    cost=sum(cost_list)+(l/(2*m))+(l/(2*m))*np.dot(theta_array[1:].T,theta_array[1:])

    return float(cost)

#second approach for computing the cost function. they both work. (though i did not extend cost_function2 to include lambda)

def cost_function2(theta_array, X,Y):
    m=len(Y)
    h_x=sigmoid(np.dot(X,theta_array.T))

    term1=np.dot(np.log(h_x),-Y.T)
    term2=np.dot(np.log(1-h_x),(1-Y).T)

    return (term1-term2)/m

#Note that you should not regularize the parameter theta_array[0]!! hence the if-else clausul
def gradient_function(theta_array,X,Y, l=0):

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
        #Note that you should not regularize the parameter theta_array[0]!! hence the if-else clausul below
        if j==0:
            derivative=np.dot(common_term_for_derivatives_array,X[:,j])
        else:
            derivative=np.dot(common_term_for_derivatives_array,X[:,j]) +(l/m)*theta_array[j]

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

def optimal_thetas(theta_array,X,Y,iterations=400):
    result = optimize.fmin(cost_function, x0=theta_array, args=(X, Y), maxiter=iterations, full_output=True)
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


##2 Regularized logistic regression
#In this part of the exercise, you will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assur- ance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly.
#Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. From these two tests, you would like to determine whether the microchips should be accepted or rejected.

#add column_names on the datafile

microchip_test_df=pd.read_csv('ex2data2.txt')

def classification_plot(df, title='plot of the training data'):
    column_names=df.columns.tolist()
    y_name=column_names[-1]

    X_sth_y0=df[df[y_name]==0]
    X_sth_y1=df[df[y_name]==1]

    plt.scatter(X_sth_y1[column_names[0]], X_sth_y1[column_names[1]], label="label 1", color="blue")
    plt.scatter(X_sth_y0[column_names[0]], X_sth_y0[column_names[1]], label="label 0", color="red")
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[1])
    plt.title(title)
    plt.legend()
    plt.show()

classification_plot(microchip_test_df)


#2.2 Feature mapping
#One way to fit the data better is to create more features from each data point.
X_df_copy=microchip_test_df.copy()
X_df_copy.columns=['x1','x2', 'y']
series1=X_df_copy['x1']
series2=X_df_copy['x2']

def mapfeature_binominal(series1,series2,n=6):

    #a bit hacky admitedely, but here i though about puting for when the code when series1 and series2 are arrays
    # consider the principle of SOLID, Open to change. perhaps i should have done this function more general. but on the other hand this was a challenging function to make:make it work, and then optimize it.

    if n==1:
        return
    else:
        for i in range(0,n+1):
            #name=str((n-i)) + " 1's * " + str(i)+ " 2's"
            name='x1**'+str((n-i)) +"*"+ "x2**"+str(i)
            formula=((series1)**(n-i))*((series2)**(i))

            X_df_copy[name]=formula

        mapfeature_binominal(series1,series2,n-1)


mapfeature_binominal(series1,series2)
#^^this works, but does not give the same order as those in the instrucionts.
#note that im not returning anything in the case of series, but only modiyfing
X_reg, Y_reg, theta_array=df_to_matrix(X_df_copy)


#You should see that the cost is about 0.693. (again)
#cost_function(theta_array,X_reg,Y_reg) works.

#now time to optimize the thetas.

theta_array_opti_reg, minimal_cost_reg=optimal_thetas(theta_array,X_reg,Y_reg,iterations=100000)
#lool needad alot of iterations! lesson: regulators cost alot in terms of complexity

#in order to do the actual prediction i need to modify the mapfeature_binominal function, so that it takes (x1,x2)-> array of their binominial combinations
#this is neccesarr in order for sigmoid(np.dot(theta_array_opti_reg,[x1,x2])) to make sense.

def mapfeature_binominal_array(x1,x2,n=6,lst=None):

    #a bit hacky admitedely, but here i though about puting for when the code when series1 and series2 are arrays
    # consider the principle of SOLID, Open to change. perhaps i should have done this function more general. but on the other hand this was a challenging function to make:make it work, and then optimize it.
    if len(lst)==None:
        #dont forget the x0=1
        lst.append(1)
        lst.append(x1)
        lst.append(x2)

    if n==1:
        print(np.array(lst))
        return np.array(lst)
    else:
        for i in range(0,n+1):
            formula=((x1)**(n-i))*((x2)**(i))
            lst.append(formula)


        return mapfeature_binominal_array(x1,x2,n-1,lst)

test_array=mapfeature_binominal_array(-0.25,1.5)

test_array=np.array([1,-0.25, 1.5, 0.000244140625, -0.00146484375, 0.0087890625, -0.052734375, 0.31640625, -1.8984375, 11.390625, -0.0009765625, 0.005859375, -0.03515625, 0.2109375, -1.265625, 7.59375, 0.00390625, -0.0234375, 0.140625, -0.84375, 5.0625, -0.015625, 0.09375, -0.5625, 3.375, 0.0625, -0.375, 2.25])

sigmoid(np.dot(theta_array_opti_reg.T,test_array))==1 #as the pdf.




#time to plot the decisionboundary
#remains to do that, and to plt with different lambdas. will do some other day!
