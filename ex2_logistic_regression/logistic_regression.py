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
sigmoid_vectorized=np.vectorize(sigmoid)

#1.2.2 implement the cost function and gradient for logistic regression.
X,Y, theta_array=df_to_matrix(df)
def cost(X,Y,theta_array):
    list=[]
    m=len(Y)
    for i in range(0,len(Y)):
        h_theta=sigmoid(np.dot(theta_array,X[i]))

        term=-Y[i]*math.log(h_theta)-(1-Y[i])*math.log(1-h_theta)
        list.append(term)
    return sum(list)/m

#initial check works out
