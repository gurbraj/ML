#Before starting on any task, it is often useful to understand the data by visualizing it, two ways shown  below.

##importing data and plotting it (without pandas)
#import matplotlib.pyplot as plt
#import csv

#x=[]
#y=[]

#with open('ex1data1.txt','r') as csvfile:
#    plots=csv.reader(csvfile, delimiter=',')
#    for row in plots:
#        x.append(float(row[0]))
#        y.append(float(row[1]))

# plt.scatter(x,y, label='Loaded from file!')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('imported withoud pandas \nCheck it out')
# plt.legend()
#plt.show()
########

##import data and plot it (using pandas)
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('ex1data1.txt')

plt.scatter(df['population'],df['profit'], label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')
plt.title('regression in one varibel \n by use of batch gradient descent ')
#plt.legend()
#plt.show()
#######



###start implementing batch gradient descent. OLS in one variable
def regression(df, theta0=0, theta1=0, alpha=0.01, iterations=1500):
    x=df.population
    y=df.profit
    m=len(x)
    #luckily x and y behaves as vectors, so summing becomes easy
    cost=(1/(2*m))*sum((theta0 +theta1*x-y)**2)
    cost_list=[]
    theta0_list=[]
    theta1_list=[]


    for i in range(0, iterations):
        theta0=theta0-alpha*sum(theta0+theta1*x-y)/m
        theta1=theta1-alpha*sum((theta0+theta1*x-y)*x)/m
        cost=(1/(2*m))*sum((theta0 +theta1*x-y)**2)
        cost_list.append(cost)
        theta0_list.append(theta0)
        theta1_list.append(theta1)
    return theta0,theta1, range(0, iterations), cost_list, theta0_list, theta1_list



theta0, theta1, iterations_array, cost_list, theta0_list, theta1_list=regression(df)

x=range(0, 25)
y=[]
for i in x:
    y.append(theta0+theta1*i)

#regression plot
plt.plot(x,y, label='regression')
plt.legend()
plt.show()
#
# costfunction against iterations plot
plt.plot(iterations_array, cost_list, label='cost function against iterations')
plt.title("the cost function is monotonically decreasing with iterations")
plt.xlabel('iterations')
plt.ylabel('cost function')
plt.legend()
plt.show()
#
#3d plot (shows tha descent that the algoritm has taken) - does not show the entire surface

from mpl_toolkits.mplot3d import axes3d
from matplotlib import style

style.use('ggplot')
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_wireframe(theta0_list, theta1_list, cost_list)
ax1.set_xlabel('theta0')
ax1.set_ylabel('theta1')
ax1.set_zlabel('costfunction')

plt.show()


#####now linear regression with multiple variables (change datafiles to name the colomns.)
#first we are going to feature normalize by subtracting mean and dividing by std dev
import pandas as pd
df1=pd.read_csv('ex1data2.txt')

size_series=(df1['size']-df1['size'].mean())/df1['size'].std()
bedrooms_series=(df1['#bedrooms']-df1['#bedrooms'].mean())/df1['#bedrooms'].std()
price_series=(df1['price']-df1['price'].mean())/df1['price'].std()

df1['size']=size_series
df1['#bedrooms']=bedrooms_series
df1['price']=price_series
##had a lot of trouble using concat or to simply just populate a new dataframe
#below is general, including the mean normalization
def multivariate_regression(df, alpha=0.01, iterations=1500,Y_name='price'):
    df1=df

    column_names=df1.columns.tolist()
    #mean normalization
    for i in range(0,len(column_names)):
        column_name=column_names[i]
        df1[column_name]=(df1[column_name]-df1[column_name].mean())/df1[column_name].std()

    #filll X with intercept with 1's so that we can smoothtly use matrix algebra
    df1['intercept']=1
    column_names=['intercept']+ column_names
    #split the data into a X dataFrame and a y series
    column_names.remove(Y_name)
    X_df=df[column_names]
    Y_series=df[Y_name]


    m=len(df['intercept'])
    #initialize thetas
    theta_list=[]
    for i in column_names:
        theta_list.append(0)

    theta_series=pd.Series(theta_list)

    #as_matrix has to be done on the series becuse othervise the .dot method does not work!!!!

    common_vector=(X_df.dot(theta_series.as_matrix())-Y_series)/m
    cost=sum((common_vector**2)/2)
    derivatives=[]
    for column_name in column_names:
        derivative=common_vector.dot(X_df[column_name].as_matrix())
        derivatives.append(derivative)

    #so inital derivatives calculated
    #todo: compute inital cost function       , updating rule, stopping condition


    for i in iterations:
        for j in range(0, len(theta_series)):
            theta_series[j]=theta_series[j]-alpha*derivatives[j]

    return theta_series

        ##tired of multivariable case. look over in future.
