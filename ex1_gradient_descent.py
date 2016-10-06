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
plt.xlabel('population')
plt.ylabel('profit')
plt.title('regression in one varibel \n by use of batch gradient descent ')
#plt.legend()
#plt.show()
#######



###start implementing batch gradient descent. OLS in one variable
def least_squares(df, theta0=0, theta1=0, alpha=0.01, iterations=1500):
    x=df.population
    y=df.profit
    m=len(x)
    #luckily x and y behaves as vectors, so summing becomes easy
    cost=(1/(2*m))*sum((theta0 +theta1*x-y)**2)
    cost_list=[]


    for i in range(0, iterations):
        theta0=theta0-alpha*sum(theta0+theta1*x-y)/m
        theta1=theta1-alpha*sum((theta0+theta1*x-y)*x)/m
        cost=(1/(2*m))*sum((theta0 +theta1*x-y)**2)
        cost_list.append(cost)
    return theta0,theta1



theta0, theta1=least_squares(df)

x=range(0, 25)
y=[]
for i in x:
    y.append(theta0+theta1*i)

plt.plot(x,y, label='regression')
plt.legend()
plt.show()
