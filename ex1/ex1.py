import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# A = np.random.rand(4,2)
# B = np.random.rand(2,1)
# C = A.dot(B)

# print(B)
# print(A.shape)
# print(B.shape)
# print(C.shape)
#5x5 Identity Matrix
#A = np.eye(5)
#print(A)

#读取数据
df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])

#print(df.head())

#Plotting Data ...
sns.set(context="notebook", style="whitegrid", palette="dark")
sns.lmplot('population', 'profit', df, height=6, fit_reg=False)
#plt.show()


#=================== Part 3: Cost and Gradient descent ===================
#ones是m行1列的dataframe
def getX(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return data.iloc[:, :-1].values


def getY(df):
    return df.iloc[:, -1:].values



def computeCost(X, y, theta):
    m = X.shape[0]
    #print("m: %s", m)
    temp = X @ theta - y
    square_sum = temp.T @ temp
    #print("temp: %s", temp)
    #print("temp.T: %s", temp.T)
    return square_sum / (2 * m)

def gradientDescent(X, y, theta, alpha, num_iters, j_cost):
    m = X.shape[0]

    for i in range(0, num_iters):
        temp = X @ theta - y
        theta = theta - (alpha / m) * np.dot(temp.T, X).T
        j_cost[i][0] = computeCost(X, y, theta)
    
    return theta


X = getX(df)
y = getY(df)

#print(X.shape)
#print(y.shape)

#print(X)
#print(y.T)

theta = np.zeros((2 ,1))

#print(X.shape, type(X))
#print(y.shape, type(y))
#print("theta: ", theta)
#print(X)
#print(Y)

#Some gradient descent settings
iterations = 1500
alpha = 0.01

#compute and display initial cost
J = computeCost(X, y, theta)

#print(J)


print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, [ [-1], [2]])
print(J)
#printf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);
print('Expected cost value (approx) 54.24\n')


print('\nRunning Gradient Descent ...\n')
# run gradient descent
#j_cost = X.shape[0]
#print(j_cost)
j_cost = np.zeros((iterations,1))

theta = gradientDescent(X, y, theta, alpha, iterations, j_cost)
#print(theta)
#print(j_cost[2][0])
#print(X)
#print(X[:,1])
plt.plot(X[:,1], X * theta)
