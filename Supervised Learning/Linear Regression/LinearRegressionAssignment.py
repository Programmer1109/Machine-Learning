'''
        Name:- Anthony Claude Dsouza
        Reg No.:- 202370413
        Module:- DM954 Intelligent Sensing and Reasoning through ML
        Date:- 10/19/2023
'''

from numpy.lib import arraysetops
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import random
import math

plt.style.use('_mpl-gallery')

# Generate Dataset for Linear Regression
x = np.linspace(0, 100, 100)
y = 4.2 + 2.31415 * x

noisy_x = np.linspace(0, 100, 124)
noise = np.random.normal(0, 2, len(noisy_x))
noisy_y = 4.2 + 0.31415 * noisy_x + np.random.normal(0, 2, len(noisy_x))

# Add outliers to noisy
outlier_fraction = 0.04
random_out = random.sample(range(0, len(noisy_y)), int(outlier_fraction * len(noisy_y)))

for i in random_out:
    noisy_y[i] = np.mean(noisy_y) + (random.randint(5, 15) * np.std(noisy_y))

# Add Nan values to noisy y
nan_fraction = 0.2
random_nan = random.sample(range(0, len(noisy_y)), int(nan_fraction * len(noisy_y)))

for i in random_nan:
    noisy_y[i] = np.nan

for i in range(0, 124):
    print(f'x[{i}] = {noisy_x[i]:10} \t y[{i}] = {noisy_y[i]:10} ')
# Data Generation Ends Here

# Visualizing the Data for ML Model
print("Graphical view of Dataset before Pre-Processing")
plt.scatter(noisy_x, noisy_y)
plt.xlabel('Input Data')
plt.ylabel('Output Data')
plt.show()
plt.clf()

''' Perform Data Pre-Processing here '''
#               Replace NAN values in the array
array = np.array(noisy_y)
# Calculate the median excluding NaN values
median = np.nanmedian(array)
# print("Median = ", median)
# Replace NaN values with the median
array = np.where(np.isnan(array), median, array)
# print(f"Results:- \nNew Data -> \n{array}\nOld Data -> \n{noisy_y}\n")

#               Remove Outliers from the data
quartile1 = np.percentile(array, 25)
quartile3 = np.percentile(array, 75)
IQR = quartile3 - quartile1
upperbound = quartile3 + 1.5 * IQR
lowerbound = quartile1 - 1.5 * IQR
# print(f"Q1 = {quartile1}\tQ3 = {quartile3}\tIQR = {IQR}\tUpperbound = {upperbound}\tLowerbound = {lowerbound}\n")
newY = []
newX = []
no_of_outliers = 0
# print("Entering for loop...")
for i in range(len(array)):
    # print(f"Array[{i}] = {array[i]}\tupperbond = {upperbound}\tlowerbond = {lowerbound}")
    if array[i] >= upperbound:
      no_of_outliers += 1
      # print(f"Outlier = {array[i]}")
      # newY.append(upperbound)
      # newX.append(noisy_x[i])
    elif(array[i] <= lowerbound):
      no_of_outliers += 1
      # print(f"Outlier = {array[i]}")
      # newY.append(lowerbound)
      # newX.append(noisy_x[i])
    else:
      # print("Append element")
      newY.append(array[i])
      newX.append(noisy_x[i])

# print("Exiting for loop...")
# print(no_of_outliers)

# newX = noisy_x
# newY = array

# print(f"Results:- \nProcessed Data-> \n{newY}\n {newX})\n Length of X: {len(newX)}\nLength of Y: {len(newY)}")
print("Graphical view of Dataset after Pre-Processing")
plt.scatter(newX, newY)
plt.xlabel('Input Data')
plt.ylabel('Output Data')
plt.show()
plt.clf()
# Data Pre-Processing Ends Here

#             Processing the Data using Linear Regression
# Splitting the Dataset into Training and Testing Set
trainX, testX, trainY, testY = train_test_split(newX, newY, test_size=0.2) # 80-20 train test split
#print(f"SplittingResults:-\n\t trainX = \n{trainX} Length = {len(trainX)}\n\ttestX = \n{testX} Length = {len(testX)}\n\ttrainY = \n{trainY} Length = {len(trainY)}\n\ttestY = \n{testY} Length = {len(testY)}" )
meanX = np.mean(trainX)
meanY = np.mean(trainY)
#print(f"Mean of X = {meanX}\tMEan of Y = {meanY}")

# Calculate the Co-efficients for Linear Regression
coVar = 0
Var = 0
for i in range(len(trainX)):
    coVar += (trainX[i] - meanX) * (trainY[i] - meanY)
    Var += (trainX[i] - meanX) ** 2
coefficientB1 = coVar/Var
coefficientB0 = meanY - coefficientB1 * meanX
print("\n\n\t\t\t Training Dataset Results\n")
print(f"\n\tCo-efficients of Regression:- Slope = {coefficientB1}\tIntercept = {coefficientB0}")
print(f"\tREGRESSION LINE EQUATION:- y = {coefficientB1} x + {coefficientB0}")

# Predicting values of X and Y Dataset
predictY = []
for i in range(len(trainX)):
    predictY.append(coefficientB0 + coefficientB1 * trainX[i])
# print(f"Prediction for Y = {predictY}")
RMSE = math.sqrt(mse(predictY, trainY))
print(f"\tRoot Mean Square Error = {RMSE}")

varY_hat = 0
varY = 0
meanY_hat = np.mean(predictY)
for i in range(len(trainY)):
  varY += (trainY[i] - meanY) ** 2
  varY_hat += (predictY[i] - meanY_hat) ** 2
R_square_value = varY_hat/varY
print(f"\tR2 = {R_square_value}")

plt.plot(trainX, predictY, color='red', label = 'Regression Line',)
plt.scatter(newX, newY, label= 'Data Points')
plt.xlabel('Input Data')
plt.ylabel('Output Data')
plt.show()

# Testing the model on Test Data
predictY_test = []
for i in range(len(testX)):
  predictY_test.append(coefficientB0 + coefficientB1 * testX[i])
print("\n\n\t\t\t Testing Dataset Results\n")
# print(f"\nTesting Results:-\nPrediction Y = \n{predictY_test}\nTest Dataset Y = \n{testY}\nTest Dataset X = {testX}")
RMSE = math.sqrt(mse(predictY_test, testY))
print(f"\tRoot Mean Square Error = {RMSE}")

varY_hat = 0
varY = 0
meanY = np.mean(testY)
meanY_hat = np.mean(predictY_test)
for i in range(len(testY)):
  varY += (testY[i] - meanY) ** 2
  varY_hat += (predictY_test[i] - meanY_hat) ** 2
R_square_value = varY_hat/varY
print(f"\tR2 = {R_square_value}")

plt.plot(testX, predictY_test, color='red', label = 'Regression Line')
plt.scatter(testX, predictY_test)
plt.xlabel('Input Data')
plt.ylabel('Output Data')
plt.show()
