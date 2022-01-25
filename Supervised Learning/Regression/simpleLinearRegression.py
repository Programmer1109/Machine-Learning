import csv
import math as algebra


def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        header = csv.reader(file)
        for row in header:
            if not row:
                continue
            dataset.append(row)
    return dataset


def string_to_float(dataset, column):
    for row in range(len(dataset[0])):
        row[column] = float(row[column].strip())
        

def mean(list):
    length = len(list)
    sum = 0
    meanVal = 0
    for n in range(length):
        sum += list[n]
    meanVal = sum/float(length)
    # print(f"\nMean = {meanVal}")
    return meanVal


def variance(Xi):
    Xm = mean(Xi)
    X_len = len(Xi)
    sum = 0
    var = 0
    for iter in range(X_len):
        sum = sum + algebra.pow(Xi[iter]-Xm, 2)
    var = sum/float(X_len-1)
    # print(f"\nVariance = {var}")
    return var


def covariance(Xi, Yi):
    Xm = mean(Xi)
    Ym = mean(Yi)
    X_len = len(Xi)
    Y_len = len(Yi)
    sum = 0
    coVar = 0
    if X_len == Y_len:
        for iter in range(X_len):
            term1 = Xi[iter]-Xm
            term2 = Yi[iter]-Ym
            sum = sum + term1*term2
        coVar = sum/float(Xm-1)
        return coVar
    else:
        print("Co-Variance of the two input sets is not computable as both sets are of unequal lengths.")
        return -1


def coefficients(X_data, Y_data):
    Xm = mean(X_data)
    Ym = mean(Y_data)
    beta1 = covariance(X_data, Y_data)/ variance(X_data)
    beta0 = Ym - beta1*Xm
    print(f"Coefficients :- beta0 = {beta0}\tbeta1 = {beta1}\n")
    return [beta0, beta1] 


def simpleLinearRegression(trainSet):
    predictions = []
    Set1 = []
    Set2 = []
    for elem in trainSet:
        Set1.append(elem[0])
        Set2.append(elem[1])
    coEffs = coefficients(Set1, Set2)
    for iter in Set1:
        yHat = coEffs[0] + coEffs[1] * iter
        predictions.append(yHat)
    return predictions


# Set1 -> Set containing actual values and Set2 -> Set containing predicted values
def root_mean_square_error(Set1, Set2): 
    totalError = 0.0
    iterations = len(Set1)
    for iter in range(iterations):
        predictionError = Set1[iter] - Set2[iter]
        totalError = totalError + algebra.pow(predictionError, 2)
    meanError = totalError/float(iterations)
    return meanError


def evaluateAlgorithm(trainSet):
    actualSet = []
    for elem in trainSet:
        actualSet.append(elem[1])
    print(f"Actual Values = {actualSet}")
    predictedSet = simpleLinearRegression(trainSet)
    print(f"Predicted Values = {predictedSet}")
    error = root_mean_square_error(actualSet, predictedSet)
    return error


def main():
    dataSet = [[108, 392.5], [19, 46.2], [13, 15.7], [124, 422.2], [40, 119.4], [57, 170.9], [23, 56.9], [14, 77.5], [45, 214], [10, 65.3]]
    X_dataset = []
    Y_dataset = []
    for element in dataSet:
        X_dataset.append(element[0])
        Y_dataset.append(element[1])
    rmse = evaluateAlgorithm(dataSet)
    print('Root Mean Squared Error = %.3f' %(rmse))


main()
