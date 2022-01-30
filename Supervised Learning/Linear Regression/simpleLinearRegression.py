import csv
import math as algebra


class Linear_Regression:

    def __init__(self):
        pass

    def load_csv(self, filename):
        dataset = []
        with open(filename, 'r') as file:
            header = csv.reader(file)
            for row in header:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    def string_to_float(self, dataset, column):
        for row in range(len(dataset[0])):
            row[column] = float(row[column].strip())
 
    def mean(self, list):
        length = len(list)
        sum = 0
        meanVal = 0
        for n in range(length):
            sum += list[n]
        meanVal = sum/float(length)
        # print(f"\nMean = {meanVal}")
        return meanVal

    def variance(self, Xi):
        Xm = self.mean(Xi)
        X_len = len(Xi)
        sum = 0
        var = 0
        for iter in range(X_len):
            sum = sum + algebra.pow(Xi[iter]-Xm, 2)
        var = sum/float(X_len-1)
        # print(f"\nVariance = {var}")
        return var

    def covariance(self, Xi, Yi):
        Xm = self.mean(Xi)
        Ym = self.mean(Yi)
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

    def coefficients(self, trainSet):
        X_data = []
        Y_data = []
        for iter in trainSet:
            X_data.append(iter[0])
            Y_data.append(iter[1])
        Xm = self.mean(X_data)
        Ym = self.mean(Y_data)
        beta1 = self.covariance(X_data, Y_data)/ self.variance(X_data)
        beta0 = Ym - beta1*Xm
        print(f"Coefficients :- beta0 = {beta0}\tbeta1 = {beta1}\n")
        return [beta0, beta1] 

    def predict(self, trainSet, beta):
        predictions = []
        for iter in range(self.length):
            yhat = beta[0] + beta[1] * trainSet[iter][0]
            predictions.append(yhat)
        return predictions

    # Set1 -> Set containing actual values and Set2 -> Set containing predicted values
    def root_mean_square_error(self, Set1, Set2): 
        totalError = 0.0
        iterations = len(Set1)
        for iter in range(iterations):
            predictionError = Set1[iter] - Set2[iter]
            totalError = totalError + algebra.pow(predictionError, 2)
        meanError = totalError/float(iterations)
        return meanError


# Training the ML model using Linear Regression Algorithm 
print("\n\t\t\t\t Training ML model \n")
linear_reg_var = Linear_Regression()
excel = 'Mouse Size.csv'
dataSet = linear_reg_var.load_CSV(excel)
for i in range(len(dataSet[0])):
    linear_reg_var.string_to_float(dataSet, i)
params = linear_reg_var.coefficients(dataSet)
predictionSet = linear_reg_var.predict(dataSet, params)
actualSet = []
for row in dataSet:
    actualSet.append(dataSet[-1])
rmse = linear_reg_var.root_mean_square_error(actualSet, predictionSet)
print(f"Actual Set = {actualSet}")
print(f"Predicted Set = {predictionSet}")
print('Root Mean Squared Error = %.3f' %(rmse))

# Testing the ML model using Test Dataset
print("\n\t\t\t\t Testing ML model \n")
excel = 'Mouse Size test.csv'
dataSet = linear_reg_var.load_CSV(dataSet)
actual_value = []
for row in dataSet:
    actual_value.append(row[-1])
print(f"Testing:- Actual Set = {actual_value}")
predicted_value = linear_reg_var.predict(dataSet, params)
print(f"Testing:- Predicted Set = {predicted_value}")
rmse = linear_reg_var.root_mean_square_error(actual_value, predicted_value)
print(f"Root Mean Square Error = {rmse}")

