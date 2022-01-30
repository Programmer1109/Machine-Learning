import math as algebra
import csv


class simpleLogistic_regression:

	def __init__(self):
		self.threshold = 0.5

	def load_CSV(self, filename):
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
		sum = 0
		coVar = 0
		if len(Xi) == len(Yi):
			for iter in range(len(Xi)):
				term1 = Xi[iter]-Xm
				term2 = Yi[iter]-Ym
				sum = sum + term1*term2
			coVar = sum/float(Xm-1)
			return coVar
		else:
			print("Co-Variance of the two input sets is not computable as both sets are of unequal lengths.")
			exit()

	# Set1 -> Set of actual values & Set2 -> Set of predicted values 
	def calcThreshold(self, Set1, Set2):
		TPR = 0.0
		FPR = 0.0
		TNR = 0.0
		FNR = 0.0
		for iter in range(len(Set1)):
			if Set1[iter] == Set2[iter]:
				if Set1[iter] == 1:
					TPR += 1.00
				elif Set1[iter] == 0:
					TNR += 1.00
			elif Set1[iter] != Set2[iter]:
				if Set1[iter] == 1:
					FPR += 1.00
				elif Set1[iter] == 0:
					FNR += 1.00
		totalElements = float(len(Set1))
		TPR = TPR/totalElements
		TNR = TNR/totalElements
		FPR = FPR/totalElements
		FNR = FNR/totalElements
		

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
		proability = []
		for iter in range(trainSet):
			yhat = beta[0] + beta[1] * trainSet[iter][0]
			proSuccess = algebra.exp(yhat)/(1+algebra.exp(yhat))
			if proSuccess>self.threshold:
				proSuccess = 1
			elif proSuccess<self.threshold:
				proSuccess = 0
			proability.append(proSuccess)
		return proability


# Training ML model using Logistics Regression
print("\n\t\t\t\t Training ML model \n")
logistics_var = simpleLogistic_regression()
# Preparing Data for 
excel = 'House Price.csv'
dataSet = simpleLogistic_regression.load_CSV(excel)
for i in range(len(dataSet[0])):
    simpleLogistic_regression.string_to_float(dataSet, i)
params = simpleLogistic_regression.coefficients(dataSet)
predictSet = simpleLogistic_regression.predict(dataSet, params)
print(f"Training:- Predicted Set = {predictSet}") 
# Testing the ML model using Test Dataset
print("\n\t\t\t\t Testing ML model \n")
test_csv = 'House Price test.csv'
test_dataSet = simpleLogistic_regression.load_CSV(test_csv)
actual_value = []
for row in test_dataSet:
    actual_value.append(row[-1])
print(f"Testing:- Actual Set = {actual_value}")
predicted_value = simpleLogistic_regression.prediction(test_dataSet, params)
print(f"Testing:- Predicted Set = {predicted_value}")



