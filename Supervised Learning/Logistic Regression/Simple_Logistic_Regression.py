import math as algebra
import csv


class simpleLogistic_regression:

	def __init__(self):
		self.threshold = 0.5
		self.count = [0, 0, 0, 0]

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
		for row in dataset[0]:
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
		return [beta0, beta1]

	def predict(self, trainSet, beta):
		proability = []
		for iter in range(trainSet):
			yhat = beta[0] + beta[1] * trainSet[iter][0]
			proSuccess = algebra.exp(yhat)/(1+algebra.exp(yhat))
			if proSuccess>=self.threshold:
				proSuccess = 1
			elif proSuccess<self.threshold:
				proSuccess = 0
			proability.append(proSuccess)
		return proability

	# Set1 -> Set of actual values & Set2 -> Set of predicted values 
	def measureAccuracy(self, Set1, Set2):
		count = [0, 0, 0, 0]
		for iter in range(len(Set1)):
			# Correct Predictions
			if Set1[iter] == Set2[iter]:
				# Prediction:- success and Actually:- success -> True Positive	
				if Set1[iter] == 1:			  
					count[0] += 1
				# Prediction:- failure and Actually:- failure -> True Negative
				elif Set1[iter] == 0:		
					count[1] += 1
			# Incorrect Predictions
			elif Set1[iter] != Set2[iter]:
				# Prediction:- failure and Actually:- success -> False Negative	
				if Set1[iter] == 1:			
					count[2] += 1
				# Prediction:- success and Actually:- failure -> False Positive
				elif Set1[iter] == 0:		 
					count[3] += 1
		Recall = count[0]/(count[0]+count[2])
		Specificity = count[1]/(count[1]+count[3])
		Precision = count[0]/(count[0]+count[3])
		Prevalence = (count[0]+count[2])/(count[0]+count[1]+count[2]+count[3])
		correctPrediction = (count[0]+count[1])/(count[0]+count[1]+count[2]+count[3])
		return [Recall, Specificity, Precision, Prevalence, correctPrediction]

	def calc_Threshold(self, Set1, Set2):
		threshold_values = list(range(0,1, 0.1))
		print(f"Set of threshold values = {threshold_values}")
		Ones = 0
		Zeros = 0
		for row in Set1:
			if row==1:
				Ones = Ones + 1
			else:
				Zeros = Zeros + 1
		if Ones<=0.70*(Ones+Zeros):
			ROC_point = list()
			for value in threshold_values:
				self.count = [0, 0, 0, 0]
				for iter in range(len(Set1)):
				# Correct Predictions
					if Set1[iter] == Set2[iter]:
						# Prediction:- success and Actually:- success -> True Positive	
						if Set1[iter] == 1:			  
							self.count[0] += 1
						# Prediction:- failure and Actually:- failure -> True Negative
						elif Set1[iter] == 0:		
							self.count[1] += 1
					# Incorrect Predictions
					elif Set1[iter] != Set2[iter]:
						# Prediction:- failure and Actually:- success -> False Negative	
						if Set1[iter] == 1:			
							self.count[2] += 1
						# Prediction:- success and Actually:- failure -> False Positive
						elif Set1[iter] == 0:		 
							self.count[3] += 1
				Recall = self.count[0]/(self.count[0]+self.count[2])
				Specificity = self.count[1]/(self.count[1]+self.count[3])
				ROC_point.append(Recall, 1-Specificity)
		elif Ones>0.7*(Ones+Zeros):
			ROC_point = list()
			for value in threshold_values:
				self.count = [0, 0, 0, 0]
				for iter in range(len(Set1)):
				# Correct Predictions
					if Set1[iter] == Set2[iter]:
						# Prediction:- success and Actually:- success -> True Positive	
						if Set1[iter] == 1:			  
							self.count[0] += 1
						# Prediction:- failure and Actually:- failure -> True Negative
						elif Set1[iter] == 0:		
							self.count[1] += 1
					# Incorrect Predictions
					elif Set1[iter] != Set2[iter]:
						# Prediction:- failure and Actually:- success -> False Negative	
						if Set1[iter] == 1:			
							self.count[2] += 1
						# Prediction:- success and Actually:- failure -> False Positive
						elif Set1[iter] == 0:		 
							self.count[3] += 1
				Recall = self.count[0]/(self.count[0]+self.count[2])
				Precision = self.count[0]/(self.count[0]+self.count[3])
				ROC_point.append(Recall, Precision)


# Training ML model using Logistics Regression
print("\n\t\t\t\t Training ML model \n")
logistics_var = simpleLogistic_regression()
# Preparing Data for 
excel = 'Mouse Obesity.csv'
dataSet = simpleLogistic_regression.load_CSV(excel)
for i in range(len(dataSet[0])):
    simpleLogistic_regression.string_to_float(dataSet, i)
actualSet = []
for row in dataSet:
	actualSet.append(row[-1])
print(f"Actual Set = {actualSet}")
params = simpleLogistic_regression.coefficients(dataSet)
print(f"Coefficients:-\n\t beta0 = {params[0]}\t beta1 = {params[1]}")
predictSet = simpleLogistic_regression.predict(dataSet, params)
print(f"Predicted Set = {predictSet}")
accuracyMetrics = simpleLogistic_regression.measureAccuracy(actualSet, predictSet)
print(f"Training Accuracy Results:-\n\tRecall = {accuracyMetrics[0]}\n\tSpecificity = {accuracyMetrics[1]}\n\tPrecision = {accuracyMetrics[2]}")
print(f"\tPrevalence = {accuracyMetrics[3]}\n\tClassification Accuracy = {accuracyMetrics[4]}\n\tMisclassification Accuracy = {1-accuracyMetrics[4]}")
simpleLogistic_regression.calc_Threshold(actualSet, predictSet)

# Testing the ML model using Test Dataset
print("\n\n\t\t\t\t Testing ML model \n")
excel = 'Mouse Obesity test.csv'
dataSet = simpleLogistic_regression.load_CSV(excel)
actualSet = []
for row in dataSet:
    actualSet.append(row[-1])
print(f"Testing:- Actual Set = {actualSet}")
predictSet = simpleLogistic_regression.prediction(dataSet, params)
print(f"Testing:- Predicted Set = {predictSet}")
accuracyMetrics = simpleLogistic_regression.measureAccuracy(actualSet, predictSet)
print(f"Training Accuracy:-\n\tRecall = {accuracyMetrics[0]}\n\tSpecificity = {accuracyMetrics[1]}\n\tPrecision = {accuracyMetrics[2]}")
print(f"\tPrevalence = {accuracyMetrics[3]}\n\tClassification Accuracy = {accuracyMetrics[4]}\n\tMisclassification Accuracy = {accuracyMetrics[5]}")
