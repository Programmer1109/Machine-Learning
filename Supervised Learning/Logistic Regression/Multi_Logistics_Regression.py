import csv
import math as algebra


class Gradient_Descent:

    def __init__(self):
        self.alpha = input(print("Enter the learning rate = "))
        self.max_iterations = input(print("Enter no. of epochs = "))

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

    def prediction(self, Set, beta):
        Yhat = []
        for outer in range(len(Set)):
            value = 0.0
            for inner in range(len(Set[0])):
                value += beta[inner] * Set[outer][inner] 
                Yhat.append(value)
        return Yhat

    def sum_of_derivative(self, dataSet, y_Bar, param):
        derivativesum = 0.0
        length = len(dataSet)
        for iter in range(length):
            derivativesum += (y_Bar[iter]-dataSet[iter][-1]) * dataSet[iter][param]
        return derivativesum/float(length)

    def coefficients(self, trainSet):
        if self.max_iterations>1000:
            self.max_iterations = 1000
        epoch = 1
        beta = []
        stepSize = []
        loopVar = 0
        for i in range(len(trainSet[0])-1):
            beta.append(0.0)
            stepSize.append(0.0)
        # print(f"Training:- Actual Set = {Y_data}")
        while(epoch<=self.max_iterations and loopVar!=0):
            loopVar = 0
            for iter in range(len(trainSet[0])-1):
                if stepSize[iter]>=0.001 or iter==0:
                    yBar = self.prediction(trainSet, beta)
                    stepSize[iter] = self.alpha * self.sum_of_derivative(trainSet, yBar, iter)
                    beta[iter] -= stepSize[iter]
                    epoch = epoch + 1
                    loopVar += 1
                else:
                    break
        return beta

    # Set1 -> Set containing actual values and Set2 -> Set containing predicted values
    def root_mean_square_error(self, Set1, Set2): 
        if len(Set1) == len(Set2):
            totalError = 0.0 
            iterations = len(Set1)
            for iter in range(iterations):
                predictionError = Set1[iter] - Set2[iter]
                totalError = totalError + algebra.pow(predictionError, 2)
            meanError = totalError/float(iterations)
            return meanError
        else:
            return None
            
    def testing(self, param):
        Size = float(input("Enter size of house(in feet^2) = "))
        Bedrooms = float(input("Enter no. of bedrooms = "))
        Age = float(input("Enter age of house(in years) = "))
        House_Price = param[0] + param[1]*Size + param[2]*Bedrooms + param[3]*Age
        return House_Price


# Training the ML model using Gradient Descent Algorithm
print("\n\t\t\t\t Training ML model \n")
gradDes_var = Gradient_Descent()
# Preparing Data for 
excel = 'House Price.csv'
dataSet = gradDes_var.load_CSV(excel)
params = gradDes_var.coefficients(dataSet)
predictSet = gradDes_var.prediction(params, dataSet)
print(f"Training:- Predicted Set = {predictSet}") 
# Testing the ML model using Test Dataset
print("\n\t\t\t\t Testing ML model \n")
test_csv = 'House Price test.csv'
test_dataSet = gradDes_var.load_CSV(test_csv)
actual_value = []
for row in test_dataSet:
    actual_value.append(row[-1])
print(f"Testing:- Actual Set = {actual_value}")
predicted_value = gradDes_var.prediction(test_dataSet, params)
print(f"Testing:- Predicted Set = {predicted_value}")
RMSE = gradDes_var.root_mean_square_error(actual_value, predicted_value)
print(f"Root Mean Square Error = {RMSE}")
# Deploying model to real world applications
print("\n\n\t\t\t\t House Price Prediction APP\n")
while True:
    result = gradDes_var.testing(params)
    print(f"Your house Price = {result}") 
    Quit_status = input("\nDo you want to Quit?\t1.Y\t or \t2.No")
    if Quit_status.lower() == "yes" or Quit_status.lower() == 'y':
        break
    elif Quit_status.lower() == "yes" or Quit_status.lower() == 'y':
        continue
print("Thanks for using the APP!!!")
