import csv
import math as algebra


def load_CSV(filename):
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


def prediction(Set, beta):
    Yhat = []
    for iter in range(len(Set)):
        value = beta[0] + beta[1]*Set[iter][0] + beta[2]*Set[iter][1] + beta[3]*Set[iter][2] 
        Yhat.append(value)
    return Yhat


def sum_of_derivative(Y_set, X_set, y_Bar):
    if len(X_set) == len(Y_set):
        derivativesum = 0.0
        length = len(X_set)
        for iter in range(length):
            derivativesum = derivativesum + (y_Bar[iter]-Y_set[iter]) * X_set[iter]
        return derivativesum/float(length)
    else:
        return None



def coefficients(trainSet):
    alpha = float(input("Enter the learning rate: "))
    max_iterations = int(input("Enter no. of iterations: "))
    if max_iterations>1000:
        max_iterations = 1000
    epoch = 1
    Y_data = []
    X0_data = []
    X1_data = []
    X2_data = []
    X3_data = []
    beta0 = 0.0
    beta1 = 0.0
    beta2 = 0.0
    beta3 = 0.0
    beta = [beta0, beta1, beta2, beta3]
    for row in trainSet:
        X0_data.append(1)
        X1_data.append(row[0])
        X2_data.append(row[1])
        X3_data.append(row[2])
        Y_data.append(row[3]) 
    print(f"Training:- Actual Set = {Y_data}")
    while((stepSize0>=0.001 or stepSize1>=0.001 or stepSize2>=0.001 or stepSize3>=0.001) and epoch<=max_iterations):
        yBar = prediction(trainSet, beta)
        stepSize0 = alpha * sum_of_derivative(Y_data, X0_data, yBar)
        stepSize1 = alpha * sum_of_derivative(Y_data, X1_data, yBar)
        stepSize2 = alpha * sum_of_derivative(Y_data, X2_data, yBar)
        stepSize3 = alpha * sum_of_derivative(Y_data, X3_data, yBar)
        beta0 = beta0 - stepSize0
        beta1 = beta1 - stepSize1
        beta2 = beta2 - stepSize2
        beta3 = beta3 - stepSize3
        epoch = epoch + 1
    return [beta0, beta1, beta2, beta3]


# Set1 -> Set containing actual values and Set2 -> Set containing predicted values
def root_mean_square_error(Set1, Set2): 
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
        

def testing(param):
    Size = float(input("Enter size of house(in feet^2) = "))
    Bedrooms = float(input("Enter no. of bedrooms = "))
    Age = float(input("Enter age of house(in years) = "))
    House_Price = param[0] + param[1]*Size + param[2]*Bedrooms + param[3]*Age
    return House_Price


# Training the ML model using Gradient Descent Algorithm
print("\n\t\t\t\t Training ML model \n")
excel = 'House Price.csv'
dataSet = load_CSV(excel)
params = coefficients(dataSet)
predictSet = prediction(params, dataSet)
print(f"Training:- Predicted Set = {predictSet}") 
# Testing the ML model using Test Dataset
print("\n\t\t\t\t Testing ML model \n")
test_csv = 'House Price test.csv'
test_dataSet = load_CSV(test_csv)
actual_value = []
for row in test_dataSet:
    actual_value.append(row[-1])
print(f"Testing:- Actual Set = {actual_value}")
predicted_value = prediction(test_dataSet, params)
print(f"Testing:- Predicted Set = {predicted_value}")
RMSE = root_mean_square_error(actual_value, predicted_value)
print(f"Root Mean Square Error = {RMSE}")
# Deploying model to real world applications
print("\n\n\t\t\t\t House Price Prediction APP\n")
while True:
    result = testing(params)
    print(f"Your house Price = {result}") 
    Quit_status = input("\nDo you want to Quit?\t1.Y\t or \t2.No")
    if Quit_status.lower() == "yes" or Quit_status.lower() == 'y':
        break
    elif Quit_status.lower() == "yes" or Quit_status.lower() == 'y':
        continue
print("Thanks for using the APP!!!")
