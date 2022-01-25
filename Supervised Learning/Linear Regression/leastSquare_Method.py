import math as algebra


def mean(Set):
    total = 0
    len_set = len(Set)
    mean_val = 0 
    for n in range(len_set):
        total = total + Set[n]
    mean_val = total/float(len_set)
    print(f"Mean = {mean_val}")
    return mean_val


def coefficients(x_data, y_data):
    A = mean(y_data)
    y_len = len(y_data)
    x_len = len(x_data)
    # Summation of X 
    Sum = 0
    if (x_len % 2) != 0: 
        for iter in range(y_len):
            x_data[iter] = iter - (x_len//2)
            Sum = Sum + x_data[iter]
    elif (x_len%2) == 0:
        for iter in range(y_len):
            if iter >= x_len//2:
                x_data[iter] = iter - (x_len//2) + 1
            elif iter < x_len//2:
                x_data[iter] = iter - (x_len//2)    
            Sum = Sum + x_data[iter]
    # print(f"NEW X Dataset :- X Data = {x_data}")
    if Sum != 0:
        print("Error: Non-zero sum of X error")
        exit()
    else:
        # Summation of X square
        Sum_X2 = 0
        # Summation of XY
        Sum_XY = 0
        for iter in range(y_len):
            Sum_XY += x_data[iter]*y_data[iter]
            Sum_X2 += algebra.pow(x_data[iter], 2)
        B = Sum_XY/Sum_X2
        # print(f"Co-efficients :- \n\tA = {A}\tB = {B}")
        return [A, B]


def predict(trainSet):
    predictions = []
    Xi = []
    Yi = []
    for row in trainSet:
        Xi.append(row[0])
        Yi.append(row[1])
    # print(f"X Dataset = {Xi}\nY Dataset = {Yi}")
    Constants = coefficients(Xi, Yi)
    # print(f"Actual Dataset = {Yi}")
    length = len(trainSet)
    if (length % 2) != 0: 
        for iter in range(length):
            Xi[iter] = iter - (length//2)
    elif (length%2) == 0:
        for iter in range(length):
            if iter >= length//2:
                Xi[iter] = iter - (length//2) + 1
            elif iter < length//2:
                Xi[iter] = iter - (length//2)  
    for iter in range(length):
        yhat = Constants[0] + Constants[1] * Xi[iter]
        predictions.append(yhat)
    # print(f"Predicted Dataset = {predictions}")


def main():
    dataSet = [[2010, 60], [2011, 72], [2012, 75], [2013, 65], [2014, 80], [2015, 85], [2016, 95]]
    predict(dataSet)


main()
