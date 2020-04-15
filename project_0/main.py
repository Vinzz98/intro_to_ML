import numpy as np

# file = open("task0_sl19d1/train.csv")
# file.readline()  # skip the header
# train_data = np.loadtxt(file)

train_data = np.genfromtxt("task0_sl19d1/train.csv", delimiter=',', skip_header=1)

X = train_data[:, 2:]
Y = train_data[:, 1]

w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

# file = open("task0_sl19d1/test.csv")
# file.readline()
# test_data = np.loadtxt(file)

test_data = np.genfromtxt("task0_sl19d1/test.csv", delimiter=',', skip_header=1)

X_test = test_data[:, 1:]
Y_test = np.dot(X_test, w)

y_ref = np.empty(X_test.shape[0])

for i in range(X_test.shape[0]):
    sum = 0
    for j in range(X_test.shape[1]):
       sum += X_test[i, j]
    np.append(y_ref, (sum / X_test.shape[1]))

error = 0
for i in range(X_test.shape[0]):
    error += (Y_test[i] - y_ref[i])**2
error = error / X_test.shape[0]
error = error**0.5

print(error)

output = test_data[:, 0]
np.append(output, Y_test, axis=0)

np.set_printoptions(threshold=np.inf)
# print(Y_test)
# print(output)

file = open("output.csv", "w+")
file.write("Id,y\n")
for i in range(output.shape[0]):
    file.write("{:.0f},{:.1f}\n".format(output[i], Y_test[i]))
file.close()
