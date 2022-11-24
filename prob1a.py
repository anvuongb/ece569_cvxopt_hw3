import numpy as np
import cvxpy as cp
import scipy.io
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt

# Load data
data_train = scipy.io.loadmat("separable_case/train_separable.mat")
data_test = scipy.io.loadmat("separable_case/test_separable.mat")

A = data_train["A"]
B = data_train["B"]
test = data_test["X_test"]
test_label = data_test["true_labels"].ravel()

# define optimization problem
n = 100 # convex combination of 100 data points

u = cp.Variable((n, 1))
v = cp.Variable((n, 1))
ones = np.ones((1,100))

prob = cp.Problem(cp.Minimize( cp.square(cp.norm2(cp.matmul(A,u)-cp.matmul(B,v))) ),
                 [
                     cp.matmul(ones, u)==1,
                     cp.matmul(ones, v)==1,
                     u>=0,
                     v>=0
                 ])

# solve
prob.solve()
print("The optimal value is", prob.value)

# get separating hyperplane
u_opt = u.value
v_opt = v.value

A_opt = np.matmul(A, u_opt)
B_opt = np.matmul(B, v_opt)

gamma = 0.5*(np.linalg.norm(A_opt)**2-np.linalg.norm(B_opt)**2)
normal_vector = A_opt - B_opt
normal_vector = normal_vector

# prediction
preds = np.matmul(test.T, normal_vector) - gamma
preds = np.array([1 if p > 0 else -1 for p in preds])

# calculate accuracy
true_pred = preds == test_label
acc = sum(true_pred)/len(true_pred)
print("accuracy = {:.2f}".format(acc))

# plot figures
x_lin = np.linspace(-5, 5, 50)
y_lin = (-x_lin*normal_vector[0][0] + gamma)/normal_vector[1][0]

# TRAINING PLOT
fig = plt.figure(figsize=(10,10))
plt.scatter(A[0,:],A[1,:], alpha=0.5, color='g', label='Class A')
plt.scatter(B[0,:],B[1,:], alpha=0.5, color='r', label='Class B')

# plot optimal point
plt.scatter(A_opt[0], A_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b', label='support vectors')
plt.scatter(B_opt[0], B_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b')

# plot hyperplane
plt.plot(x_lin, y_lin, label='trained classifier', color='black')
plt.plot([A_opt[0], B_opt[0]], [A_opt[1], B_opt[1]], alpha=0.5, ls="dotted", color='black')

plt.title("Problem 1A - C-Hull - Plot training data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_aspect('equal')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
# plt.show()
plt.tight_layout()
fig.savefig("./figures/plot_1a_training_data.png")

# TESTING PLOT
fig = plt.figure(figsize=(10,10))

plt.scatter(test[:,test_label==1][0,:],test[:,test_label==1][1,:], alpha=0.5, color='g', label='Class A')
plt.scatter(test[:,test_label==-1][0,:],test[:,test_label==-1][1,:], alpha=0.5, color='r', label='Class B')

# plot optimal point
plt.scatter(A_opt[0], A_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b', label='support vectors from training set')
plt.scatter(B_opt[0], B_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b')

# plot hyperplane
plt.plot(x_lin, y_lin, label='trained classifier', color='black')
plt.plot([A_opt[0], B_opt[0]], [A_opt[1], B_opt[1]], alpha=0.5, ls="dotted", color='black')

plt.title("Problem 1A - C-Hull - Plot test data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_aspect('equal')
plt.xlim(-8, 8 )
plt.ylim(-8, 8 )
# plt.show()
plt.tight_layout()
fig.savefig("./figures/plot_1a_testing_data.png")


