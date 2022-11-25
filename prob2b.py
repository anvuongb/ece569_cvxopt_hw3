import numpy as np
import cvxpy as cp
import scipy.io
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import time
import pickle

# Helpers for calculate GD and projection
def calc_gradient_descent(u, v, A, B, iteration=None, alpha=0.1):
    i = 1
    if iteration != None:
        i = iteration
    return u - alpha/(np.sqrt(i))*(2*np.matmul(np.matmul(A.T, A), u) - 2*np.matmul(np.matmul(A.T, B), v))

def cal_projection_prob_simplex_simple(x):
    x_sort = x[np.argsort(-x, axis=0).ravel()]
    t = 1/np.arange(1,len(x)+1)*(1-np.cumsum(x_sort))
    z = (x_sort + t.reshape((-1, 1))).ravel()
    rho = len(z[z>0])
    lamb = 1/rho*(np.sum(x_sort[:rho])-1)
    x_proj = np.max(np.concatenate([x - lamb, np.zeros(x.shape)], axis=1), axis=1)
    return x_proj.reshape((-1,1)), 0

def calc_projection_optimization(x, n=100, d=0.02):
    # optimmization problem
    y = cp.Variable((n, 1))
    ones = np.ones((1,100))
    prob = cp.Problem(cp.Minimize( cp.square(cp.norm2(y-x)) ),
                 [cp.matmul(ones, y)==1,
                  y>=0,
                  d*ones.T >= y])
    prob.solve(verbose=False)
    return y.value, prob.value

###################
### CONVEX HULL ###
###################
print("Working on Convex Hull")
data_train = scipy.io.loadmat("separable_case/train_separable.mat")
data_test = scipy.io.loadmat("separable_case/test_separable.mat")

A = data_train["A"]
B = data_train["B"]

test = data_test["X_test"]
test_label = data_test["true_labels"].ravel()

np.random.seed(42) # for reproducibility

u_0 = scipy.special.softmax(np.random.uniform(0, 1, (100, 1)))
v_0 = scipy.special.softmax(np.random.uniform(0, 1, (100, 1)))

u_prev = u_0
v_prev = v_0
u_prev_prev = u_0
v_prev_prev = v_0

alpha = 0.1
prob_vals = []
time_list = []
uv_list = []

min_d = 0.01 # stopping criteria
max_it = 100 # max number of iters
min_it = 20 # min number of iters

# calc first 
prob_vals.append(np.square(np.linalg.norm(np.matmul(A, u_prev_prev)-np.matmul(B, v_prev_prev))))
prob_vals.append(np.square(np.linalg.norm(np.matmul(A, u_prev)-np.matmul(B, v_prev))))
# projected gradient loop
for i in range(max_it): 
    start = time.time()
    t = ((i+1)/2-1)/((i+1+1)/2)
    u_t = (1+t)*u_prev - t*u_prev_prev
    u_curr_grad = calc_gradient_descent(u_t, v_prev, A, B, alpha=alpha)
    # u_curr, _ = calc_projection_optimization(u_curr_grad, d=1)
    u_curr, _ = cal_projection_prob_simplex_simple(u_curr_grad)
    # print(sum(u_curr))
    v_t = (1+t)*v_prev - t*v_prev_prev
    v_curr_grad = calc_gradient_descent(v_prev, u_prev, B, A, alpha=alpha) 
    # v_curr, _ = calc_projection_optimization(v_curr_grad, d=1)
    v_curr, _ = cal_projection_prob_simplex_simple(v_curr_grad)
    # print(sum(v_curr))
    end = time.time()

    # save time and current opt vectors
    time_list.append(end-start)
    uv_list.append([u_curr, v_curr])
    prob_vals.append(np.square(np.linalg.norm(np.matmul(A, u_curr)-np.matmul(B, v_curr))))
    if i >= min_it: 
        if np.square(np.linalg.norm(u_curr - u_prev)) <= min_d and np.square(np.linalg.norm(v_curr - v_prev)) <= min_d:
            print("stopping criteria satisfied at it {}".format(i))
            print("optimal value = {}".format(prob_vals[-1]))
            break
    
    u_prev_prev = u_prev
    u_prev = u_curr
    v_prev_prev = v_prev
    v_prev = v_curr

data_dict = {"time_list": time_list,
             "uv_list":uv_list,
             "prob_vals":prob_vals
            }

with open("data_plot/q2b_c_hull_data.pickle", "wb") as f:
    pickle.dump(data_dict, f)

u_opt = u_curr
v_opt = v_curr

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
x_lin = np.linspace(-5, 5, 500)
y_lin = (-x_lin*normal_vector[0][0] + gamma)/normal_vector[1][0]

fig = plt.figure(figsize=(8,8))
plt.scatter(A[0,:],A[1,:], alpha=0.5, color='g', label='Class A')
plt.scatter(B[0,:],B[1,:], alpha=0.5, color='r', label='Class B')

# plot optimal point
plt.scatter(A_opt[0], A_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b', label='support vectors')
plt.scatter(B_opt[0], B_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b')

# plot hyperplane
plt.plot(x_lin, y_lin, label='trained classifier', color='black')
plt.plot([A_opt[0], B_opt[0]], [A_opt[1], B_opt[1]], alpha=0.5, ls="dotted", label='normal vector', color='black')

plt.title("Problem 2B - Separable C-Hull - Plot training data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_aspect('equal')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
# plt.show()
plt.tight_layout()
fig.savefig("./figures/plot_2b_c_hull_training_data.png")

fig = plt.figure(figsize=(8,8))

plt.scatter(test[:,test_label==1][0,:],test[:,test_label==1][1,:], alpha=0.5, color='g', label='Class A')
plt.scatter(test[:,test_label==-1][0,:],test[:,test_label==-1][1,:], alpha=0.5, color='r', label='Class B')

# plot optimal point
plt.scatter(A_opt[0], A_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b', label='support vectors')
plt.scatter(B_opt[0], B_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b')

# plot hyperplane
plt.plot(x_lin, y_lin, label='trained classifier', color='black')
plt.plot([A_opt[0], B_opt[0]], [A_opt[1], B_opt[1]], alpha=0.5, ls="dotted", label='normal vector', color='black')

plt.title("Problem 2B - Separable C-Hull - Plot test data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_aspect('equal')
plt.xlim(-8, 8 )
plt.ylim(-8, 8 )
# plt.show()
plt.tight_layout()
fig.savefig("./figures/plot_2b_c_hull_testing_data.png")
#######################
### END CONVEX HULL ###
#######################


###########################
### REDUCED CONVEX HULL ###
###########################
print("\nWorking on Reduced Convex Hull")
data_train = scipy.io.loadmat("overlap_case/train_overlap.mat")
data_test = scipy.io.loadmat("overlap_case/test_overlap.mat")

A = data_train["A"]
B = data_train["B"]

test = data_test["X_test"]
test_label = data_test["true_labels"].ravel()

np.random.seed(42) # for reproducibility

u_0 = scipy.special.softmax(np.random.uniform(0, 1, (100, 1)))
v_0 = scipy.special.softmax(np.random.uniform(0, 1, (100, 1)))

u_prev = u_0
v_prev = v_0

alpha = 0.1
prob_vals = []
uv_list = []
time_list = []

min_d = 0.01 # stopping criteria
max_it = 100 # max number of iters
min_it = 20 # min number of iters

# calc first 2 
prob_vals.append(np.square(np.linalg.norm(np.matmul(A, u_prev_prev)-np.matmul(B, v_prev_prev))))
prob_vals.append(np.square(np.linalg.norm(np.matmul(A, u_prev)-np.matmul(B, v_prev))))

# projected gradient loop
for i in range(max_it): 
    start = time.time()
    t = ((i+1)/2-1)/((i+1+1)/2)
    u_t = (1+t)*u_prev - t*u_prev_prev
    u_curr_grad = calc_gradient_descent(u_t, v_prev, A, B, alpha=alpha)
    u_curr, _ = calc_projection_optimization(u_curr_grad, d=0.02)
    # u_curr, _ = cal_projection_prob_simplex_simple(u_curr_grad)
    # print(sum(u_curr))
    v_t = (1+t)*v_prev - t*v_prev_prev
    v_curr_grad = calc_gradient_descent(v_prev, u_prev, B, A, alpha=alpha) 
    v_curr, _ = calc_projection_optimization(v_curr_grad, d=0.02)
    # v_curr, _ = cal_projection_prob_simplex_simple(v_curr_grad)
    # print(sum(v_curr))
    end = time.time()
    # save time and current opt vectors
    time_list.append(end-start)
    uv_list.append([u_curr, v_curr])
    prob_vals.append(np.square(np.linalg.norm(np.matmul(A, u_curr)-np.matmul(B, v_curr))))
    if i >= min_it: 
        if np.square(np.linalg.norm(u_curr - u_prev)) <= min_d and np.square(np.linalg.norm(v_curr - v_prev)) <= min_d:
            print("stopping criteria satisfied at it {}".format(i))
            print("optimal value = {}".format(prob_vals[-1]))
            break
    
    u_prev_prev = u_prev
    u_prev = u_curr
    v_prev_prev = v_prev
    v_prev = v_curr
    
data_dict = {"time_list": time_list,
             "uv_list":uv_list,
             "prob_vals":prob_vals
            }

with open("data_plot/q2b_reduced_c_hull_data.pickle", "wb") as f:
    pickle.dump(data_dict, f)

u_opt = u_curr
v_opt = v_curr

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
x_lin = np.linspace(-5, 5, 500)
y_lin = (-x_lin*normal_vector[0][0] + gamma)/normal_vector[1][0]

fig = plt.figure(figsize=(8,8))
plt.scatter(A[0,:],A[1,:], alpha=0.5, color='g', label='Class A')
plt.scatter(B[0,:],B[1,:], alpha=0.5, color='r', label='Class B')

# plot optimal point
plt.scatter(A_opt[0], A_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b', label='support vectors')
plt.scatter(B_opt[0], B_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b')

# plot hyperplane
plt.plot(x_lin, y_lin, label='trained classifier', color='black')
plt.plot([A_opt[0], B_opt[0]], [A_opt[1], B_opt[1]], alpha=0.5, ls="dotted", label='normal vector', color='black')

plt.title("Problem 2B - Overlap C-Hull - Plot training data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_aspect('equal')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
# plt.show()
plt.tight_layout()
fig.savefig("./figures/plot_2b_reduced_c_hull_training_data.png")

fig = plt.figure(figsize=(8,8))

plt.scatter(test[:,test_label==1][0,:],test[:,test_label==1][1,:], alpha=0.5, color='g', label='Class A')
plt.scatter(test[:,test_label==-1][0,:],test[:,test_label==-1][1,:], alpha=0.5, color='r', label='Class B')

# plot optimal point
plt.scatter(A_opt[0], A_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b', label='support vectors')
plt.scatter(B_opt[0], B_opt[1], alpha=1, s=180, facecolors='none', edgecolors='b')

# plot hyperplane
plt.plot(x_lin, y_lin, label='trained classifier', color='black')
plt.plot([A_opt[0], B_opt[0]], [A_opt[1], B_opt[1]], alpha=0.5, ls="dotted", label='normal vector', color='black')

plt.title("Problem 2B - Overlap C-Hull - Plot test data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_aspect('equal')
plt.xlim(-8, 8 )
plt.ylim(-8, 8 )
# plt.show()
plt.tight_layout()
fig.savefig("./figures/plot_2b_reduced_c_hull_testing_data.png")
###############################
### END REDUCED CONVEX HULL ###
###############################