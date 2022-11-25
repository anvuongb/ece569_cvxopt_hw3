import numpy as np
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import pickle

# Plot C-Hull
with open("data_plot/q2a_c_hull_data.pickle", "rb") as f:
    data = pickle.load(f)

time_list_projected_grad = data["time_list"]
prob_vals_projected_grad = data["prob_vals"]

with open("data_plot/q2b_c_hull_data.pickle", "rb") as f:
    data = pickle.load(f)

time_list_projected_grad_nesterov = data["time_list"]
prob_vals_projected_grad_nesterov = data["prob_vals"]

with open("data_plot/q3_c_hull_data.pickle", "rb") as f:
    data = pickle.load(f)

time_list_admm = data["time_list"]
prob_vals_admm = data["prob_vals"]

fig = plt.figure(figsize=(7,5))

plt.plot(range(1, len(prob_vals_admm)+1), prob_vals_admm, marker="x", label="admm")
plt.plot(range(1, len(prob_vals_projected_grad)+1), prob_vals_projected_grad, marker="x", label="projected gradient")
plt.plot(range(1, len(prob_vals_projected_grad_nesterov)+1), prob_vals_projected_grad_nesterov, marker="x", c="m", label="projected gradient + Nesterov accel")

plt.legend()
plt.xticks([1,5,10,15,20])
plt.ylabel("objective value")
plt.xlabel("#iteration")
plt.title("Problem 4 - C-Hull - Objective value vs. Iteration")
# plt.show()
plt.tight_layout()
plt.savefig("figures/q4_chull_iter.png")

fig = plt.figure(figsize=(7,5))

plt.plot(np.cumsum(time_list_admm + [0.01]), prob_vals_admm+[prob_vals_admm[-1]], marker="x", label="admm")
plt.plot(np.cumsum([0] + time_list_projected_grad), prob_vals_projected_grad, marker="x", label="projected gradient")
plt.plot(np.cumsum([0,0] + time_list_projected_grad_nesterov), prob_vals_projected_grad_nesterov, c="m", marker="x", label="projected gradient + Nesterov accel")

plt.legend()
# plt.xticks([1,5,10,15,20])
plt.xlim(0,0.012)
plt.ylabel("objective value")
plt.xlabel("#time (seconds)")
plt.title("Problem 4 - C-Hull - Objective value vs. Time")
# plt.show()
plt.tight_layout()
plt.savefig("figures/q4_chull_time.png")



# Plot RC-Hull
with open("data_plot/q2a_reduced_c_hull_data.pickle", "rb") as f:
    data = pickle.load(f)

time_list_projected_grad = data["time_list"]
prob_vals_projected_grad = data["prob_vals"]

with open("data_plot/q2b_reduced_c_hull_data.pickle", "rb") as f:
    data = pickle.load(f)

time_list_projected_grad_nesterov = data["time_list"]
prob_vals_projected_grad_nesterov = data["prob_vals"]

with open("data_plot/q3_reduced_c_hull_data.pickle", "rb") as f:
    data = pickle.load(f)

time_list_admm = data["time_list"]
prob_vals_admm = data["prob_vals"]

fig = plt.figure(figsize=(7,5))

plt.plot(range(1, len(prob_vals_admm)+1), prob_vals_admm, marker="x", label="admm")
plt.plot(range(1, len(prob_vals_projected_grad)+1), prob_vals_projected_grad, marker="x", label="projected gradient")
plt.plot(range(1, len(prob_vals_projected_grad_nesterov)), prob_vals_projected_grad_nesterov[1:], marker="x", c="m", label="projected gradient + Nesterov accel")

plt.legend()
plt.xticks([1,5,10,15,20])
plt.ylabel("objective value")
plt.xlabel("#iteration")
plt.title("Problem 4 - Reduced C-Hull - Objective value vs. Iteration")
# plt.show()
plt.tight_layout()
plt.savefig("figures/q4_rchull_iter.png")

fig = plt.figure(figsize=(7,5))

plt.plot(np.cumsum([0] + time_list_admm + [0.8]), [prob_vals_admm[0]] + prob_vals_admm+[prob_vals_admm[-1]], marker="x", label="admm")
plt.plot(np.cumsum([0] + time_list_projected_grad), prob_vals_projected_grad, marker="x", label="projected gradient")
plt.plot(np.cumsum([0] + time_list_projected_grad_nesterov), prob_vals_projected_grad_nesterov[1:], c="m", marker="x", label="projected gradient + Nesterov accel")

plt.legend()
# plt.xticks([1,5,10,15,20])
plt.xlim(-0.05,0.8)
plt.ylabel("objective value")
plt.xlabel("#time (seconds)")
plt.title("Problem 4 - Reduced C-Hull -Objective value vs. Time")
# plt.show()
plt.tight_layout()
plt.savefig("figures/q4_rchull_time.png")