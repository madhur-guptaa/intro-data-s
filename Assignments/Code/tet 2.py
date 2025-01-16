import numpy as np

# Define the vectors of U
u1 = np.array([1, 2, 1])
u2 = np.array([1, 1, 0])
u3 = np.array([0, 1, 3])

# Define the vectors of W
v1 = np.array([0, 1, 1])
v2 = np.array([2, 5, 1])
v3 = np.array([1, 0, 1])

# Form the matrix U (columns as u1, u2, u3)
U = np.column_stack([u1, u2, u3])

# Solve for the transition matrix P_{W->U} by finding the coefficients for each v_i in terms of U
P_W_to_U = np.linalg.solve(U, np.column_stack([v1, v2, v3]))
print(P_W_to_U)