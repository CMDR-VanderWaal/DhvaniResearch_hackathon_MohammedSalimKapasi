import numpy as np
import matplotlib.pyplot as plt

def given_equations(x, y, z, a, b, c):
    dx_dt = a * (y - x)
    dy_dt = b * x - y - x * z
    dz_dt = x * y - c * z
    return dx_dt, dy_dt, dz_dt

# Defined Constant parameters
a = 10.0
b = 28.0
c = 2.667

# Starting position of the bee
x = 0.0
y = 1.0
z = 1.05


dt = 0.01 #time step
num_steps = 10000 #number of steps to simulate

# Lists cordinates
x_values = []
y_values = []
z_values = []


for i in range(num_steps):

    dx, dy, dz = given_equations(x, y, z, a, b, c)

    # Update positions
    x = x + dx * dt
    y = y + dy * dt
    z = z + dz * dt

    # Save posotion cordinates
    x_values.append(x)
    y_values.append(y)
    z_values.append(z)

#PLOTTING:-
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d') 

ax.plot(x_values, y_values, z_values, color='cyan', linewidth=0.5)


ax.set_title('Plotted Path')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

# Show the plot
plt.show()
