import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.animation import FuncAnimation
import time

# Define initial parameters
D = 1.66e-4  # Silver diffusion coefficient
Lx = 0.3  # Length of the plate in the x-direction
Ly = 0.3  # Length of the plate in the y-direction

nx = 60  # Number of steps in the x-direction
ny = 60  # Number of steps in the y-direction
dt = 0.0005  # Time step
tf = 20  # Final time
t1 = time.time()

# Calculate spatial steps
dx = Lx / nx
dy = Ly / ny

alpha = D * dt / dx**2  # Abbreviation

r = D * (dt * dx**2 + dt * dy**2) / (dx**2 * dy**2)
if r > 0.5:
    raise TypeError('Unstable solution')
else:
    print(f"dx: {dx} \n dy: {dy}")
    print(f"{dt}")

# Boundary conditions
def boundary_dirichlet():
    return 0

def boundary_neumann(x, y, t, T):
    if x != 0 and x != nx - 1:
        if y == 0:
            return T[x, 1, t]
        else:
            return T[x, y - 1, t]
    else:
        if x == 0:
            return T[x + 1, y, t]
        else:
            return T[x - 1, y, t]

# Initial conditions
def initial_constant():
    return 100

def initial_circle(x, y):
    r = np.sqrt(pow(x - 0.15 / dx, 2) + pow(y - 0.15 / dy, 2))
    if r < 0.08 / dx:
        return 100
    else:
        return 0

# Initial conditions - 45-degree circular cut
def initial_45_degree_cut(x, y):
    r = np.sqrt(pow(x - 0.15 / dx, 2) + pow(y - 0.15 / dy, 2))
    angle = np.arctan2(y - 0.15 / dy, x - 0.15 / dx)
    if r < 0.13 / dx and -np.pi / 4 <= angle <= np.pi / 4:
        return 100
    else:
        return 0

def initial_antisymmetric(x, y):
    if (0.05 / dx < x < 0.1 / dx and 0.1 / dy < y < 0.3 / dy) or (0.2 / dx < x < 0.25 / dx and 0.15 / dy < y < 0.2 / dy):
        return 50
    elif (0.1 / dx < x < 0.15 / dx and 0.2 / dy < y < 0.25 / dy) or (0.3 / dx < x < 0.35 / dx and 0.05 / dy < y < 0.1 / dy):
        return 75
    elif (0.1 / dx < x < 0.25 / dx and 0.1 / dy < y < 0.2 / dy) or (0.4 / dx < x < 0.45 / dx and 0.25 / dy < y < 0.3 / dy):
        return 100
    else:
        return 30

# 3D array to store data
T = np.zeros((nx, ny, int(tf / dt)))

# Set initial conditions
for i in range(0, nx - 1):
    for j in range(1, ny - 1):
        T[i, j, 0] = initial_constant()

# Set boundary conditions
def set_boundary_neumann(t):
    for i in range(0, nx):
        T[i, 0, t] = boundary_neumann(i, 0, t, T)
        T[i, ny - 1, t] = boundary_neumann(i, ny - 1, t, T)

    for i in range(0, ny):
        T[0, i, t] = boundary_neumann(0, i, t, T)
        T[nx - 1, i, t] = boundary_neumann(nx - 1, i, t, T)

def set_boundary_dirichlet(t):
    for i in range(0, nx):
        T[i, 0, t] = boundary_dirichlet()
        T[i, ny - 1, t] = boundary_dirichlet()

    for i in range(0, ny):
        T[0, i, t] = boundary_dirichlet()
        T[nx - 1, i, t] = boundary_dirichlet()

vri1 = time.time()
for t in range(0, int(tf / dt) - 1):
    # set_boundary_neumann(t)
    set_boundary_dirichlet(t)
    for i in range(1, (nx - 1)):
        for j in range(1, (ny - 1)):                
            T[i, j, t + 1] = alpha * (T[i + 1, j, t] - 4 * T[i, j, t] + T[i - 1, j, t] + T[i, j + 1, t] + T[i, j - 1, t]) + T[i, j, t]

set_boundary_neumann(int(tf / dt) - 1)
vri2 = time.time()
print(vri2 - vri1)

def heatmap2d(arr, t):
    time_step = arr[:, :, t]

    # Swap array dimensions manually
    time_step = np.transpose(time_step)

    plt.imshow(time_step, cmap='jet', vmin=np.min(arr), vmax=np.max(arr))
    plt.colorbar(label='Temperature')
    plt.xlabel('X[nx]')
    plt.ylabel('Y[ny]')
    if t != 0:
        plt.title(f'Temperature map for t = {int((t + 1) * dt)} s')
    else:
        plt.title(f'Temperature map for t = 0 s')

# Display 4 images for different time steps
plt.figure(figsize=(12, 8))

for i, t in enumerate([0, 3, 6, 10]):
    plt.subplot(2, 2, i + 1)
    if t != 0:
        heatmap2d(T, int(t / dt) - 1)
    else:
        heatmap2d(T, t)
plt.tight_layout()
plt.savefig("Temp_map_antisymmetry.png")
plt.show()

fig, ax = plt.subplots()
heatmap = ax.imshow(T[:, :, 0], cmap='jet', vmin=0, vmax=100, extent=[0, Lx, 0, Ly])
plt.colorbar(heatmap)

# Set the title using ax.text
text_x = Lx * 0.75  # X coordinate
text_y = Ly * 1.05  # Y coordinate
title = ax.text(text_x, text_y, "", fontsize=12, ha='right', va='center')

def animate(k):
    heatmap.set_data(T[:, :, k * 200].T)  # Swap x and y coordinates
    title.set_text(f"Time t = {k * 200 * dt:.3f} s")  # Properly update the title
    return heatmap, title

anim = FuncAnimation(fig, animate, frames=int(tf / (dt * 200)), blit=True)
anim.save("heat_equation-Dirichlet_boundary_conditions.gif", writer='pillow', fps=100)

