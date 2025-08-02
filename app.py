import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation

# Masses
m1 = 1.0
m2 = 1.0
m3 = 1.0

# Initial positions
initial_position_1 = [1.0, 0.0, 0.0]
initial_position_2 = [0.0, 1.0, 0.0]
initial_position_3 = [0.0, 0.0, 1.0]

# Initial velocities
initial_velocity_1 = [1.0, 0.0, 0.0]
initial_velocity_2 = [0.0, 1.0, 0.0]
initial_velocity_3 = [0.0, 0.0, 1.0]

# Combine all into a flat array
initial_conditions = np.array([
    initial_position_1, initial_position_2, initial_position_3,
    initial_velocity_1, initial_velocity_2, initial_velocity_3
]).ravel()

# ODE function
def system_odes(t, S, m1, m2, m3):
    p1, p2, p3 = S[0:3], S[3:6], S[6:9]
    v1, v2, v3 = S[9:12], S[12:15], S[15:18]

    def acceleration(pi, pj, mj):
        diff = pj - pi
        return mj * diff / np.linalg.norm(diff)**3

    a1 = acceleration(p1, p2, m2) + acceleration(p1, p3, m3)
    a2 = acceleration(p2, p1, m1) + acceleration(p2, p3, m3)
    a3 = acceleration(p3, p1, m1) + acceleration(p3, p2, m2)

    return np.concatenate([v1, v2, v3, a1, a2, a3])

# Time setup
time_s, time_e = 0, 20
t_points = np.linspace(time_s, time_e, 1000)

# Solve the ODE
solution = solve_ivp(
    fun=system_odes,
    t_span=(time_s, time_e),
    y0=initial_conditions,
    t_eval=t_points,
    args=(m1, m2, m3)
)

# Extract positions
positions = solution.y
p1 = positions[0:3, :]  # x, y, z of body 1
p2 = positions[3:6, :]  # body 2
p3 = positions[6:9, :]  # body 3


# --- ANIMATION ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_title("3-Body Problem Animation")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Dot for each body
dot1, = ax.plot([], [], [], 'go', label='Planet 1')
dot2, = ax.plot([], [], [], 'ro', label='Planet 2')
dot3, = ax.plot([], [], [], 'bo', label='Planet 3')

#Trails

# Trail buffers
trail_len = 100
x1, y1, z1 = [], [], []
x2, y2, z2 = [], [], []
x3, y3, z3 = [], [], []

def update(frame):
    i = frame

    # --- ESCAPE CHECK ---
    for p in (p1, p2, p3):
        if np.any(np.abs(p[:, i]) > 3):
            x1.clear(); y1.clear(); z1.clear()
            x2.clear(); y2.clear(); z2.clear()
            x3.clear(); y3.clear(); z3.clear()
            break


    # Update trails
    x1.append(p1[0, i]); y1.append(p1[1, i]); z1.append(p1[2, i])
    x2.append(p2[0, i]); y2.append(p2[1, i]); z2.append(p2[2, i])
    x3.append(p3[0, i]); y3.append(p3[1, i]); z3.append(p3[2, i])

    # Limit trail length
    if len(x1) > trail_len:
        x1.pop(0); y1.pop(0); z1.pop(0)
        x2.pop(0); y2.pop(0); z2.pop(0)
        x3.pop(0); y3.pop(0); z3.pop(0)

    # Update dots
    dot1.set_data([p1[0, i]], [p1[1, i]])
    dot1.set_3d_properties([p1[2, i]])

    dot2.set_data([p2[0, i]], [p2[1, i]])
    dot2.set_3d_properties([p2[2, i]])

    dot3.set_data([p3[0, i]], [p3[1, i]])
    dot3.set_3d_properties([p3[2, i]])


    return dot1, dot2, dot3


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=range(0, p1.shape[1], 3), interval=10, blit=False)

plt.legend()
plt.tight_layout()
plt.show() 