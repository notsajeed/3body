import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation
import pandas as pd

# ---------- Input Utilities ----------
def parse_float(prompt, default):
    try:
        val = input(f"{prompt} (default {default}): ").strip()
        return float(val) if val else default
    except ValueError:
        print("⚠️ Invalid input. Using default.")
        return default

def parse_vector(prompt, default):
    val = input(f"{prompt} (default {','.join(map(str, default))}): ").strip()
    if not val:
        return default
    try:
        parts = list(map(float, val.split(',')))
        if len(parts) != 3:
            raise ValueError
        return parts
    except:
        print("⚠️ Invalid vector input. Using default.")
        return default

def validate_positions(p1, p2, p3):
    if np.array_equal(p1, p2) or np.array_equal(p2, p3) or np.array_equal(p1, p3):
        print("⚠️ Warning: Two or more bodies have the same initial position. This may cause instability.")

# ---------- Config Menu ----------
def run_config_menu():
    print("🎮 3-Body Simulator Config Menu")
    print("Press ENTER to use defaults, or type 'E' to edit settings.")
    choice = input("Your choice: ").strip().lower()

    if choice == 'e':
        m1 = parse_float("Mass of body 1", 1.0)
        m2 = parse_float("Mass of body 2", 5.0)
        m3 = parse_float("Mass of body 3", 1.0)

        pos1 = parse_vector("Body 1 position (x,y,z)", [1, 0, 0])
        pos2 = parse_vector("Body 2 position (x,y,z)", [0, 1, 0])
        pos3 = parse_vector("Body 3 position (x,y,z)", [0, 0, 1])

        vel1 = parse_vector("Body 1 velocity (vx,vy,vz)", [1, 0, 0])
        vel2 = parse_vector("Body 2 velocity (vx,vy,vz)", [0, 1, 0])
        vel3 = parse_vector("Body 3 velocity (vx,vy,vz)", [0, 0, 1])

        G = parse_float("Gravitational constant G", 1.0)
    else:
        m1, m2, m3 = 1.0, 5.0, 1.0
        pos1 = [1, 0, 0]
        pos2 = [0, 1, 0]
        pos3 = [0, 0, 1]
        vel1 = [1, 0, 0]
        vel2 = [0, 1, 0]
        vel3 = [0, 0, 1]
        G = 1.0

    validate_positions(pos1, pos2, pos3)
    return m1, m2, m3, pos1, pos2, pos3, vel1, vel2, vel3, G

# ---------- Simulation ----------
def run_simulation(m1, m2, m3, pos1, pos2, pos3, vel1, vel2, vel3, G):
    data = []

    # Initial state
    initial_conditions = np.array([pos1, pos2, pos3, vel1, vel2, vel3]).ravel()

    # Define the ODE system
    def system_odes(t, S, m1, m2, m3):
        p1, p2, p3 = S[0:3], S[3:6], S[6:9]
        v1, v2, v3 = S[9:12], S[12:15], S[15:18]

        def acceleration(pi, pj, mj):
            diff = pj - pi
            dist = np.linalg.norm(diff)
            if dist == 0: return np.zeros(3)
            return G * mj * diff / dist**3

        a1 = acceleration(p1, p2, m2) + acceleration(p1, p3, m3)
        a2 = acceleration(p2, p1, m1) + acceleration(p2, p3, m3)
        a3 = acceleration(p3, p1, m1) + acceleration(p3, p2, m2)

        return np.concatenate([v1, v2, v3, a1, a2, a3])

    # Time setup
    time_s, time_e = 0, 20
    t_points = np.linspace(time_s, time_e, 1000)

    # Solve
    solution = solve_ivp(system_odes, (time_s, time_e), initial_conditions,
                         t_eval=t_points, args=(m1, m2, m3))
    positions = solution.y
    p1 = positions[0:3, :]
    p2 = positions[3:6, :]
    p3 = positions[6:9, :]

    # Setup plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_title("3-Body Simulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    dot1, = ax.plot([], [], [], 'go', label='Body 1')
    dot2, = ax.plot([], [], [], 'ro', label='Body 2')
    dot3, = ax.plot([], [], [], 'bo', label='Body 3')

    trail_len = 100
    x1, y1, z1 = [], [], []
    x2, y2, z2 = [], [], []
    x3, y3, z3 = [], [], []

    def update(frame):
        i = frame
        time = t_points[i]

        for p in (p1, p2, p3):
            if np.any(np.abs(p[:, i]) > 3):
                x1.clear(); y1.clear(); z1.clear()
                x2.clear(); y2.clear(); z2.clear()
                x3.clear(); y3.clear(); z3.clear()
                break

        x1.append(p1[0, i]); y1.append(p1[1, i]); z1.append(p1[2, i])
        x2.append(p2[0, i]); y2.append(p2[1, i]); z2.append(p2[2, i])
        x3.append(p3[0, i]); y3.append(p3[1, i]); z3.append(p3[2, i])

        if len(x1) > trail_len:
            x1.pop(0); y1.pop(0); z1.pop(0)
            x2.pop(0); y2.pop(0); z2.pop(0)
            x3.pop(0); y3.pop(0); z3.pop(0)

        dot1.set_data([p1[0, i]], [p1[1, i]])
        dot1.set_3d_properties([p1[2, i]])

        dot2.set_data([p2[0, i]], [p2[1, i]])
        dot2.set_3d_properties([p2[2, i]])

        dot3.set_data([p3[0, i]], [p3[1, i]])
        dot3.set_3d_properties([p3[2, i]])

        data.append([i, time, 1, p1[0, i], p1[1, i], p1[2, i]])
        data.append([i, time, 2, p2[0, i], p2[1, i], p2[2, i]])
        data.append([i, time, 3, p3[0, i], p3[1, i], p3[2, i]])

        return dot1, dot2, dot3

    ani = animation.FuncAnimation(fig, update, frames=range(0, p1.shape[1], 3), interval=10, blit=False)

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Export data
    df = pd.DataFrame(data, columns=["frame", "time", "planet", "x", "y", "z"])
    df.to_csv("3body_simulation_output.csv", index=False)
    print("Simulation complete. Data saved to '3body_simulation_output.csv'.")

# ---------- Run ----------
if __name__ == "__main__":
    m1, m2, m3, pos1, pos2, pos3, vel1, vel2, vel3, G = run_config_menu()
    run_simulation(m1, m2, m3, pos1, pos2, pos3, vel1, vel2, vel3, G)
