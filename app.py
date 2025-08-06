import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation
import pandas as pd
import os

# ---------------------- CONFIG MENU ----------------------
def parse_float(prompt, default):
    val = input(f"{prompt} (default {default}): ").strip()
    return float(val) if val else default

def parse_vector(prompt, default):
    val = input(f"{prompt} (default {','.join(map(str, default))}): ").strip()
    if val:
        try:
            parts = list(map(float, val.split(',')))
            if len(parts) != 3:
                raise ValueError
            return parts
        except:
            print("Invalid format. Expected: x,y,z")
            return default
    return default

def validate_positions(p1, p2, p3):
    if np.array_equal(p1, p2) or np.array_equal(p2, p3) or np.array_equal(p1, p3):
        print("⚠️ Warning: Two or more bodies have the same initial position. This may cause instability.")
        exit(1)

ani = None 

# ---------- Config Menu ----------
def run_config_menu():
    print("\n🎮 3-Body Simulator Config Menu")
    print("Press ENTER to use defaults, or type 'E' to edit settings.")
    choice = input("Your choice: ").strip().lower()

    if choice == 'e':
        m1 = parse_float("Mass of body 1", 1.0)
        m2 = parse_float("Mass of body 2", 3.0)
        m3 = parse_float("Mass of body 3", 1.0)

        pos1 = parse_vector("Body 1 position (x,y,z)", [1, 0, 0])
        pos2 = parse_vector("Body 2 position (x,y,z)", [0, 2, 0])
        pos3 = parse_vector("Body 3 position (x,y,z)", [0, 0, 3])

        vel1 = parse_vector("Body 1 velocity (vx,vy,vz)", [0, 1, 1])
        vel2 = parse_vector("Body 2 velocity (vx,vy,vz)", [0, 1, 0])
        vel3 = parse_vector("Body 3 velocity (vx,vy,vz)", [1, 1, 0])

        G = parse_float("Gravitational constant G", 1.0)
        axis_limit = parse_float("Axis limit for 3D plot", 10.0)

    else:
        # Defaults
        m1, m2, m3 = 1.0, 3.0, 1.0
        pos1 = [1, 0, 0]
        pos2 = [0, 2, 0]
        pos3 = [0, 0, 3]
        vel1 = [0, 1, 1]
        vel2 = [0, 1, 0]
        vel3 = [1, 1, 0]
        G = 1.0
        axis_limit = 10.0

    validate_positions(pos1, pos2, pos3)
    return m1, m2, m3, pos1, pos2, pos3, vel1, vel2, vel3, G, axis_limit


# ---------------------- MAIN SIMULATION ----------------------
def run_simulation(m1, m2, m3, pos1, pos2, pos3, vel1, vel2, vel3, G, axis_limit):
    data = []  # Reset for every run

    initial_conditions = np.array([pos1, pos2, pos3, vel1, vel2, vel3]).ravel()

    def system_odes(t, S, m1, m2, m3):
        p1, p2, p3 = S[0:3], S[3:6], S[6:9]
        v1, v2, v3 = S[9:12], S[12:15], S[15:18]

        def acceleration(pi, pj, mj):
            diff = pj - pi
            dist = np.linalg.norm(diff) + 1e-5  # Avoid div by zero
            return G * mj * diff / dist**3

        a1 = acceleration(p1, p2, m2) + acceleration(p1, p3, m3)
        a2 = acceleration(p2, p1, m1) + acceleration(p2, p3, m3)
        a3 = acceleration(p3, p1, m1) + acceleration(p3, p2, m2)

        return np.concatenate([v1, v2, v3, a1, a2, a3])

    # Time setup
    time_s, time_e = 0, 20
    t_points = np.linspace(time_s, time_e, 1000)

    solution = solve_ivp(system_odes, (time_s, time_e), initial_conditions, t_eval=t_points, args=(m1, m2, m3))
    positions = solution.y
    p1 = positions[0:3, :]
    p2 = positions[3:6, :]
    p3 = positions[6:9, :]

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    ax.set_title("3-Body Problem Animation")
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
        nonlocal ani 

        # Current positions
        x1_val, y1_val, z1_val = p1[0, i], p1[1, i], p1[2, i]
        x2_val, y2_val, z2_val = p2[0, i], p2[1, i], p2[2, i]
        x3_val, y3_val, z3_val = p3[0, i], p3[1, i], p3[2, i]

        # Check bounds
        def is_out_of_bounds(x, y, z):
            return abs(x) > axis_limit+5 or abs(y) > axis_limit+5 or abs(z) > axis_limit+5

        out1 = is_out_of_bounds(x1_val, y1_val, z1_val)
        out2 = is_out_of_bounds(x2_val, y2_val, z2_val)
        out3 = is_out_of_bounds(x3_val, y3_val, z3_val)

        # Stop animation if all 3 are out
        if out1 and out2 and out3:
           # print(f"⛔ Simulation stopped: all 3 bodies exceeded axis limits at time {time:.2f}s (frame {i})")
            dot1.set_alpha(0.0)
            dot2.set_alpha(0.0)
            dot3.set_alpha(0.0)
            if ani and ani.event_source:
                ani.event_source.stop()

            return dot1, dot2, dot3

        # Hide individual bodies if out of view
        dot1.set_alpha(0.0 if out1 else 1.0)
        dot2.set_alpha(0.0 if out2 else 1.0)
        dot3.set_alpha(0.0 if out3 else 1.0)

        # Trail update
        x1.append(x1_val); y1.append(y1_val); z1.append(z1_val)
        x2.append(x2_val); y2.append(y2_val); z2.append(z2_val)
        x3.append(x3_val); y3.append(y3_val); z3.append(z3_val)

        if len(x1) > trail_len:
            x1.pop(0); y1.pop(0); z1.pop(0)
            x2.pop(0); y2.pop(0); z2.pop(0)
            x3.pop(0); y3.pop(0); z3.pop(0)

        # Update dot positions
        dot1.set_data([x1_val], [y1_val])
        dot1.set_3d_properties([z1_val])

        dot2.set_data([x2_val], [y2_val])
        dot2.set_3d_properties([z2_val])

        dot3.set_data([x3_val], [y3_val])
        dot3.set_3d_properties([z3_val])

        # Save data
        data.append([i, time, 1, x1_val, y1_val, z1_val])
        data.append([i, time, 2, x2_val, y2_val, z2_val])
        data.append([i, time, 3, x3_val, y3_val, z3_val])

        return dot1, dot2, dot3



    ani = animation.FuncAnimation(fig, update, frames=range(0, p1.shape[1], 3), interval=10, blit=False)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    # Save animation as MP4
    print("💾 Saving animation to '3body_simulation.mp4'...")
    # Ensure 'media' folder exists
    os.makedirs("media", exist_ok=True)

    # Save animation in the 'media' folder
    ani.save("media/3body_simulation.mp4", writer="ffmpeg", fps=30)
    print("✅ Animation saved as 'media/3body_simulation.mp4'")

    # If you want a GIF instead, use this:
    # ani.save("3body_simulation.gif", writer="pillow", fps=15)

    # Save simulation data
    df = pd.DataFrame(data, columns=["frame", "time", "planet", "x", "y", "z"])
    df.to_csv("3body_simulation_output.csv", index=False)
    print("📦 Simulation data saved to '3body_simulation_output.csv'")



# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    config = run_config_menu()
    run_simulation(*config)
