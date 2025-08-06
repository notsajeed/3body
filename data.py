import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load CSV
df = pd.read_csv("3body_simulation_output.csv")

def plot_3d_trajectories():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3-Body Trajectories")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    for pid, color, label in zip([1, 2, 3], ['g', 'r', 'b'], ['Body 1', 'Body 2', 'Body 3']):
        body = df[df['planet'] == pid]
        ax.plot(body['x'], body['y'], body['z'], color=color, label=label)

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_speed_vs_time():
    df['speed'] = df.groupby('planet')[['x', 'y', 'z']].diff().pow(2).sum(axis=1).pow(0.5)

    plt.figure()
    for pid, color, label in zip([1, 2, 3], ['g', 'r', 'b'], ['Body 1', 'Body 2', 'Body 3']):
        body = df[df['planet'] == pid]
        plt.plot(body['time'], body['speed'], color=color, label=label)

    plt.title("Speed vs Time")
    plt.xlabel("Time"); plt.ylabel("Speed")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_distance_between_bodies():
    d12 = compute_distance(1, 2)
    d13 = compute_distance(1, 3)
    d23 = compute_distance(2, 3)

    time = df[df['planet'] == 1]['time']
    plt.plot(time, d12, label="Distance: Body 1 ↔ 2")
    plt.plot(time, d13, label="Distance: Body 1 ↔ 3")
    plt.plot(time, d23, label="Distance: Body 2 ↔ 3")

    plt.title("Pairwise Distance Over Time")
    plt.xlabel("Time"); plt.ylabel("Distance")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_distance(p1, p2):
    df1 = df[df['planet'] == p1].reset_index()
    df2 = df[df['planet'] == p2].reset_index()
    return np.linalg.norm(df1[['x', 'y', 'z']] - df2[['x', 'y', 'z']], axis=1)

def menu():
    while True:
        print("\n🔍 3-Body Simulation Analyzer")
        print("1. 🌀 View 3D Trajectories")
        print("2. 🚀 Speed vs Time")
        print("3. 📏 Distance Between Bodies")
        print("4. ❌ Exit")

        choice = input("Choose an option (1-4): ").strip()

        if choice == '1':
            plot_3d_trajectories()
        elif choice == '2':
            plot_speed_vs_time()
        elif choice == '3':
            plot_distance_between_bodies()
        elif choice == '4':
            print("👋 Exiting.")
            break
        else:
            print("❌ Invalid input. Try again.")

# Run it
if __name__ == "__main__":
    menu()
