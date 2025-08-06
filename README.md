# 🌌 3-Body Gravitational Simulation

A Python-based 3D visualization of the **three-body problem** using `matplotlib`, `scipy`, and `numpy`. This project is a physics-inspired simulation that visually demonstrates the chaotic dynamics of three mutually gravitating bodies in space.

> 🎯 Focus: Educational simulation with real-time animation and configurable parameters.

---

## 📽️ Demo

![simulation-demo](./media/3body_simulation.mp4)

> _(Insert gif/mp4 or link to YouTube or GitHub media folder)_

---

## ✨ Features

- 🔧 Interactive CLI config menu (customize masses, positions, velocities, `G`, and more)
- 🎥 Smooth 3D animation using `matplotlib.animation`
- 📦 Saves simulation data as CSV (`3body_simulation_output.csv`)
- 🎬 Exports animation as MP4 using FFmpeg
- 🧠 Stop condition when all bodies leave the viewport
- 🚫 Hides out-of-bound bodies gracefully to avoid clutter
- 📈 Easy to analyze trajectory data post-simulation

---

## 📦 Requirements

- Python 3.7+
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `ffmpeg` (for saving MP4)

### Install dependencies:

```bash
pip install numpy scipy matplotlib pandas
```

### Install FFmpeg (for saving animation):

- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to PATH
- **Linux/macOS:**

```bash
sudo apt install ffmpeg   # Debian/Ubuntu
brew install ffmpeg       # macOS (Homebrew)
```

---

## 🚀 Run the Simulation

```bash
python 3body_simulation.py
```

You'll see a CLI menu:

- Press **Enter** to use defaults
- Type **E** to edit custom values like mass, velocity, position, `G`, etc.

After the simulation:

- You’ll get an **MP4 animation**
- All positional data will be saved to a **CSV** for further analysis

---

## 📊 Optional Data Plotting (Post-Simulation)

To analyze saved trajectories:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("3body_simulation_output.csv")

for planet_id in [1, 2, 3]:
    subset = df[df['planet'] == planet_id]
    plt.plot(subset['x'], subset['y'], label=f'Body {planet_id}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title("2D Trajectory Plot (XY Plane)")
plt.legend()
plt.grid(True)
plt.show()
```

---

## ⚠️ Known Limitations and Inaccuracies

This simulator prioritizes **visual understanding** and **usability** over exact scientific accuracy.

### 🔬 Physics & Math Simplifications:

- **No collisions**: Bodies are point masses with no radius or collision detection
- **No energy tracking**: System energy is not computed or conserved
- **Simplified gravity**: A small constant (`+1e-5`) is added in the denominator to avoid division by zero
- **No adaptive time step**: Fixed time sampling via `np.linspace` is used
- **Newtonian only**: No general relativity, relativistic corrections, or external forces

### 🎥 Rendering/Animation:

- Bodies are still rendered briefly after leaving axis limits for smoother transition
- Simulation stops **only after all 3 bodies exit bounds**, not individually

---

## 🛠️ To-Do / Future Enhancements

- [ ] Add energy and angular momentum conservation metrics
- [ ] Enable real-world unit support (e.g., AU, kg, seconds)
- [ ] Collision detection or merging logic
- [ ] Symplectic integrators (Leapfrog/Euler-Cromer) for better long-term stability
- [ ] Web-based interface using WebGL (maybe Three.js)

---

## 📁 Project Structure

```
.
├── 3body_simulation.py            # Main simulation code
├── 3body_simulation.mp4           # Output animation (auto-generated)
├── 3body_simulation_output.csv    # Trajectory data (auto-generated)
├── media/
│   └── demo.gif                   # Optional: gif for demo purposes
└── README.md                      # This file
```

---

## 💡 Inspiration

The chaotic nature of the 3-body problem is a beautiful representation of how **simple rules can lead to complex behavior**. This simulation is my take on exploring it visually, interactively, and accessibly.

---

## 🤝 License

This project is free to use under the **MIT License**.
Feel free to fork, improve, or contribute back!

---

## 🙌 Acknowledgments

- `scipy.integrate.solve_ivp` — for solving the ODE system
- `matplotlib.animation` — for powerful visual rendering
- Community contributors and educators for inspiring open-source learning

---

Let me know if you'd like:

- 🌐 A version styled for a portfolio website
- 🧩 A GUI version (Tkinter, PyQt, or Web)
- 🧪 Jupyter notebook version
- 🛠️ Modular version for reusability or educational kits
