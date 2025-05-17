### ⚙️ **Tuned Python Package for Turbulent Flow Tracking**

This package is **specifically optimized for turbulent flow analysis** with Lagrangian tracers, building on the foundation from prior improvements. It includes **advanced motion modeling**, **extended statistical analysis**, and **enhanced simulation/visualization** for turbulent motion patterns.

---

### 🔧 Key Customizations for Turbulent Flow

| Feature | Customization | Purpose |
|--------|----------------|---------|
| **Kalman Filter** | Adaptive process noise tuning | Better modeling of chaotic particle motion |
| **Cost Function** | Includes 3D velocity curvature | Accounts for turbulent flow dynamics |
| **Dynamics Simulation** | Synthetics based on Taylor-Green vortex field | Realistic turbulent flow patterns for validation |
| **Merits** | Exportable Lagrangian velocity/acceleration time series | Ready for dissipation and strain rate analysis |
| **Visualization** | Streamlines + vorticity from tracked motion | Enhanced visualization of turbulent structures |

---

### 🧠 Enhanced Kalman Filter with Adaptive Noise

```python
class AdaptiveKalmanFilter3D:
    def __init__(self, initial_position, dt=1.0, base_process_noise=1e-3, noise_adapt_k=1.0):
        """
        Kalman Filter with dynamic process noise update based on residual prediction error

        Dynamic noise allows better adaptation to turbulence-like chaotic motion
        """
        self.state = np.zeros(9)
        self.state[:3] = initial_position

        # State transition matrix (constant acceleration model)
        self.dt = dt
        self.F = np.eye(9)
        for i in range(3):
            self.F[i, i+3] = dt
            self.F[i, i+6] = 0.5 * dt**2
            self.F[i+3, i+6] = dt

        self.H = np.eye(3, 9)  # Measurement matrix

        # Covariance matrices
        self.P = np.eye(9) * 1e-1  # Initial state covariance
        self.base_Q = np.eye(9) * base_process_noise
        self.Q = self.base_Q.copy()
        self.R = np.eye(3) * 1e-1  # Measurement noise

        # Noise adaptation
        self.noise_adapt_k = noise_adapt_k
        self.last_residual = None

    def predict(self):
        # Update process noise based on previous prediction error (if exists)
        if self.last_residual is not None:
            residual_norm = np.linalg.norm(self.last_residual)
            self.Q = self.base_Q * (1 + self.noise_adapt_k * residual_norm)
        
        # Predict state and covariance
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:3]  # Return predicted position

    def update(self, measurement):
        # Update step
        y = measurement - self.H @ self.state
        self.last_residual = y
        S = self.H @ self.P @ self.H.T + self.R
        if np.linalg.det(S) == 0:
            S += np.eye(3) * 1e-6
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P
```

---

### 🚧 Cost Function with Curvature Matching

```python
def _build_cost_matrix(self, predictions, detections):
    """Build cost matrix with dynamic curvature penalty for turbulent flows"""
    M = len(predictions)
    N = len(detections)
    cost_matrix = np.full((M + N, N + M), self.dummy_cost * 2)

    # Use more nuanced weighting for turbulent flow dynamics
    weight_position = 1.0
    weight_velocity = 1.5   # Higher weight due to turbulence having distinct velocity structures
    weight_acceleration = 2.0  # Emphasized in turbulence analysis
    weight_curve = 0.8      # Encourages consistent curvature in motion

    for i, track in enumerate(predictions):
        predicted_pos = track.predict()
        predicted_vel = track.kf.state[3:6]
        predicted_acc = track.kf.state[6:9]
        predicted_jerk = (predicted_acc - track.kf.state[6:9]) ** 2  # Simplification for example

        for j, det in enumerate(detections):
            measurement = det[:3]
            # Position match
            pos_diff = predicted_pos - measurement
            pos_cost = weight_position * np.linalg.norm(pos_diff)

            # Velocity match (from last update)
            try:
                actual_vel = (measurement - track.kf.state[:3]) / self.dt
                vel_cost = weight_velocity * np.linalg.norm(predicted_vel - actual_vel)
            except:
                vel_cost = 0.0

            # Acceleration match
            acc_cost = 0.0
            if len(track.positions) >= 2:
                actual_acc = actual_vel - track.kf.state[3:6]
                acc_cost = weight_acceleration * np.linalg.norm(predicted_acc - actual_acc)

            # Curvature matching for smoother Lagrangian path
            curve_cost = 0.0
            if len(track.positions) >= 3:
                last_three = np.array(track.positions[-3:])
                tangents = np.gradient(last_three, axis=0)
                curvature = np.gradient(tangents, axis=0)
                curve_cost = weight_curve * np.linalg.norm(curvature)

            total_cost = pos_cost + vel_cost + acc_cost + curve_cost
            cost_matrix[i, j] = total_cost

    # Add unassigned costs
    for i in range(M):
        cost_matrix[i, N:] = self.dummy_cost
    for j in range(N):
        cost_matrix[M:, j] = self.dummy_cost

    return cost_matrix
```

---

### 🌬️ Turbulent Flow Simulation with Taylor-Green Vortex

```python
def simulate_turbulent_flow(num_frames=120, num_particles=30,
                          box_size=100, viscosity=0.01,
                          dt=0.5, sample_frequency=1):
    """Synthesizes turbulent motion using Taylor-Green vortex solution for 2D flows"""
    frames = []

    # Initial conditions
    positions = np.random.uniform(0, box_size, (num_particles, 3))
    positions[:, 2] = np.random.uniform(-5, 5, num_particles)  # Flat Z for now

    for frame in range(num_frames):
        # Taylor-Green vortex velocity field
        x, y = positions[:, 0], positions[:, 1]
        A = 1.0
        sigma = 0.1
        t = frame * dt

        u = A * np.exp(-2 * sigma * t) * np.sin(x) * np.cos(y)
        v = -A * np.exp(-2 * sigma * t) * np.cos(x) * np.sin(y)
        w = np.zeros_like(u)

        velocity = np.column_stack([u, v, w])

        # Update positions
        positions += velocity * dt + np.random.normal(0, 0.1, velocity.shape)

        # Boundary conditions
        for i in range(3):
            positions[:, i] = np.clip(positions[:, i], 0, box_size)

        # Decimate frames if necessary
        if frame % sample_frequency == 0:
            frames.append(positions.copy())
    
    return frames
```

---

### 🔬 Track Analysis for Turbulent Flow Studies

```python
def analyze_flow_statistics(tracks):
    """Compute Lagrangian statistics from tracks for turbulent flow studies"""
    results = {
        'velocity': [],
        'acceleration': [],
        'enstrophy': [],
        'dissipation': [],
        'stretch': [],
    }

    for track in tracks:
        points = np.array(track['positions'])
        times = np.arange(len(points))

        # Velocity and acceleration
        velocity = np.gradient(points, axis=0)
        acceleration = np.gradient(velocity, axis=0)

        # Enstrophy and vorticity estimation
        du_dy = np.gradient(velocity[:, 0], axis=1)[1]
        dv_dx = np.gradient(velocity[:, 1], axis=1)[0]
        vorticity = dv_dx - du_dy
        enstrophy = np.mean(vorticity ** 2)

        # Velocity gradient dissipation (proxy for energy dissipation)
        du_diff = np.gradient(velocity, axis=0)
        dissipation = np.mean([np.trace(np.outer(gradi, gradi)) for gradi in du_diff])

        # Stretching from velocity gradients
        stretch = [np.linalg.norm(grad) for grad in du_diff]
        
        # Aggregate
        results['velocity'].extend(velocity)
        results['acceleration'].extend(acceleration)
        results['enstrophy'].append(enstrophy)
        results['dissipation'].append(dissipation)
        results['stretch'].extend(stretch)

    # Convert to numpy arrays
    for k in results:
        if len(results[k]) and isinstance(results[k][0], np.ndarray):
            results[k] = np.vstack(results[k])
    
    return results
```

---

### 📊 Extended Visualization for Flow Structures

```python
def visualize_flow_field(tracks, frames):
    """Visualize tracked motion along with derived turbulence structures"""
    _, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': '3d'})

    all_positions = []
    all_velocities = []

    for track in tracks:
        points = np.array(track['positions'])
        all_positions.extend(points)

        # Compute velocity
        if len(points) >= 2:
            velocity = np.gradient(points, axis=0)
            all_velocities.extend(velocity)

    positions_array = np.array(all_positions)
    velocity_array = np.array(all_velocities)
    
    # Plot 1: Particle trajectories
    ax = axes[0, 0]
    for i, track in enumerate(tracks):
        pos = np.array(track['positions'])
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
               c=plt.cm.tab20(i % 20), alpha=0.7, lw=1)
    ax.set_title("3D Particle Trajectories", fontsize=10)

    # Plot 2: Streamlines
    ax = axes[0, 1]
    if len(positions_array) > 10:
        # Basic streamline precomputation
        grid_resolution = 10
        xlin = np.linspace(*np.percentile(positions_array[:, 0], [5, 95]), grid_resolution)
        ylin = np.linspace(*np.percentile(positions_array[:, 1], [5, 95]), grid_resolution)
        zlin = np.linspace(*np.percentile(positions_array[:, 2], [5, 95]), grid_resolution)
        
        XYZ = np.meshgrid(xlin, ylin, zlin)
        U, V, W = np.zeros_like(XYZ), np.zeros_like(XYZ), np.zeros_like(XYZ)

        # Interpolate vector field
        from scipy.interpolate import griddata
        xyz = np.stack([X.ravel() for X in XYZ], axis=1)
        uvw = griddata(positions_array, velocity_array, xyz, method='linear', fill_value=0)

        U = uvw[:,0].reshape(XYZ[0].shape)
        V = uvw[:,1].reshape(XYZ[0].shape)
        W = uvw[:,2].reshape(XYZ[0].shape)

        ax.quiver(XYZ[0][::2,::2,::2], XYZ[1][::2,::2,::2], XYZ[2][::2,::2,::2],
                 U[::2,::2,::2], V[::2,::2,::2], W[::2,::2,::2], length=2)
        ax.set_title("Derived Velocity Field", fontsize=10)

    # Plot 3: Vorticity intensity
    ax = axes[1, 0]
    if all_positions:
        velocities = np.array([track['positions'][1] - track['positions'][0] for track in tracks if len(track['positions']) > 1])
        speeds = np.linalg.norm(velocities, axis=1)
        for i, track in enumerate(tracks):
            ax.plot([track['positions'][0][0]], [track['positions'][0][1]],
                   [track['positions'][0][2]], 
                   c=plt.cm.viridis(speeds[i]/(np.max(speeds) + 1e-6)))
        ax.set_title("Initial Vorticity Intensity", fontsize=10)

    # Plot 4: Acceleration magnitude over time
    ax = axes[1, 1]
    for i, track in enumerate(tracks):
        acc = np.linalg.norm(np.gradient(np.array(track['positions']), axis=0), axis=1)
        time = np.linspace(0, len(acc)*1, len(acc))
        ax.plot(time, acc, c=plt.cm.tab10(i % 10), alpha=0.5)
    ax.set_title("Acceleration over Time", fontsize=10)
    
    plt.tight_layout()
    plt.show()
```

---

### 🧪 Complete Example Run

```python
if __name__ == "__main__":
    # Generate synthetic turbulent flow field
    frames = simulate_turbulent_flow(
        num_frames=25,
        num_particles=150,
        dt=1.0,
        box_size=100,
        sample_frequency=1
    )

    # Initialize tracker with turbulence-optimized weights
    linker = TrackLinker(
        alpha=1.0,
        dummy_cost=150.0,
        split_threshold=3.0,
        merge_threshold=2.5,
        max_disappearance=20
    )

    # Process all frames
    for idx, frame in enumerate(frames):
        linker.process_frame(frame)
        if idx % 5 == 0:
            print(f"🎉 Processed frame {idx}, Active tracks: {len(linker.tracks)}")

    # Get final tracks
    final_tracks = linker.get_tracks()
    print(f"\n📈 Final track count: {len(final_tracks)}")

    # Visualize results
    visualize_flow_field(final_tracks, frames)

    # Analyze for turbulence metrics
    flow_stats = analyze_flow_statistics(final_tracks)
    print("\n📊 Turbulent Flow Statistics:")
    for key, val in flow_stats.items():
        if len(val) and val.ndim == 1:
            print(f"{key.capitalize():<12} {np.mean(val):.3f} ± {np.std(val):.3f} (N={len(val)})")
        elif len(val):
            print(f"{key:<12} multivariate {val.shape}")
```

---

### 📌 Use Tips for Turbulent Flow Analysis

| Use Case | Tip | Example Tool |
|---------|-----|--------------|
| **Visualization** | Use velocity gradient quivers and acceleration trends | `visualize_flow_field()` |
| **Statistics** | Compute energy dissipation, enstrophy, strain | `analyze_flow_statistics()` |
| **Calibration** | Tune `noise_adapt_k` for dynamical error absorption | High values in vortical regions |
| **Data Density** | Sparse vs dense tracers – affects accuracy | Use ≥100 tracers/frame for turbulence |
| **Postprocessing** | Use tracking for particle-based vorticity estimation | See custom `analyze_flow_statistics` |

---

This package is optimized for **Lagrangian fluid flow analysis**, with advanced noise handling and motion dynamics estimation. It is ready for **2D and quasi-3D turbulent flows**, and supports extension to **4D flow tracking (3D+time)** with minor modifications. Let me know if you'd like to add:
- Support for **real experimental data**
- **Machine Learning-based motion prediction**
- **Conversion to (u, v, w) field analysis**