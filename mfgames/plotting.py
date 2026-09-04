"""
Unified Plotting, RGB multi-swarm compositing, and video creation utilities.

This module provides visualization tools for Mean Field Games (MFG) simulations,
including density evolution (m) and value function (u) from the coupled HJB-KFP
system. Supports standalone functions and the object-oriented MFGPlotter class
across 1-Population, 2-Population, Pursuit-Evasion scenarios with GIF, PNG, and
MP4 export capabilities.
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# =============================================================================
# STANDALONE PLOTTING FUNCTIONS
# =============================================================================

def plot_snapshot(
    m, u, k, Lx=50.0, Ly=50.0, Dt=0.1667, omask=None, door_mask=None,
    m_range=(0.0, 4.0), u_range=(0.0, 500.0)
):
    """
    Plots a side-by-side snapshot of density and value function at time step k.

    Generates a dual-panel figure showing the population density (m) and value
    function (u) at a specific time index. Includes obstacle overlays and door
    contours if provided. Uses Gaussian interpolation for smooth visualization.

    Args:
        m: Population density array, shape (Nt, Nx, Ny). Represents the solution
            to the Fokker-Planck equation.
        u: Value function array, shape (Nt, Nx, Ny). Represents the solution to
            the Hamilton-Jacobi-Bellman equation.
        k: Time step index to visualize (0 <= k < Nt).
        Lx: Physical domain width in meters (default: 50.0).
        Ly: Physical domain height in meters (default: 50.0).
        Dt: Time step size in seconds (default: 0.1667).
        omask: Optional obstacle mask array, shape (Nx, Ny). Values of 0 indicate
            obstacles, 1 indicates walkable space.
        door_mask: Optional exit door mask, shape (Nx, Ny) or (Nt, Nx, Ny). If 3D,
            extracts time-specific slice. Doors drawn as lime contours.
        m_range: Color scale range for density plot as (vmin, vmax) tuple.
        u_range: Color scale range for value function plot as (vmin, vmax) tuple.

    Returns:
        matplotlib.figure.Figure: The generated figure with two subplots.

    Raises:
        IndexError: If k is outside valid range [0, Nt).
    """
    Nt = m.shape[0]
    if not (0 <= k < Nt):
        raise IndexError(f"k={k} out of range for Nt={Nt}")
    current_time = k * Dt
    extent = [0, Lx, 0, Ly]
    Nx, Ny = m.shape[1], m.shape[2]
    xs = np.linspace(0, Lx, Nx)
    ys = np.linspace(0, Ly, Ny)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im_m = axes[0].imshow(
        m[k].T, cmap="magma", origin="lower", extent=extent, interpolation="gaussian",
        vmin=m_range[0], vmax=m_range[1]
    )
    axes[0].set_title(f"Density ($m$) at $t = {current_time:.2f}$s", fontsize=14)
    fig.colorbar(im_m, ax=axes[0], label="Density")

    im_u = axes[1].imshow(
        u[k].T, cmap="viridis", origin="lower", extent=extent, interpolation="gaussian",
        vmin=u_range[0], vmax=u_range[1]
    )
    axes[1].set_title(f"Value ($u$) at $t = {current_time:.2f}$s", fontsize=14)
    fig.colorbar(im_u, ax=axes[1], label="Value Function")

    door_mask_k = door_mask[k] if (door_mask is not None and door_mask.ndim == 3) else door_mask

    for ax in axes:
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        if omask is not None:
            obstacle_rgba = np.zeros((*omask.shape, 4))
            # Semi-transparent black overlay for obstacles (alpha=0.75)
            obstacle_rgba[..., 3] = np.where(omask == 0, 0.75, 0.0)
            ax.imshow(obstacle_rgba.transpose(1, 0, 2), origin="lower", extent=extent)
        if door_mask_k is not None:
            ax.contour(xs, ys, door_mask_k.T, levels=[0.5], colors="lime", linewidths=2)

    plt.tight_layout()
    return fig


def plot_progression(
    data, title, cmap="magma", num_snapshots=10, Lx=50.0, Ly=50.0, Dt=0.1667,
    omask=None, door_mask=None, data_range=(None, None)
):
    """
    Plots evenly spaced time progression snapshots across a simulation.

    Creates a multi-panel figure showing temporal evolution of density or value
    function fields at uniformly spaced time indices. Useful for visualizing
    convergence behavior and transient dynamics.

    Args:
        data: Array to visualize, shape (Nt, Nx, Ny). Typically density (m) or
            value function (u) from MFG solver output.
        title: Super-title for the figure and colorbar label.
        cmap: Matplotlib colormap name (default: 'magma').
        num_snapshots: Number of temporal snapshots to display. If greater than
            Nt, will be clamped to Nt (default: 10).
        Lx: Physical domain width in meters (default: 50.0).
        Ly: Physical domain height in meters (default: 50.0).
        Dt: Time step size in seconds (default: 0.1667).
        omask: Optional obstacle mask array, shape (Nx, Ny).
        door_mask: Optional door mask, shape (Nx, Ny) or (Nt, Nx, Ny).
        data_range: Color scale limits as (vmin, vmax). Use (None, None) for
            automatic scaling based on data range.

    Returns:
        matplotlib.figure.Figure: The generated multi-panel figure.
    """
    Nt = data.shape[0]
    num_snapshots = min(num_snapshots, Nt)
    # Uniformly sample time indices across the entire simulation span
    indices = np.linspace(0, Nt - 1, num_snapshots, dtype=int)
    extent = [0, Lx, 0, Ly]
    Nx, Ny = data.shape[1], data.shape[2]
    xs = np.linspace(0, Lx, Nx)
    ys = np.linspace(0, Ly, Ny)

    fig, axes = plt.subplots(1, num_snapshots, figsize=(4 * num_snapshots, 4))
    if num_snapshots == 1:
        axes = [axes]

    im = None
    for ax, k in zip(axes, indices):
        im = ax.imshow(
            data[k].T, cmap=cmap, origin="lower", extent=extent, interpolation="gaussian",
            vmin=data_range[0], vmax=data_range[1]
        )
        ax.set_title(f"$t = {k * Dt:.1f}$s", fontsize=10)
        ax.set_xlabel("X (m)", fontsize=8)
        if omask is not None:
            obstacle_rgba = np.zeros((*omask.shape, 4))
            obstacle_rgba[..., 3] = np.where(omask == 0, 0.75, 0.0)
            ax.imshow(obstacle_rgba.transpose(1, 0, 2), origin="lower", extent=extent)
        if door_mask is not None:
            door_mask_k = door_mask[k] if door_mask.ndim == 3 else door_mask
            ax.contour(xs, ys, door_mask_k.T, levels=[0.5], colors="lime", linewidths=2)

    fig.colorbar(im, ax=axes, label=title, shrink=0.7)
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# OBJECT-ORIENTED PLOTTER CLASS
# =============================================================================

class MFGPlotter:
    """
    Handles snapshot dashboards, multi-population RGB heatmaps, PNG frames, GIFs, and MP4 exports.

    Provides object-oriented interface for visualizing 1-population or 2-population
    MFG simulations, including pursuit-evasion scenarios. Supports static dashboards,
    frame-by-frame PNG export, animated GIF generation, and direct MP4 video rendering.

    Attributes:
        Lx (float): Physical domain width in meters.
        Ly (float): Physical domain height in meters.
        Dt (float): Time step size in seconds.
        Nt (int): Number of time steps in simulation.
        M1 (np.ndarray): Population 1 density array.
        M2 (np.ndarray or None): Population 2 density array.
        U1 (np.ndarray): Population 1 value function array.
        U2 (np.ndarray or None): Population 2 value function array.
        evader_trajectories (np.ndarray or None): Evader positions for pursuit-evasion.
        door_mask (np.ndarray or None): Time-dependent door masks, shape (Nt+1, Nx, Ny).
        door_mask_3d (np.ndarray or None): Alias for door_mask for backward compatibility.
        wall_mask (np.ndarray): Boolean array marking obstacle locations.
        extent (list): Matplotlib extent [x_min, x_max, y_min, y_max].
        goals_1 (list): List of (x, y) goal positions for population 1.
        goals_2 (list): List of (x, y) goal positions for population 2.
    """
    def __init__(self, pde_mesh_data_1=None, solver_instance=None, pde_mesh_data_2=None, pde_mesh_data=None):
        """
        Initialize MFGPlotter with mesh geometry and solver state.

        Args:
            pde_mesh_data_1: Primary mesh data object containing domain geometry (Lx, Ly),
                spatial grid coordinates (X, Y), cell spacing (dx, dy), and goal positions.
                If None, falls back to pde_mesh_data.
            solver_instance: Solver object containing computed state arrays (M, U, M1, M2,
                U1, U2), temporal parameters (Dt, Nt), obstacle mask (omask), and goal manager.
            pde_mesh_data_2: Optional secondary mesh for 2-population simulations.
            pde_mesh_data: Deprecated alias for pde_mesh_data_1 (backwards compatibility).

        Raises:
            ValueError: If mesh_1 or solver_instance is None.
        """
        mesh_1 = pde_mesh_data_1 if pde_mesh_data_1 is not None else pde_mesh_data
        if mesh_1 is None or solver_instance is None:
            raise ValueError("Must provide mesh data and solver instance to MFGPlotter.")

        # Spatial domain dimensions and discretization steps
        self.Lx, self.Ly = mesh_1.Lx, mesh_1.Ly
        self.Dx = getattr(mesh_1, 'dx', getattr(solver_instance, 'Dx', None))
        self.Dy = getattr(mesh_1, 'dy', getattr(solver_instance, 'Dy', None))
        self.X = getattr(mesh_1, 'X', None)
        self.Y = getattr(mesh_1, 'Y', None)

        # Temporal parameters
        self.Dt, self.Nt = solver_instance.Dt, solver_instance.Nt

        # Flexible state extraction: try 2-pop fields (M1/M2) first, fall back to 1-pop (M/U)
        self.M1 = getattr(solver_instance, 'M1', getattr(solver_instance, 'M', None))
        self.M2 = getattr(solver_instance, 'M2', None)
        self.U1 = getattr(solver_instance, 'U1', getattr(solver_instance, 'U', None))
        self.U2 = getattr(solver_instance, 'U2', None)

        # Goal manager and trajectory tracking
        self.goal_instance = getattr(solver_instance, 'goal', getattr(solver_instance, 'evader_swarm', None))
        self.evader_trajectories = getattr(self.goal_instance, 'Y_trajectories', None)

        # Spatial masks
        self.door_mask = getattr(solver_instance, 'door_mask', getattr(solver_instance, 'door_mask_3d', None))
        self.door_mask_3d = self.door_mask
        self.wall_mask = (solver_instance.omask == 0)
        self.extent = [0, self.Lx, 0, self.Ly]

        # Helper: extract goal positions from mesh with flexible API
        def _extract_goals(mesh):
            if mesh is None:
                return []
            if hasattr(mesh, 'get_goals'):
                return mesh.get_goals()
            if hasattr(mesh, 'get_goal_positions'):
                return mesh.get_goal_positions()
            return []

        self.goals_1 = _extract_goals(mesh_1)
        self.goals_2 = _extract_goals(pde_mesh_data_2)

    def _get_spatial_frame(self, data_array, t_index):
        """
        Extracts the spatial grid slice corresponding to time step t_index.

        Handles multiple array storage conventions: (Nt+1, Nx, Ny), (Nx, Ny, Nt+1),
        or static (Nx, Ny) arrays. Infers layout based on shape matching.

        Args:
            data_array: Density or value function array in any supported layout.
            t_index: Time step index to extract.

        Returns:
            np.ndarray: Spatial slice of shape (Nx, Ny), or None if data_array is None.
        """
        if data_array is None:
            return None
        shape = data_array.shape
        Nx, Ny = self.wall_mask.shape
        if len(shape) == 2:
            # Static array (no time dimension)
            return data_array
        if shape == (self.Nt + 1, Nx, Ny):
            # Time-first layout (common in solver output)
            return data_array[t_index, :, :]
        elif shape == (Nx, Ny, self.Nt + 1):
            # Time-last layout (legacy compatibility)
            return data_array[:, :, t_index]
        else:
            # Fallback: assume first dimension is time
            return data_array[t_index]

    def _draw_goals(self, ax, t_idx=0):
        """
        Draws static goals, dynamic door outlines, or evader targets on the axis.

        Conditional rendering based on scenario type: evader trajectories for pursuit-
        evasion, static goal markers for 1/2-population, or time-dependent door contours.

        Args:
            ax: Matplotlib axis to draw on.
            t_idx: Time index for extracting time-dependent features (default: 0).
        """
        if self.evader_trajectories is not None and self.goal_instance is not None:
            X, Y = self.X, self.Y

            for g_idx, g_info in enumerate(self.goal_instance.goals):
                pos = self.evader_trajectories[t_idx, g_idx]
                cap = g_info.get('capacity', float('inf'))

                # Calculate cumulative mass absorbed by goal g up to frame t_idx
                cum_mass = 0.0
                if np.isfinite(cap) and self.M1 is not None and X is not None and Y is not None:
                    for k in range(t_idx + 1):
                        gx, gy = self.evader_trajectories[k, g_idx]
                        region = (np.abs(X - gx) <= self.Dx) & (np.abs(Y - gy) <= self.Dy)
                        cum_mass += np.sum(self.M1[k][region]) * self.Dx * self.Dy

                # Set color: Red (#ff2222) if saturated, Cyan/Blue (#00f2fe) if active
                is_saturated = cum_mass >= cap
                marker_color = '#ff2222' if is_saturated else '#00f2fe'

                ax.scatter(
                    pos[0], pos[1],
                    color=marker_color,
                    marker='X',
                    s=80,
                    edgecolor='black',
                    linewidth=0.8,
                    zorder=10
                )

            if self.door_mask is not None and np.sum(self.door_mask[t_idx]) > 0:
                xs = np.linspace(0, self.Lx, self.M1.shape[1])
                ys = np.linspace(0, self.Ly, self.M1.shape[2])
                ax.contour(xs, ys, self.door_mask[t_idx].T, levels=[0.5], colors="lime", linewidths=2)
        else:
            # Standard MFG: plot static goal locations
            if self.goals_1:
                gxs, gys = zip(*self.goals_1)
                ax.scatter(gxs, gys, color='#ff2222', marker='X', s=70, edgecolor='white', linewidth=1.2, label='Pop 1 Goals', zorder=10)
            if self.goals_2:
                gxs, gys = zip(*self.goals_2)
                ax.scatter(gxs, gys, color='#2288ff', marker='X', s=70, edgecolor='white', linewidth=1.2, label='Pop 2 Goals', zorder=10)
            if self.door_mask is not None and np.sum(self.door_mask[t_idx]) > 0:
                xs = np.linspace(0, self.Lx, self.M1.shape[1])
                ys = np.linspace(0, self.Ly, self.M1.shape[2])
                ax.contour(xs, ys, self.door_mask[t_idx].T, levels=[0.5], colors="lime", linewidths=2)

    def _build_combined_rgb(self, m1_frame, m2_frame, m1_max, m2_max):
        """
        Builds a combined dual-population RGB heatmap array with power-scale alpha blending.

        Uses perceptually-uniform power-law alpha scaling (exponent 0.4) based on Stevens'
        power law for brightness perception. Densities are composited via subtractive RGB
        blending on a white background: Population 1 controls red channel (appears red/magenta),
        Population 2 controls blue channel (appears blue/magenta). Overlapping regions blend
        toward magenta.

        Mathematical approach:
            alpha_i = (m_i / m_i_max)^0.4   # Gamma correction for perceptual uniformity
            RGB = [1, 1, 1] - alpha_1 * [0, 0.95, 0.95] - alpha_2 * [0.95, 0.95, 0]
            RGB[obstacles] = [0.1725, 0.2431, 0.3137]  # Dark gray

        Args:
            m1_frame: Population 1 density array, shape (Nx, Ny).
            m2_frame: Population 2 density array, shape (Nx, Ny).
            m1_max: Normalization constant for population 1 (typically max density at t=0).
            m2_max: Normalization constant for population 2.

        Returns:
            np.ndarray: RGB image array, shape (Ny, Nx, 3), values in [0, 1].
        """
        # Normalize densities to [0, 1] range
        safe_m1 = np.clip(m1_frame.T / m1_max, 0.0, 1.0) if m1_max > 0 else np.zeros_like(m1_frame.T)
        safe_m2 = np.clip(m2_frame.T / m2_max, 0.0, 1.0) if m2_max > 0 else np.zeros_like(m2_frame.T)

        # Apply power-law gamma correction (0.4 exponent) for perceptual uniformity
        alpha1 = safe_m1 ** 0.4
        alpha2 = safe_m2 ** 0.4

        # Initialize with near-white background (0.95 to avoid harsh pure white)
        Ny, Nx = alpha1.shape
        rgb = np.ones((Ny, Nx, 3)) * 0.95

        # Subtractive blending: Pop 1 reduces green/blue (→ red), Pop 2 reduces red/green (→ blue)
        rgb[:, :, 1] -= alpha1 * 0.95
        rgb[:, :, 2] -= alpha1 * 0.95
        rgb[:, :, 0] -= alpha2 * 0.95
        rgb[:, :, 1] -= alpha2 * 0.95

        # Clamp to valid RGB range and apply wall mask
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb[self.wall_mask.T] = [0.1725, 0.2431, 0.3137]
        return rgb

    def plot_snapshots(self, output_file="Output/dashboard_snapshots.png"):
        """
        Generates snapshot dashboard (1x3 panel for 2-Pop, 2x2 panel for 1-Pop).

        Creates a multi-panel figure showing simulation state at key time points (start,
        midpoint, end). For 2-population scenarios, displays RGB composite density heatmaps.
        For 1-population, shows density evolution and initial value function.

        Uses stable color scaling: computes vmax from t=0 frame to prevent color washout
        as density disperses over time. This preserves visual contrast throughout the
        simulation.

        Args:
            output_file: Path to save PNG dashboard (default: 'Output/dashboard_snapshots.png').
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        t0, t_mid, t_end = 0, self.Nt // 2, self.Nt

        # Calculate stable vmax from initial t=0 frame to preserve vibrant colors
        # (density typically decreases as population disperses toward goals)
        m0_frame = self._get_spatial_frame(self.M1, t0)
        m0_masked = np.ma.masked_where(self.wall_mask, m0_frame)
        stable_vmax = np.max(m0_masked) if np.max(m0_masked) > 0 else 1.0

        if self.M2 is not None:
            # Two-population scenario: RGB composite visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            m2_0_frame = self._get_spatial_frame(self.M2, t0)
            m2_0_masked = np.ma.masked_where(self.wall_mask, m2_0_frame)
            m2_max = np.max(m2_0_masked) if np.max(m2_0_masked) > 0 else 1.0

            def _style_2pop_axis(ax, title, rgb_img, t_idx):
                ax.set_facecolor('#2c3e50')
                ax.set_title(title)
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Y (meters)")
                ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
                self._draw_goals(ax, t_idx)

            rgb0 = self._build_combined_rgb(self._get_spatial_frame(self.M1, t0), self._get_spatial_frame(self.M2, t0), stable_vmax, m2_max)
            rgbMid = self._build_combined_rgb(self._get_spatial_frame(self.M1, t_mid), self._get_spatial_frame(self.M2, t_mid), stable_vmax, m2_max)
            rgbEnd = self._build_combined_rgb(self._get_spatial_frame(self.M1, t_end), self._get_spatial_frame(self.M2, t_end), stable_vmax, m2_max)

            _style_2pop_axis(axes[0], "Start ($t=0$)", rgb0, t0)
            _style_2pop_axis(axes[1], f"Midpoint ($t={self.Dt * t_mid:.1f}s$)", rgbMid, t_mid)
            _style_2pop_axis(axes[2], f"End ($t={self.Dt * t_end:.1f}s$)", rgbEnd, t_end)
            axes[0].legend(loc='upper right')
        else:
            # Single-population scenario: density + value function visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            def _style_1pop_axis(ax, title, im, t_idx):
                ax.set_facecolor('#2c3e50')
                ax.set_title(title)
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Y (meters)")
                fig.colorbar(im, ax=ax, label='Value')
                self._draw_goals(ax, t_idx)

            im1 = axes[0, 0].imshow(m0_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd', vmin=0, vmax=stable_vmax)
            _style_1pop_axis(axes[0, 0], "Initial Density $M$ ($t=0$)", im1, t0)

            mmid_frame = self._get_spatial_frame(self.M1, t_mid)
            mmid_masked = np.ma.masked_where(self.wall_mask, mmid_frame)
            im2 = axes[0, 1].imshow(mmid_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd', vmin=0, vmax=stable_vmax)
            _style_1pop_axis(axes[0, 1], f"Midpoint Density $M$ ($t={self.Dt * t_mid:.1f}s$)", im2, t_mid)

            mend_frame = self._get_spatial_frame(self.M1, t_end)
            mend_masked = np.ma.masked_where(self.wall_mask, mend_frame)
            im3 = axes[1, 0].imshow(mend_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd', vmin=0, vmax=stable_vmax)
            _style_1pop_axis(axes[1, 0], f"Final Density $M$ ($t={self.Dt * t_end:.1f}s$)", im3, t_end)

            u0_frame = self._get_spatial_frame(self.U1, t0)
            u0_masked = np.ma.masked_where(self.wall_mask, u0_frame)
            im4 = axes[1, 1].imshow(u0_masked.T, origin='lower', extent=self.extent, cmap='viridis_r')
            _style_1pop_axis(axes[1, 1], "Value Function $U$ ($t=0$)", im4, t0)

            axes[0, 0].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"--> Saved snapshot dashboard to '{output_file}'", flush=True)
        plt.close(fig)

    def save_density_frames(self, output_dir="mfg_simulation_frames"):
        """
        Saves individual PNG frames across all time steps using t=0 stable color scaling.

        Exports Nt+1 individual PNG files (frame_000.png, frame_001.png, ...) for each
        time step in the simulation. Uses t=0 density maximum for color scale normalization
        to maintain consistent visual appearance across frames. Frame naming uses zero-
        padded 3-digit numbering for proper lexicographic ordering.

        Args:
            output_dir: Directory path for frame output (default: 'mfg_simulation_frames').
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Exporting animation PNG frames to './{output_dir}'...", flush=True)

        # Compute stable color scale from t=0 to avoid washed-out appearance
        m1_0_frame = self._get_spatial_frame(self.M1, 0)
        m1_0_masked = np.ma.masked_where(self.wall_mask, m1_0_frame)
        stable_vmax1 = np.max(m1_0_masked) if np.max(m1_0_masked) > 0 else 1.0

        if self.M2 is not None:
            m2_0_frame = self._get_spatial_frame(self.M2, 0)
            m2_0_masked = np.ma.masked_where(self.wall_mask, m2_0_frame)
            stable_vmax2 = np.max(m2_0_masked) if np.max(m2_0_masked) > 0 else 1.0

        for k in range(self.Nt + 1):
            fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
            ax.set_facecolor('#2c3e50')

            m1_frame = self._get_spatial_frame(self.M1, k)
            if self.M2 is not None:
                m2_frame = self._get_spatial_frame(self.M2, k)
                rgb_img = self._build_combined_rgb(m1_frame, m2_frame, stable_vmax1, stable_vmax2)
                ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
                ax.set_title(f"Pop 1 (Red) & Pop 2 (Blue) - Time: {k * self.Dt:.2f}s")
            else:
                m_masked = np.ma.masked_where(self.wall_mask, m1_frame)
                im = ax.imshow(m_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd', vmin=0, vmax=stable_vmax1)
                fig.colorbar(im, ax=ax, label='Density')
                ax.set_title(f"Crowd Density - Time: {k * self.Dt:.2f}s")

            self._draw_goals(ax, k)
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")

            fig.savefig(f"{output_dir}/frame_{k:03d}.png", dpi=150)
            plt.close(fig)

        print(f"Successfully exported {self.Nt + 1} frames to '{output_dir}'.", flush=True)

    def create_movie(self, frame_dir="mfg_simulation_frames", output_file="Output/mfg_simulation.gif", fps=15):
        """
        Compiles frame sequence into an animated GIF file.

        Reads PNG frames from frame_dir (must be named frame_NNN.png with zero-padded
        indices) and assembles them into an animated GIF using PIL. Frame duration is
        computed from fps parameter.

        Args:
            frame_dir: Directory containing input PNG frames (default: 'mfg_simulation_frames').
            output_file: Path for output GIF file (default: 'Output/mfg_simulation.gif').
            fps: Frame rate in frames per second (default: 15). Duration per frame = 1000/fps ms.

        Note:
            Requires PIL (Pillow) library. If no frames found, prints warning and returns.
        """
        frame_files = sorted(glob.glob(f"{frame_dir}/frame_*.png"))
        if not frame_files:
            print(f"No frames found in '{frame_dir}' to compile.", flush=True)
            return

        from PIL import Image
        frames = [Image.open(f) for f in frame_files]
        frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
        print(f"--> Successfully created GIF animation: '{output_file}'", flush=True)

    def save_mp4(self, filename="Output/mfg_simulation.mp4", fps=15):
        """
        Compiles simulation directly into an MP4 video file using FFmpeg.

        Generates MP4 video directly from simulation state arrays without intermediate
        PNG frame storage. Uses matplotlib.animation.FuncAnimation with FFmpeg writer.
        Falls back to GIF via create_movie() if FFmpeg is unavailable.

        The update function is a closure that captures stable color scaling parameters
        (vmax from t=0) and efficiently updates only image data, not the entire plot.

        Args:
            filename: Path for output MP4 file (default: 'Output/mfg_simulation.mp4').
            fps: Frame rate in frames per second (default: 15).

        Note:
            Requires FFmpeg installed and accessible in system PATH. If FFmpeg fails,
            prints notice and user should call create_movie() for GIF fallback.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"Exporting MP4 video directly to '{filename}'...", flush=True)

        fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
        ax.set_facecolor('#2c3e50')

        m1_0_frame = self._get_spatial_frame(self.M1, 0)
        m1_0_masked = np.ma.masked_where(self.wall_mask, m1_0_frame)
        stable_vmax1 = np.max(m1_0_masked) if np.max(m1_0_masked) > 0 else 1.0

        if self.M2 is not None:
            m2_0_frame = self._get_spatial_frame(self.M2, 0)
            m2_0_masked = np.ma.masked_where(self.wall_mask, m2_0_frame)
            stable_vmax2 = np.max(m2_0_masked) if np.max(m2_0_masked) > 0 else 1.0

            rgb_img = self._build_combined_rgb(self._get_spatial_frame(self.M1, 0), self._get_spatial_frame(self.M2, 0), stable_vmax1, stable_vmax2)
            im = ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
            title = ax.set_title("Pop 1 (Red) & Pop 2 (Blue) - Time: 0.00s")
        else:
            m_masked = np.ma.masked_where(self.wall_mask, m1_0_frame)
            im = ax.imshow(m_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd', vmin=0, vmax=stable_vmax1)
            fig.colorbar(im, ax=ax, label='Density')
            title = ax.set_title("Density - Time: 0.00s")

        self._draw_goals(ax, 0)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")

        # Animation update closure: efficiently updates only image data per frame
        def update(k):
            m1_k = self._get_spatial_frame(self.M1, k)
            if self.M2 is not None:
                m2_k = self._get_spatial_frame(self.M2, k)
                new_rgb = self._build_combined_rgb(m1_k, m2_k, stable_vmax1, stable_vmax2)
                im.set_data(new_rgb)
                title.set_text(f"Pop 1 (Red) & Pop 2 (Blue) - Time: {k * self.Dt:.2f}s")
            else:
                m_masked = np.ma.masked_where(self.wall_mask, m1_k)
                im.set_data(m_masked.T)
                title.set_text(f"Density - Time: {k * self.Dt:.2f}s")
            return [im, title]

        ani = animation.FuncAnimation(fig, update, frames=self.Nt + 1, blit=True)
        try:
            ani.save(filename, writer='ffmpeg', fps=fps, dpi=150)
            print(f"--> Successfully exported MP4 video to '{filename}'.", flush=True)
        except Exception as e:
            # FFmpeg missing or incompatible: inform user of GIF alternative
            print(f"Notice: Could not save MP4 via FFmpeg ({e}). Fallback to GIF via create_movie().", flush=True)
        finally:
            plt.close(fig)