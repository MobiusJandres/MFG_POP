"""
Unified Plotting, RGB Multi-Crowd Compositing, and Video Creation Utilities.

This module provides visualization tools for Mean Field Games (MFG) simulations,
including density evolution (m) and value function (u) from coupled HJB-KFP
systems. It supports standalone functions and the object-oriented MFGPlotter class
across 1-Population, 2-Population, and N-Population scenarios, featuring static
dashboards, frame-by-frame PNG export, animated GIFs, and direct MP4 export.

Key Visualization Components:
------------------------------
- Standalone Snapshot & Progression Tools: Direct inspection of (m, u) arrays
  with obstacle overlays and lime-green door contours.
- Perceptually Uniform Dual-Crowd RGB Compositing: Power-law alpha blending
  (exponent 0.4) based on Stevens' power law for brightness perception. Pop 1
  renders as red, Pop 2 as blue, and overlapping crowds blend toward purple/magenta.
- Time-Dependent Exit Door Contours: Renders dynamic Dirichlet absorbing exit
  boundaries (door_mask / door_mask_3d) as lime-green contour lines.
- Saturated vs. Active Goal Styling: Reactive evading and stationary goals render
  in cyan (#00f2fe) while active, switching to red (#ff2222) upon saturation.
- Stable Color Scaling: Normalizes density colorbars against t = 0 maximums to
  prevent visual washout as populations disperse across the spatial domain.
"""

import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# =============================================================================
# STANDALONE PLOTTING FUNCTIONS
# =============================================================================

def plot_snapshot(
    m,
    u,
    k: int,
    Lx: float = 50.0,
    Ly: float = 50.0,
    Dt: float = 0.1667,
    omask: np.ndarray | None = None,
    door_mask: np.ndarray | None = None,
    m_range: tuple[float, float] = (0.0, 4.0),
    u_range: tuple[float, float] = (0.0, 500.0),
):
    """
    Plot a side-by-side snapshot of density and value function at time step k.

    Generates a dual-panel figure showing the population density (m) from the
    Fokker-Planck equation and the value function (u) from the Hamilton-Jacobi-Bellman
    equation at a specific discrete time index. Includes semi-transparent obstacle
    overlays and contour outlines for active exit doors.

    Parameters
    ----------
    m : ndarray of shape (Nt, Nx, Ny) or (Nt+1, Nx, Ny)
        Population density array representing macroscopic crowd mass.
    u : ndarray of shape (Nt, Nx, Ny) or (Nt+1, Nx, Ny)
        Value function array representing the optimal cost-to-go.
    k : int
        Time step index to visualize (0 <= k < Nt).
    Lx : float, default=50.0
        Physical domain width in meters.
    Ly : float, default=50.0
        Physical domain height in meters.
    Dt : float, default=0.1667
        Time step size in seconds.
    omask : ndarray of shape (Nx, Ny), optional
        Binary obstacle mask array (1 = walkable space, 0 = solid obstacle).
    door_mask : ndarray of shape (Nx, Ny) or (Nt, Nx, Ny), optional
        Exit door mask array. If 3D, extracts the slice at time index k.
        Active doors are rendered as lime-green contour boundaries.
    m_range : tuple of (float, float), default=(0.0, 4.0)
        Color scale limits (vmin, vmax) for the density subplot.
    u_range : tuple of (float, float), default=(0.0, 500.0)
        Color scale limits (vmin, vmax) for the value function subplot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated dual-panel figure.

    Raises
    ------
    IndexError
        If time index k is outside the valid range [0, Nt).
    """
    Nt = m.shape[0]
    if not (0 <= k < Nt):
        raise IndexError(f"Time index k={k} is out of range for array with Nt={Nt} slices.")

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
    axes[0].set_facecolor('#2c3e50')
    fig.colorbar(im_m, ax=axes[0], label="Density")

    im_u = axes[1].imshow(
        u[k].T, cmap="viridis", origin="lower", extent=extent, interpolation="gaussian",
        vmin=u_range[0], vmax=u_range[1]
    )
    axes[1].set_title(f"Value ($u$) at $t = {current_time:.2f}$s", fontsize=14)
    axes[1].set_facecolor('#2c3e50')
    fig.colorbar(im_u, ax=axes[1], label="Value Function")

    door_mask_k = door_mask[k] if (door_mask is not None and door_mask.ndim == 3) else door_mask

    for ax in axes:
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        if omask is not None:
            obstacle_rgba = np.zeros((*omask.shape, 4))
            obstacle_rgba[..., 3] = np.where(omask == 0, 0.75, 0.0)
            ax.imshow(obstacle_rgba.transpose(1, 0, 2), origin="lower", extent=extent)
        if door_mask_k is not None and np.sum(door_mask_k) > 0:
            ax.contour(xs, ys, door_mask_k.T, levels=[0.5], colors="lime", linewidths=2)

    plt.tight_layout()
    return fig


def plot_progression(
    data: np.ndarray,
    title: str,
    cmap: str = "magma",
    num_snapshots: int = 10,
    Lx: float = 50.0,
    Ly: float = 50.0,
    Dt: float = 0.1667,
    omask: np.ndarray | None = None,
    door_mask: np.ndarray | None = None,
    data_range: tuple[float | None, float | None] = (None, None),
):
    """
    Plot evenly spaced temporal progression snapshots across an entire simulation.

    Creates a multi-panel figure illustrating the continuous spatial evolution of
    density or value function fields at uniformly distributed time indices.

    Parameters
    ----------
    data : ndarray of shape (Nt, Nx, Ny) or (Nt+1, Nx, Ny)
        Spatiotemporal tensor to visualize (density m or value function u).
    title : str
        Super-title for the figure and descriptive label for the colorbar.
    cmap : str, default='magma'
        Matplotlib colormap identifier.
    num_snapshots : int, default=10
        Number of temporal snapshot panels to display across the time horizon.
    Lx : float, default=50.0
        Physical domain width in meters.
    Ly : float, default=50.0
        Physical domain height in meters.
    Dt : float, default=0.1667
        Discretization time step size in seconds.
    omask : ndarray of shape (Nx, Ny), optional
        Binary obstacle mask array (1 = walkable, 0 = solid obstacle).
    door_mask : ndarray of shape (Nx, Ny) or (Nt, Nx, Ny), optional
        Exit door mask rendered as lime-green contour lines.
    data_range : tuple of (float or None, float or None), default=(None, None)
        Color scale limits (vmin, vmax). If (None, None), automatically inferred.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated multi-panel progression figure.
    """
    Nt = data.shape[0]
    num_snapshots = min(num_snapshots, Nt)
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
        ax.set_facecolor('#2c3e50')
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
            if np.sum(door_mask_k) > 0:
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
    Universal Dashboard, Multi-Crowd RGB Compositor, and Video Generation Engine.

    Provides a complete visualization suite for Mean Field Games simulations, supporting
    1-population traffic evacuation, 2-population competitive pursuit-evasion, and
    arbitrary N-population swarms. Automatically detects solver structure, tracks dynamic
    goal coordinates, and renders exit door boundaries.

    Attributes
    ----------
    Lx : float
        Physical domain width in meters.
    Ly : float
        Physical domain height in meters.
    Dx : float
        Grid cell spatial width in meters.
    Dy : float
        Grid cell spatial height in meters.
    Dt : float
        Time step increment in seconds.
    Nt : int
        Total number of simulation time intervals.
    M1 : ndarray of shape (Nt+1, Nx, Ny)
        Macroscopic density tensor for Population 1 (or primary single crowd).
    M2 : ndarray of shape (Nt+1, Nx, Ny) or None
        Macroscopic density tensor for Population 2 (None in 1-population runs).
    U1 : ndarray of shape (Nt+1, Nx, Ny)
        Value function tensor for Population 1.
    U2 : ndarray of shape (Nt+1, Nx, Ny) or None
        Value function tensor for Population 2 (None in 1-population runs).
    goal_instance : Goal or None
        Dynamic goal manager providing goal kinematics and capacity saturation flags.
    evader_trajectories : ndarray of shape (Nt+1, G, 2) or None
        Space-time trajectories for all G discrete goals.
    door_mask : ndarray of shape (Nt+1, Nx, Ny) or None
        Time-dependent exit door mask (1 = absorbing exit, 0 = walkable interior).
    door_mask_3d : ndarray or None
        Backward-compatible alias for door_mask.
    wall_mask : ndarray of shape (Nx, Ny), dtype bool
        Boolean mask where True indicates an impassable obstacle wall.
    extent : list of float
        Bounding box extent [0, Lx, 0, Ly] for Matplotlib imshow plotting.
    goals_1 : list of tuple
        Discrete static goal coordinates (x, y) assigned to Population 1.
    goals_2 : list of tuple
        Discrete static goal coordinates (x, y) assigned to Population 2.
    """

    def __init__(
        self,
        pde_mesh_data_1=None,
        solver_instance=None,
        pde_mesh_data_2=None,
        pde_mesh_data=None,
        **kwargs,
    ):
        """
        Initialize the universal MFGPlotter instance.

        Parameters
        ----------
        pde_mesh_data_1 : MAP2PDE or MFGTrafficGeometry, optional
            Primary spatial mesh containing domain dimensions, grid arrays (X, Y),
            and obstacle masks. Falls back to pde_mesh_data if omitted.
        solver_instance : MFGSolver, MFG2PopSolver, or MFGShadowLeaderSolver
            Solver instance containing state tensors (M, U, M1, M2), temporal parameters,
            door masks, and goal managers.
        pde_mesh_data_2 : MAP2PDE, optional
            Secondary spatial mesh for 2-population coupled games.
        pde_mesh_data : MAP2PDE, optional
            Backward-compatible alias for pde_mesh_data_1.

        Raises
        ------
        ValueError
            If neither mesh data nor a solver instance is supplied.
        """
        # Resolve flexible positional and keyword argument orderings
        mesh_1 = pde_mesh_data_1 if pde_mesh_data_1 is not None else pde_mesh_data
        if mesh_1 is None and 'mesh' in kwargs:
            mesh_1 = kwargs.get('mesh')

        solver = solver_instance
        mesh_2 = pde_mesh_data_2

        # Handle reversed positional calls where solver was passed before mesh_2
        if hasattr(mesh_2, 'run_picard_system') and not hasattr(solver, 'run_picard_system'):
            solver, mesh_2 = mesh_2, solver

        if mesh_1 is None and hasattr(solver, 'pde_mesh'):
            mesh_1 = solver.pde_mesh
        elif mesh_1 is None and hasattr(solver, 'populations') and len(solver.populations) > 0:
            mesh_1 = solver.populations[0].mesh

        if mesh_1 is None or solver is None:
            raise ValueError("Must provide valid mesh data and solver instance to MFGPlotter.")

        self.solver = solver
        self.mesh1 = mesh_1
        self.mesh2 = mesh_2

        # Spatial domain dimensions and discretization spacings
        self.Lx, self.Ly = mesh_1.Lx, mesh_1.Ly
        self.Dx = getattr(mesh_1, 'dx', getattr(solver, 'Dx', None))
        self.Dy = getattr(mesh_1, 'dy', getattr(solver, 'Dy', None))
        self.X = getattr(mesh_1, 'X', None)
        self.Y = getattr(mesh_1, 'Y', None)

        # Temporal parameters
        self.Dt, self.Nt = solver.Dt, solver.Nt
        self.extent = [0, self.Lx, 0, self.Ly]
        self.wall_mask = (solver.omask == 0)

        # Detect single-population vs. multi-population solver structure
        self.is_multipop = (
            (hasattr(solver, 'populations') and len(solver.populations) > 1)
            or (hasattr(solver, 'M2') and solver.M2 is not None)
        )

        if self.is_multipop:
            self.M1 = getattr(solver, 'M1', solver.populations[0].M if hasattr(solver, 'populations') else None)
            self.M2 = getattr(solver, 'M2', solver.populations[1].M if hasattr(solver, 'populations') else None)
            self.U1 = getattr(solver, 'U1', solver.populations[0].U if hasattr(solver, 'populations') else None)
            self.U2 = getattr(solver, 'U2', solver.populations[1].U if hasattr(solver, 'populations') else None)
        else:
            self.M1 = getattr(solver, 'M', getattr(solver, 'M1', None))
            self.M2 = None
            self.U1 = getattr(solver, 'U', getattr(solver, 'U1', None))
            self.U2 = None

        # Goal manager and dynamic evader trajectory extraction
        self.goal_instance = getattr(solver, 'goal', None)
        if self.goal_instance is None and hasattr(solver, 'populations') and len(solver.populations) > 0:
            self.goal_instance = solver.populations[0].goal

        self.evader_trajectories = getattr(self.goal_instance, 'Y_trajectories', None)

        # Door mask extraction across solver variants (traffic, evasion, multi-pop)
        self.door_mask = getattr(solver, 'door_mask', getattr(solver, 'door_mask_3d', None))
        if self.door_mask is None and hasattr(solver, 'populations') and len(solver.populations) > 0:
            self.door_mask = solver.populations[0].door_mask
        self.door_mask_3d = self.door_mask

        # Helper: Extract discrete goal locations from mesh objects
        def _extract_goals(mesh):
            if mesh is None:
                return []
            if hasattr(mesh, 'get_goals'):
                return mesh.get_goals()
            if hasattr(mesh, 'get_goal_positions'):
                return mesh.get_goal_positions()
            return []

        self.goals_1 = _extract_goals(mesh_1)
        self.goals_2 = _extract_goals(mesh_2)

    def _get_spatial_frame(self, data_array: np.ndarray | None, t_index: int) -> np.ndarray | None:
        """
        Extract the spatial 2D grid slice corresponding to time step t_index.

        Supports time-first tensors (Nt+1, Nx, Ny), time-last tensors (Nx, Ny, Nt+1),
        and static 2D grids (Nx, Ny).

        Parameters
        ----------
        data_array : ndarray or None
            Density or value function tensor.
        t_index : int
            Temporal slice index to extract.

        Returns
        -------
        spatial_slice : ndarray of shape (Nx, Ny) or None
            Extracted 2D spatial slice.
        """
        if data_array is None:
            return None

        shape = data_array.shape
        Nx, Ny = self.wall_mask.shape

        if len(shape) == 2:
            return data_array
        if shape == (self.Nt + 1, Nx, Ny) or shape[0] == self.Nt + 1:
            return data_array[t_index, :, :]
        elif shape == (Nx, Ny, self.Nt + 1) or shape[-1] == self.Nt + 1:
            return data_array[:, :, t_index]
        else:
            return data_array[t_index]

    def _draw_goals(self, ax: plt.Axes, t_idx: int = 0) -> None:
        """
        Render dynamic goals, static targets, and active exit door contours.

        Draws:
        1. Reactive / Evading Goals: Rendered as 'X' scatter markers. Colored cyan
           (#00f2fe) while active, switching to red (#ff2222) when saturated.
        2. Static Goals: Red markers for Population 1 and blue markers for Population 2.
        3. Exit Doors (door_mask): Renders active mass-absorbing doors as lime-green contours.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axis on which markers and contours are drawn.
        t_idx : int, default=0
            Time step index used to query saturation flags and door boundaries.
        """
        # 1. Render dynamic or capacity-constrained goals
        if self.evader_trajectories is not None and self.goal_instance is not None:
            for g_idx, g_info in enumerate(self.goal_instance.goals):
                pos = self.evader_trajectories[t_idx, g_idx]
                is_sat = (
                    self.goal_instance.is_saturated[t_idx, g_idx]
                    if hasattr(self.goal_instance, 'is_saturated')
                    else False
                )
                marker_color = '#ff2222' if is_sat else '#00f2fe'

                ax.scatter(
                    pos[0], pos[1],
                    color=marker_color,
                    marker='X',
                    s=85,
                    edgecolor='black',
                    linewidth=1.0,
                    zorder=10,
                )
        else:
            # 2. Render static goals from scenario definitions
            if self.goals_1:
                gxs, gys = zip(*self.goals_1)
                ax.scatter(
                    gxs, gys, color='#ff2222', marker='X', s=70,
                    edgecolor='white', linewidth=1.2, label='Pop 1 Goals', zorder=10
                )
            if self.goals_2:
                gxs, gys = zip(*self.goals_2)
                ax.scatter(
                    gxs, gys, color='#2288ff', marker='X', s=70,
                    edgecolor='white', linewidth=1.2, label='Pop 2 Goals', zorder=10
                )

        # 3. Render absorbing exit doors via lime-green contour lines
        if self.door_mask is not None:
            dm_k = self.door_mask[t_idx] if self.door_mask.ndim == 3 else self.door_mask
            if dm_k is not None and np.sum(dm_k) > 0:
                Nx, Ny = dm_k.shape
                xs = np.linspace(0, self.Lx, Nx)
                ys = np.linspace(0, self.Ly, Ny)
                ax.contour(xs, ys, dm_k.T, levels=[0.5], colors="lime", linewidths=2.0, zorder=8)

    def _build_combined_rgb(
        self,
        m1_frame: np.ndarray,
        m2_frame: np.ndarray,
        m1_max: float,
        m2_max: float,
    ) -> np.ndarray:
        """
        Build a combined dual-population RGB composite array using power-law alpha blending.

        Applies Stevens' power law for brightness perception (exponent 0.4) to normalize
        and blend two macroscopic densities against a near-white background:
            - Population 1 reduces green and blue channels (appears red/coral).
            - Population 2 reduces red and green channels (appears cyan/blue).
            - Overlapping density regions blend toward purple/magenta.
            - Obstacle wall cells are assigned dark slate gray [0.1725, 0.2431, 0.3137].

        Parameters
        ----------
        m1_frame : ndarray of shape (Nx, Ny)
            Population 1 density grid at the current time step.
        m2_frame : ndarray of shape (Nx, Ny)
            Population 2 density grid at the current time step.
        m1_max : float
            Normalization constant for Population 1 (typically max density at t = 0).
        m2_max : float
            Normalization constant for Population 2.

        Returns
        -------
        rgb : ndarray of shape (Ny, Nx, 3), dtype float64
            Rendered RGB image array with normalized float values in [0.0, 1.0].
        """
        safe_m1 = np.clip(m1_frame.T / m1_max, 0.0, 1.0) if m1_max > 0 else np.zeros_like(m1_frame.T)
        safe_m2 = np.clip(m2_frame.T / m2_max, 0.0, 1.0) if m2_max > 0 else np.zeros_like(m2_frame.T)

        alpha1 = safe_m1 ** 0.4
        alpha2 = safe_m2 ** 0.4

        Ny, Nx = alpha1.shape
        rgb = np.ones((Ny, Nx, 3), dtype=np.float64) * 0.95

        # Subtractive channel modulation
        rgb[:, :, 1] -= alpha1 * 0.95
        rgb[:, :, 2] -= alpha1 * 0.95
        rgb[:, :, 0] -= alpha2 * 0.95
        rgb[:, :, 1] -= alpha2 * 0.95

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb[self.wall_mask.T] = [0.1725, 0.2431, 0.3137]
        return rgb

    def plot_snapshots(self, output_file: str = "Output/dashboard_snapshots.png") -> None:
        """
        Generate and export a multi-panel simulation snapshot dashboard.

        Layout Adaptations:
        - Two-Population Simulations: Produces a 1x3 horizontal layout depicting Start (t=0),
          Midpoint (t=T/2), and End (t=T) using power-law RGB crowd compositing.
        - Single-Population Simulations: Produces a 2x2 dashboard displaying:
            * Top-Left: Initial Density M(0, x, y).
            * Top-Right: Midpoint Density M(T/2, x, y).
            * Bottom-Left: Final Density M(T, x, y).
            * Bottom-Right: Initial Value Function U(0, x, y).

        Color Scale Stability:
        Colorbar upper bounds (vmax) are anchored to density at t = 0 to prevent visual
        washout as crowd mass diffuses or exits across time.

        Parameters
        ----------
        output_file : str, default='Output/dashboard_snapshots.png'
            Destination file path for the exported PNG image.
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        t0, t_mid, t_end = 0, self.Nt // 2, self.Nt

        m0_frame = self._get_spatial_frame(self.M1, t0)
        m0_masked = np.ma.masked_where(self.wall_mask, m0_frame)
        stable_vmax = np.max(m0_masked) if np.max(m0_masked) > 0 else 1.0

        if self.is_multipop and self.M2 is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            m2_0_frame = self._get_spatial_frame(self.M2, t0)
            m2_0_masked = np.ma.masked_where(self.wall_mask, m2_0_frame)
            m2_max = np.max(m2_0_masked) if np.max(m2_0_masked) > 0 else 1.0

            def _style_2pop_axis(ax, title, rgb_img, t_idx):
                ax.set_facecolor('#2c3e50')
                ax.set_title(title, fontsize=12)
                ax.set_xlabel("X (meters)", fontsize=10)
                ax.set_ylabel("Y (meters)", fontsize=10)
                ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
                self._draw_goals(ax, t_idx)

            rgb0 = self._build_combined_rgb(
                self._get_spatial_frame(self.M1, t0),
                self._get_spatial_frame(self.M2, t0),
                stable_vmax, m2_max
            )
            rgbMid = self._build_combined_rgb(
                self._get_spatial_frame(self.M1, t_mid),
                self._get_spatial_frame(self.M2, t_mid),
                stable_vmax, m2_max
            )
            rgbEnd = self._build_combined_rgb(
                self._get_spatial_frame(self.M1, t_end),
                self._get_spatial_frame(self.M2, t_end),
                stable_vmax, m2_max
            )

            _style_2pop_axis(axes[0], "Start ($t=0$)", rgb0, t0)
            _style_2pop_axis(axes[1], f"Midpoint ($t={self.Dt * t_mid:.1f}$s)", rgbMid, t_mid)
            _style_2pop_axis(axes[2], f"End ($t={self.Dt * t_end:.1f}$s)", rgbEnd, t_end)
            axes[0].legend(loc='upper right')
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            def _style_1pop_axis(ax, title, im, t_idx):
                ax.set_facecolor('#2c3e50')
                ax.set_title(title, fontsize=12)
                ax.set_xlabel("X (meters)", fontsize=10)
                ax.set_ylabel("Y (meters)", fontsize=10)
                fig.colorbar(im, ax=ax, label='Magnitude')
                self._draw_goals(ax, t_idx)

            im1 = axes[0, 0].imshow(
                m0_masked.T, origin='lower', extent=self.extent,
                cmap='YlOrRd', vmin=0, vmax=stable_vmax
            )
            _style_1pop_axis(axes[0, 0], "Initial Density $M$ ($t=0$)", im1, t0)

            mmid_frame = self._get_spatial_frame(self.M1, t_mid)
            mmid_masked = np.ma.masked_where(self.wall_mask, mmid_frame)
            im2 = axes[0, 1].imshow(
                mmid_masked.T, origin='lower', extent=self.extent,
                cmap='YlOrRd', vmin=0, vmax=stable_vmax
            )
            _style_1pop_axis(axes[0, 1], f"Midpoint Density $M$ ($t={self.Dt * t_mid:.1f}$s)", im2, t_mid)

            mend_frame = self._get_spatial_frame(self.M1, t_end)
            mend_masked = np.ma.masked_where(self.wall_mask, mend_frame)
            im3 = axes[1, 0].imshow(
                mend_masked.T, origin='lower', extent=self.extent,
                cmap='YlOrRd', vmin=0, vmax=stable_vmax
            )
            _style_1pop_axis(axes[1, 0], f"Final Density $M$ ($t={self.Dt * t_end:.1f}$s)", im3, t_end)

            u0_frame = self._get_spatial_frame(self.U1, t0)
            u0_masked = np.ma.masked_where(self.wall_mask, u0_frame)
            im4 = axes[1, 1].imshow(
                u0_masked.T, origin='lower', extent=self.extent,
                cmap='viridis_r'
            )
            _style_1pop_axis(axes[1, 1], "Value Function $U$ ($t=0$)", im4, t0)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"--> Saved snapshot dashboard to '{output_file}'", flush=True)
        plt.close(fig)

    def save_density_frames(self, output_dir: str = "mfg_simulation_frames") -> None:
        """
        Export individual PNG heatmap frames for all simulation time steps.

        Generates Nt+1 sequentially numbered PNG images (frame_000.png, frame_001.png, ...)
        suitable for video compiling or presentation inspection. Normalizes density using
        initial t = 0 maxima, renders walls in dark slate gray, and renders door outlines.

        Parameters
        ----------
        output_dir : str, default='mfg_simulation_frames'
            Directory path where generated PNG frames are stored.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Exporting animation PNG frames to './{output_dir}'...", flush=True)

        m1_0_frame = self._get_spatial_frame(self.M1, 0)
        m1_0_masked = np.ma.masked_where(self.wall_mask, m1_0_frame)
        stable_vmax1 = np.max(m1_0_masked) if np.max(m1_0_masked) > 0 else 1.0

        if self.is_multipop and self.M2 is not None:
            m2_0_frame = self._get_spatial_frame(self.M2, 0)
            m2_0_masked = np.ma.masked_where(self.wall_mask, m2_0_frame)
            stable_vmax2 = np.max(m2_0_masked) if np.max(m2_0_masked) > 0 else 1.0

        for k in range(self.Nt + 1):
            fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
            ax.set_facecolor('#2c3e50')

            m1_frame = self._get_spatial_frame(self.M1, k)
            if self.is_multipop and self.M2 is not None:
                m2_frame = self._get_spatial_frame(self.M2, k)
                rgb_img = self._build_combined_rgb(m1_frame, m2_frame, stable_vmax1, stable_vmax2)
                ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
                ax.set_title(f"Pop 1 (Red) & Pop 2 (Blue) - Time: {k * self.Dt:.2f}s")
            else:
                m_masked = np.ma.masked_where(self.wall_mask, m1_frame)
                im = ax.imshow(
                    m_masked.T, origin='lower', extent=self.extent,
                    cmap='YlOrRd', vmin=0, vmax=stable_vmax1
                )
                fig.colorbar(im, ax=ax, label='Density')
                ax.set_title(f"Crowd Density - Time: {k * self.Dt:.2f}s")

            self._draw_goals(ax, k)
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")

            fig.savefig(f"{output_dir}/frame_{k:03d}.png", dpi=150)
            plt.close(fig)

        print(f"Successfully exported {self.Nt + 1} frames to '{output_dir}'.", flush=True)

    def create_movie(
        self,
        frame_dir: str = "mfg_simulation_frames",
        output_file: str = "Output/mfg_simulation.gif",
        fps: int = 15,
    ) -> None:
        """
        Compile an exported sequence of PNG frames into an animated GIF.

        Scans frame_dir for files matching 'frame_*.png' and stitches them in lexicographical
        order into a looping animated GIF using PIL.

        Parameters
        ----------
        frame_dir : str, default='mfg_simulation_frames'
            Directory containing the source PNG frames.
        output_file : str, default='Output/mfg_simulation.gif'
            Destination file path for the compiled GIF.
        fps : int, default=15
            Animation playback frame rate in frames per second.
        """
        frame_files = sorted(glob.glob(f"{frame_dir}/frame_*.png"))
        if not frame_files:
            print(f"No frames found in '{frame_dir}' to compile into GIF.", flush=True)
            return

        from PIL import Image
        frames = [Image.open(f) for f in frame_files]
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"--> Successfully created GIF animation: '{output_file}'", flush=True)

    def save_mp4(self, filename: str = "Output/mfg_simulation.mp4", fps: int = 15) -> None:
        """
        Compile the simulation directly into an MP4 video file using FFmpeg.

        Generates an MP4 video directly from memory using matplotlib.animation.FuncAnimation
        without requiring disk storage of intermediate PNG files. Incorporates stable color
        scaling, dynamic goal positions, and exit door contours.

        Parameters
        ----------
        filename : str, default='Output/mfg_simulation.mp4'
            Destination file path for the exported MP4 video.
        fps : int, default=15
            Playback frame rate in frames per second.
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        print(f"Exporting MP4 video directly to '{filename}'...", flush=True)

        fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
        ax.set_facecolor('#2c3e50')

        m1_0_frame = self._get_spatial_frame(self.M1, 0)
        m1_0_masked = np.ma.masked_where(self.wall_mask, m1_0_frame)
        stable_vmax1 = np.max(m1_0_masked) if np.max(m1_0_masked) > 0 else 1.0

        if self.is_multipop and self.M2 is not None:
            m2_0_frame = self._get_spatial_frame(self.M2, 0)
            m2_0_masked = np.ma.masked_where(self.wall_mask, m2_0_frame)
            stable_vmax2 = np.max(m2_0_masked) if np.max(m2_0_masked) > 0 else 1.0

            rgb_img = self._build_combined_rgb(m1_0_frame, m2_0_frame, stable_vmax1, stable_vmax2)
            im = ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
            title = ax.set_title("Pop 1 (Red) & Pop 2 (Blue) - Time: 0.00s")
        else:
            im = ax.imshow(
                m1_0_masked.T, origin='lower', extent=self.extent,
                cmap='YlOrRd', vmin=0, vmax=stable_vmax1
            )
            fig.colorbar(im, ax=ax, label='Density')
            title = ax.set_title("Density - Time: 0.00s")

        self._draw_goals(ax, 0)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")

        def update(k):
            m1_k = self._get_spatial_frame(self.M1, k)
            if self.is_multipop and self.M2 is not None:
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
            print(f"Notice: Could not save MP4 via FFmpeg ({e}). Fallback to GIF via create_movie().", flush=True)
        finally:
            plt.close(fig)