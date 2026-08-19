"""
Unified Plotting, RGB multi-swarm compositing, and video creation utilities.
Supports standalone functions and the object-oriented MFGPlotter class across
1-Population, 2-Population, Pursuit-Evasion, GIF, PNG, and MP4 generation.
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
    """Plots a side-by-side snapshot of density and cost-to-go at step k."""
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
    """Plots evenly spaced time progression snapshots across a simulation."""
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
    """Handles snapshot dashboards, multi-population RGB heatmaps, PNG frames, GIFs, and MP4 exports."""

    def __init__(self, pde_mesh_data_1=None, solver_instance=None, pde_mesh_data_2=None, pde_mesh_data=None):
        mesh_1 = pde_mesh_data_1 if pde_mesh_data_1 is not None else pde_mesh_data
        if mesh_1 is None or solver_instance is None:
            raise ValueError("Must provide mesh data and solver instance to MFGPlotter.")

        self.Lx, self.Ly = mesh_1.Lx, mesh_1.Ly
        self.Dt, self.Nt = solver_instance.Dt, solver_instance.Nt

        # Extract 1-population vs 2-population state fields
        self.M1 = getattr(solver_instance, 'M1', getattr(solver_instance, 'M', None))
        self.M2 = getattr(solver_instance, 'M2', None)
        self.U1 = getattr(solver_instance, 'U1', getattr(solver_instance, 'U', None))
        self.U2 = getattr(solver_instance, 'U2', None)
        self.evader_trajectories = getattr(getattr(solver_instance, 'evader_swarm', None), 'Y_trajectories', None)
        self.door_mask_3d = getattr(solver_instance, 'door_mask_3d', None)

        self.wall_mask = (solver_instance.omask == 0)
        self.extent = [0, self.Lx, 0, self.Ly]

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
        """Extracts the spatial grid slice corresponding to time step t_index."""
        if data_array is None:
            return None
        shape = data_array.shape
        Nx, Ny = self.wall_mask.shape
        if len(shape) == 2:
            return data_array
        if shape == (self.Nt + 1, Nx, Ny):
            return data_array[t_index, :, :]
        elif shape == (Nx, Ny, self.Nt + 1):
            return data_array[:, :, t_index]
        else:
            return data_array[t_index]

    def _draw_goals(self, ax, t_idx=0):
        """Draws static goals, dynamic door outlines, or evader targets on the axis."""
        if self.evader_trajectories is not None:
            ev = self.evader_trajectories[t_idx]
            ax.scatter(ev[:, 0], ev[:, 1], color='#00f2fe', marker='X', s=70, edgecolor='black', linewidth=0.8, label='Evaders', zorder=10)
        else:
            if self.goals_1:
                gxs, gys = zip(*self.goals_1)
                ax.scatter(gxs, gys, color='#ff2222', marker='X', s=70, edgecolor='white', linewidth=1.2, label='Pop 1 Goals', zorder=10)
            if self.goals_2:
                gxs, gys = zip(*self.goals_2)
                ax.scatter(gxs, gys, color='#2288ff', marker='X', s=70, edgecolor='white', linewidth=1.2, label='Pop 2 Goals', zorder=10)

            if self.door_mask_3d is not None and np.sum(self.door_mask_3d[t_idx]) > 0:
                xs = np.linspace(0, self.Lx, self.M1.shape[1])
                ys = np.linspace(0, self.Ly, self.M1.shape[2])
                ax.contour(xs, ys, self.door_mask_3d[t_idx].T, levels=[0.5], colors="lime", linewidths=2)

    def _build_combined_rgb(self, m1_frame, m2_frame, m1_max, m2_max):
        """Builds a combined dual-population RGB heatmap array with power-scale alpha blending."""
        safe_m1 = np.clip(m1_frame.T / m1_max, 0.0, 1.0) if m1_max > 0 else np.zeros_like(m1_frame.T)
        safe_m2 = np.clip(m2_frame.T / m2_max, 0.0, 1.0) if m2_max > 0 else np.zeros_like(m2_frame.T)

        alpha1 = safe_m1 ** 0.4
        alpha2 = safe_m2 ** 0.4

        Ny, Nx = alpha1.shape
        rgb = np.ones((Ny, Nx, 3)) * 0.95

        rgb[:, :, 1] -= alpha1 * 0.95
        rgb[:, :, 2] -= alpha1 * 0.95

        rgb[:, :, 0] -= alpha2 * 0.95
        rgb[:, :, 1] -= alpha2 * 0.95

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb[self.wall_mask.T] = [0.1725, 0.2431, 0.3137]
        return rgb

    def plot_snapshots(self, output_file="Output/dashboard_snapshots.png"):
        """Generates snapshot dashboard (1x3 panel for 2-Pop, 2x2 panel for 1-Pop)."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        t0, t_mid, t_end = 0, self.Nt // 2, self.Nt

        if self.M2 is not None:
            # --- 2-POPULATION 1x3 TIMELINE DASHBOARD ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            m1_max = np.max(self.M1) if np.max(self.M1) > 0 else 1.0
            m2_max = np.max(self.M2) if np.max(self.M2) > 0 else 1.0

            def _style_2pop_axis(ax, title, rgb_img, t_idx):
                ax.set_facecolor('#2c3e50')
                ax.set_title(title)
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Y (meters)")
                ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
                self._draw_goals(ax, t_idx)

            rgb0 = self._build_combined_rgb(self._get_spatial_frame(self.M1, t0), self._get_spatial_frame(self.M2, t0), m1_max, m2_max)
            rgbMid = self._build_combined_rgb(self._get_spatial_frame(self.M1, t_mid), self._get_spatial_frame(self.M2, t_mid), m1_max, m2_max)
            rgbEnd = self._build_combined_rgb(self._get_spatial_frame(self.M1, t_end), self._get_spatial_frame(self.M2, t_end), m1_max, m2_max)

            _style_2pop_axis(axes[0], "Start ($t=0$)", rgb0, t0)
            _style_2pop_axis(axes[1], f"Midpoint ($t={self.Dt * t_mid:.1f}s$)", rgbMid, t_mid)
            _style_2pop_axis(axes[2], f"End ($t={self.Dt * t_end:.1f}s$)", rgbEnd, t_end)
            axes[0].legend(loc='upper right')
        else:
            # --- 1-POPULATION 2x2 DASHBOARD ---
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            def _style_1pop_axis(ax, title, im, t_idx):
                ax.set_facecolor('#2c3e50')
                ax.set_title(title)
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Y (meters)")
                fig.colorbar(im, ax=ax, label='Value')
                self._draw_goals(ax, t_idx)

            m0_frame = self._get_spatial_frame(self.M1, t0)
            m0_masked = np.ma.masked_where(self.wall_mask, m0_frame)
            im1 = axes[0, 0].imshow(m0_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd')
            _style_1pop_axis(axes[0, 0], "Initial Density $M$ ($t=0$)", im1, t0)

            mmid_frame = self._get_spatial_frame(self.M1, t_mid)
            mmid_masked = np.ma.masked_where(self.wall_mask, mmid_frame)
            im2 = axes[0, 1].imshow(mmid_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd')
            _style_1pop_axis(axes[0, 1], f"Midpoint Density $M$ ($t={self.Dt * t_mid:.1f}s$)", im2, t_mid)

            mend_frame = self._get_spatial_frame(self.M1, t_end)
            mend_masked = np.ma.masked_where(self.wall_mask, mend_frame)
            im3 = axes[1, 0].imshow(mend_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd')
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
        """Saves individual PNG frames across all time steps."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Exporting animation PNG frames to './{output_dir}'...", flush=True)

        m1_max = np.max(self.M1) if np.max(self.M1) > 0 else 1.0
        m2_max = np.max(self.M2) if (self.M2 is not None and np.max(self.M2) > 0) else 1.0

        for k in range(self.Nt + 1):
            fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
            ax.set_facecolor('#2c3e50')

            m1_frame = self._get_spatial_frame(self.M1, k)
            if self.M2 is not None:
                m2_frame = self._get_spatial_frame(self.M2, k)
                rgb_img = self._build_combined_rgb(m1_frame, m2_frame, m1_max, m2_max)
                ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
                ax.set_title(f"Pop 1 (Red) & Pop 2 (Blue) - Time: {k * self.Dt:.2f}s")
            else:
                m_masked = np.ma.masked_where(self.wall_mask, m1_frame)
                im = ax.imshow(m_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd', vmin=0, vmax=m1_max)
                fig.colorbar(im, ax=ax, label='Density')
                ax.set_title(f"Crowd Density - Time: {k * self.Dt:.2f}s")

            self._draw_goals(ax, k)
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")

            fig.savefig(f"{output_dir}/frame_{k:03d}.png", dpi=150)
            plt.close(fig)

        print(f"Successfully exported {self.Nt + 1} frames to '{output_dir}'.", flush=True)

    def create_movie(self, frame_dir="mfg_simulation_frames", output_file="Output/mfg_simulation.gif", fps=15):
        """Compiles frame sequence into an animated GIF file."""
        frame_files = sorted(glob.glob(f"{frame_dir}/frame_*.png"))
        if not frame_files:
            print(f"No frames found in '{frame_dir}' to compile.", flush=True)
            return

        from PIL import Image
        frames = [Image.open(f) for f in frame_files]
        frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
        print(f"--> Successfully created GIF animation: '{output_file}'", flush=True)

    def save_mp4(self, filename="Output/mfg_simulation.mp4", fps=15):
        """Compiles simulation directly into an MP4 video file using FFmpeg."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"Exporting MP4 video directly to '{filename}'...", flush=True)

        fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
        ax.set_facecolor('#2c3e50')

        m1_max = np.max(self.M1) if np.max(self.M1) > 0 else 1.0
        if self.M2 is not None:
            m2_max = np.max(self.M2) if np.max(self.M2) > 0 else 1.0
            rgb_img = self._build_combined_rgb(self._get_spatial_frame(self.M1, 0), self._get_spatial_frame(self.M2, 0), m1_max, m2_max)
            im = ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
            title = ax.set_title("Pop 1 (Red) & Pop 2 (Blue) - Time: 0.00s")
        else:
            m0_frame = self._get_spatial_frame(self.M1, 0)
            m_masked = np.ma.masked_where(self.wall_mask, m0_frame)
            im = ax.imshow(m_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd', vmin=0, vmax=m1_max)
            fig.colorbar(im, ax=ax, label='Density')
            title = ax.set_title("Density - Time: 0.00s")

        self._draw_goals(ax, 0)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")

        def update(k):
            m1_k = self._get_spatial_frame(self.M1, k)
            if self.M2 is not None:
                m2_k = self._get_spatial_frame(self.M2, k)
                new_rgb = self._build_combined_rgb(m1_k, m2_k, m1_max, m2_max)
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