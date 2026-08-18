"""
Plotting and movie creation utilities.
"""
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


class MFGPlotter:
    def __init__(self, pde_mesh_data, solver_instance):
        self.Lx, self.Ly = pde_mesh_data.Lx, pde_mesh_data.Ly
        self.Dt, self.Nt = solver_instance.Dt, solver_instance.Nt
        self.M, self.U = solver_instance.M, solver_instance.U
        self.evader_trajectories = getattr(solver_instance.evader_swarm, 'Y_trajectories', None)
        self.door_mask_3d = getattr(solver_instance, 'door_mask_3d', None)
        self.wall_mask = (solver_instance.omask == 0)
        self.extent = [0, self.Lx, 0, self.Ly]

    def plot_snapshots(self, output_file="Output/dashboard_snapshots.png"):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        t0, t_mid, t_end = 0, self.Nt // 2, self.Nt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        def _style_axis(ax, title, im, t_idx):
            ax.set_facecolor('#2c3e50')
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
            if self.evader_trajectories is not None:
                ev = self.evader_trajectories[t_idx]
                ax.scatter(ev[:, 0], ev[:, 1], color='#00f2fe', marker='X', s=70, edgecolor='black', zorder=10)
            elif self.door_mask_3d is not None:
                xs = np.linspace(0, self.Lx, self.M.shape[1])
                ys = np.linspace(0, self.Ly, self.M.shape[2])
                ax.contour(xs, ys, self.door_mask_3d[t_idx].T, levels=[0.5], colors="lime", linewidths=2)

        m0_masked = np.ma.masked_where(self.wall_mask, self.M[t0])
        im1 = axes[0, 0].imshow(m0_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd')
        _style_axis(axes[0, 0], "Initial Density ($t=0$)", im1, t0)

        mmid_masked = np.ma.masked_where(self.wall_mask, self.M[t_mid])
        im2 = axes[0, 1].imshow(mmid_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd')
        _style_axis(axes[0, 1], f"Midpoint Density ($t={self.Dt * t_mid:.1f}s$)", im2, t_mid)

        mend_masked = np.ma.masked_where(self.wall_mask, self.M[t_end])
        im3 = axes[1, 0].imshow(mend_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd')
        _style_axis(axes[1, 0], f"Final Density ($t={self.Dt * t_end:.1f}s$)", im3, t_end)

        u0_masked = np.ma.masked_where(self.wall_mask, self.U[t0])
        im4 = axes[1, 1].imshow(u0_masked.T, origin='lower', extent=self.extent, cmap='viridis_r')
        _style_axis(axes[1, 1], "Value Function ($t=0$)", im4, t0)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"--> Saved snapshot dashboard to '{output_file}'", flush=True)
        plt.close(fig)

    def save_density_frames(self, output_dir="mfg_simulation_output"):
        os.makedirs(output_dir, exist_ok=True)
        m0_masked = np.ma.masked_where(self.wall_mask, self.M[0])
        stable_vmax = np.max(m0_masked) if np.max(m0_masked) > 0 else 1.0

        for k in range(self.Nt + 1):
            fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
            m_masked = np.ma.masked_where(self.wall_mask, self.M[k])
            im = ax.imshow(m_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd', vmin=0, vmax=stable_vmax)

            if self.evader_trajectories is not None:
                ev = self.evader_trajectories[k]
                ax.scatter(ev[:, 0], ev[:, 1], color='#00f2fe', marker='X', s=60, edgecolor='black', zorder=10)
            elif self.door_mask_3d is not None:
                xs = np.linspace(0, self.Lx, self.M.shape[1])
                ys = np.linspace(0, self.Ly, self.M.shape[2])
                ax.contour(xs, ys, self.door_mask_3d[k].T, levels=[0.5], colors="lime", linewidths=2)

            ax.set_facecolor('#2c3e50')
            ax.set_title(f"MFG Simulation - Time: {k * self.Dt:.2f}s")
            fig.colorbar(im, ax=ax, label='Density')
            fig.savefig(f"{output_dir}/frame_{k:03d}.png", dpi=150)
            plt.close(fig)

    def create_movie(self, frame_dir="mfg_simulation_output", output_file="Output/mfg_simulation.gif", fps=15):
        frame_files = sorted(glob.glob(f"{frame_dir}/frame_*.png"))
        if not frame_files: return
        from PIL import Image
        frames = [Image.open(f) for f in frame_files]
        frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
        print(f"--> Successfully created GIF animation: '{output_file}'", flush=True)