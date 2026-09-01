"""
Universal command-line argument parser for all project scripts.
"""
import argparse
import logging
from pathlib import Path
from yaml import safe_load
from mfgames.time import fancy_timestamp


class Options:
    """Universal command-line argument and YAML config parser for MFG scripts."""

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Mean Field Games Drone Simulation Suite")

        # --- Core Execution & Logging Arguments ---
        self.parser.add_argument('--config', default='configs/pursuit_evasion.yml', type=str, help='Path to YAML run configuration file.')
        self.parser.add_argument('--results_dir', default='results', type=str, help='Base directory for storing experiment outputs.')
        self.parser.add_argument('--exp_name', default='default_exp', type=str, help='Experiment group name.')
        self.parser.add_argument('--run_name', default=None, type=str, help='Optional run identifier name.')
        self.parser.add_argument('--timezone', default='UTC', type=str, help='Timezone for output directory timestamps.')

        # --- Domain & Environment Parameters ---
        self.parser.add_argument('--map_file', default=None, type=str, help='Path to MovingAI .map file.')
        self.parser.add_argument('--scen_file', default=None, type=str, help='Path to MovingAI .scen file.')
        self.parser.add_argument('--room_width', default=768.0, type=float, help='Domain width Lx (meters).')
        self.parser.add_argument('--room_height', default=768.0, type=float, help='Domain height Ly (meters).')
        self.parser.add_argument('--Nx', default=100, type=int, help='Grid points along X.')
        self.parser.add_argument('--Ny', default=100, type=int, help='Grid points along Y.')
        self.parser.add_argument('--num_agents', default=50, type=int, help='Number of agents/goals to parse.')

        # --- Time & Solver Parameters ---
        self.parser.add_argument('--T', default=3.0, type=float, help='Total simulation duration T (seconds).')
        self.parser.add_argument('--Nt', default=100, type=int, help='Number of time subintervals.')
        self.parser.add_argument('--max_iters', default=10, type=int, help='Maximum Picard relaxation iterations.')
        self.parser.add_argument('--relaxation_theta', default=0.1, type=float, help='Picard under-relaxation parameter.')

        # --- Evader Dynamics ---
        self.parser.add_argument('--v_max_evader', default=15.0, type=float, help='Evader maximum speed limit (m/s).')

        self._paths = ['config', 'map_file', 'scen_file', 'results_dir']

    def _load_conf(self) -> None:
        """Loads settings from YAML config file and merges them with CLI arguments."""
        config_path = Path(self.args.config)
        if not config_path.exists():
            logging.warning(f"Config file not found at {config_path}. Proceeding with CLI defaults.")
            return

        try:
            with open(config_path, 'r') as file:
                settings = safe_load(file) or {}

            for key, value in settings.items():
                if hasattr(self.args, key):
                    if isinstance(getattr(self.args, key), dict):
                        current_dict = getattr(self.args, key)
                        for sub_key, sub_value in value.items():
                            if sub_key not in current_dict or current_dict.get(sub_key) is None:
                                current_dict[sub_key] = sub_value
                    elif getattr(self.args, key) is None or self._is_default_cli_val(key):
                        setattr(self.args, key, value)
                else:
                    logging.warning(f'Invalid or obsolete configuration key: {key}')
        except Exception as e:
            logging.exception(f"Error parsing YAML config file at {config_path}.")
            raise e

    def _is_default_cli_val(self, key: str) -> bool:
        """Checks if a parameter still holds its default CLI value."""
        return getattr(self.args, key) == self.parser.get_default(key)

    def _update_path_args(self) -> None:
        """Converts string path arguments to absolute Path objects."""
        for arg_key in self._paths:
            if hasattr(self.args, arg_key):
                path_str = getattr(self.args, arg_key)
                if path_str is not None:
                    setattr(self.args, arg_key, Path(path_str).absolute())

    def parseArgs(self) -> argparse.Namespace:
        """Parses CLI inputs, applies YAML settings, and generates the output save directory."""
        self.args = self.parser.parse_args()

        if self.args.config:
            self._load_conf()

        self._update_path_args()

        timestamp = fancy_timestamp(tz=self.args.timezone) if self.args.timezone else fancy_timestamp()
        run_identifier = f"{timestamp}_{self.args.run_name}" if self.args.run_name else timestamp

        if self.args.results_dir and self.args.exp_name:
            self.args.save_dir = self.args.results_dir / self.args.exp_name / run_identifier
        elif self.args.results_dir:
            self.args.save_dir = self.args.results_dir / run_identifier
        else:
            self.args.save_dir = Path.cwd() / "results" / run_identifier

        self.args.save_dir.mkdir(parents=True, exist_ok=True)
        return self.args