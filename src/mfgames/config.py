"""
Universal Command-Line Argument Parser and YAML Configuration Loader.

This module provides the Options class for managing configuration arguments across
all Mean Field Games (MFG) simulation scripts. It supports default values, YAML config file
merging, command-line parameter overrides, timezone-aware timestamped output directory
generation, and automated configuration snapshot logging.
"""
import argparse
import logging
from pathlib import Path
import yaml
from mfgames.time import fancy_timestamp


class Options:
    """
    Universal command-line argument and YAML configuration parser for MFG scripts.

    Provides a centralized command-line interface (CLI) for parsing execution parameters,
    merging YAML settings files, configuring domain and time discretizations, setting
    decoupled running and terminal crowd objective weights, and managing output logging directories.

    Attributes:
        parser (argparse.ArgumentParser): Core argument parser instance.
        args (argparse.Namespace): Parsed and resolved argument namespace.
        _paths (list[str]): List of attribute names storing file or directory path strings.
    """

    def __init__(self) -> None:
        """
        Initialize the command-line argument parser and register all execution parameters.
        
        Registers arguments for core logging, spatial domain dimensions, grid resolution,
        temporal discretization, solver relaxation, decoupled crowd cost weights, and goal kinematics.
        """
        self.parser = argparse.ArgumentParser(
            description="Mean Field Games Crowd Dynamics Simulation Suite"
        )

        # --- Core Execution & Logging Arguments ---
        self.parser.add_argument(
            '--config', default='configs/map_simulation.yml', type=str,
            help='Path to YAML run configuration file.'
        )
        self.parser.add_argument(
            '--results_dir', default='results', type=str,
            help='Base directory for storing experiment outputs.'
        )
        self.parser.add_argument(
            '--exp_name', default='default_exp', type=str,
            help='Experiment group name.'
        )
        self.parser.add_argument(
            '--run_name', default=None, type=str,
            help='Optional run identifier name.'
        )
        self.parser.add_argument(
            '--timezone', default='UTC', type=str,
            help='Timezone for output directory timestamps.'
        )

        # --- Domain & Environment Parameters ---
        self.parser.add_argument(
            '--map_file', default=None, type=str,
            help='Path to MovingAI .map file.'
        )
        self.parser.add_argument(
            '--scen_file', default=None, type=str,
            help='Path to MovingAI .scen file.'
        )
        self.parser.add_argument(
            '--room_width', default=768.0, type=float,
            help='Domain width Lx (meters).'
        )
        self.parser.add_argument(
            '--room_height', default=768.0, type=float,
            help='Domain height Ly (meters).'
        )
        self.parser.add_argument(
            '--Nx', default=100, type=int,
            help='Grid points along X.'
        )
        self.parser.add_argument(
            '--Ny', default=100, type=int,
            help='Grid points along Y.'
        )
        self.parser.add_argument(
            '--num_agents', default=50, type=int,
            help='Number of agents/goals to parse.'
        )

        # --- Behavior & Decoupled Objective Parameters ---
        self.parser.add_argument(
            '--goals_are_exits', action='store_true', default=False,
            help='If True, active goals act as absorbing exit boundaries.'
        )
        self.parser.add_argument(
            '--obstacle_penalty', default=None, type=float,
            help='Obstacle cell potential penalty in HJB solver.'
        )
        self.parser.add_argument(
            '--running_cost_weight', default=0.01, type=float,
            help='Running cost scaling weight w_R for t < T.'
        )
        self.parser.add_argument(
            '--terminal_cost_weight', default=None, type=float,
            help='Terminal cost scaling weight w_T at t = T (defaults to w_R if None).'
        )
        self.parser.add_argument(
            '--running_cost_power', default=2, type=int,
            help='Exponent p_R for running distance cost (1 for linear, 2 for squared).'
        )
        self.parser.add_argument(
            '--terminal_cost_power', default=2, type=int,
            help='Exponent p_T for terminal distance cost (1 for linear, 2 for squared).'
        )

        # --- Time & Solver Parameters ---
        self.parser.add_argument(
            '--T', default=3.0, type=float,
            help='Total simulation duration T (seconds).'
        )
        self.parser.add_argument(
            '--Nt', default=100, type=int,
            help='Number of time subintervals.'
        )
        self.parser.add_argument(
            '--max_iters', default=10, type=int,
            help='Maximum Picard relaxation iterations.'
        )
        self.parser.add_argument(
            '--relaxation_theta', default=0.1, type=float,
            help='Picard under-relaxation parameter.'
        )

        # --- Goal Dynamics & Saturation ---
        self.parser.add_argument(
            '--v_max_evader', default=15.0, type=float,
            help='Default goal maximum speed limit (m/s).'
        )
        self.parser.add_argument(
            '--saturated_goal_penalty', default=0.0, type=float,
            help='Repulsive cost weight added around saturated goals (0.0 disables penalty).'
        )

        self._paths = ['config', 'map_file', 'scen_file', 'results_dir']

    def _load_conf(self) -> None:
        """
        Load configuration settings from YAML file and merge with arguments.

        Reads the YAML file specified in self.args.config and updates parameters in
        self.args if they hold default values or are not explicitly provided via CLI.
        """
        config_path = Path(self.args.config)
        if not config_path.exists():
            logging.warning(f"Config file not found at {config_path}. Proceeding with CLI defaults.")
            return

        try:
            with open(config_path, 'r') as file:
                settings = yaml.safe_load(file) or {}

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
        """
        Check if a given argument parameter currently holds its default CLI value.

        Args:
            key (str): Argument parameter name to verify.

        Returns:
            bool: True if parameter equals the default value registered in self.parser, False otherwise.
        """
        return getattr(self.args, key) == self.parser.get_default(key)

    def _update_path_args(self) -> None:
        """Convert string path arguments listed in self._paths to absolute Path objects."""
        for arg_key in self._paths:
            if hasattr(self.args, arg_key):
                path_str = getattr(self.args, arg_key)
                if path_str is not None:
                    setattr(self.args, arg_key, Path(path_str).absolute())

    def _save_config(self) -> None:
        """
        Save the fully resolved experiment configuration dictionary to config.yml.

        Exports all resolved argument parameters from self.args into 'config.yml'
        within the timestamped results output directory (self.args.save_dir). Converts
        non-YAML-serializable objects (such as pathlib.Path) into string representations.
        """
        config_out_path = self.args.save_dir / "config.yml"
        
        config_dict = {}
        for key, val in vars(self.args).items():
            if isinstance(val, Path):
                config_dict[key] = str(val)
            else:
                config_dict[key] = val

        try:
            with open(config_out_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logging.info(f"Saved run configuration snapshot to '{config_out_path}'")
        except Exception as e:
            logging.exception(f"Failed to save configuration file to {config_out_path}: {e}")

    def parseArgs(self) -> argparse.Namespace:
        """
        Parse command-line inputs, merge YAML config settings, and generate output directory.

        Executes the full argument resolution pipeline:
        1. Parses command-line inputs via argparse.
        2. Merges settings from the specified YAML configuration file.
        3. Converts filesystem path strings to absolute Path instances.
        4. Generates a timezone-aware timestamped save directory (save_dir)
        5. Logs a snapshot of the resolved configuration to config.yml in save_dir.

        Returns:
            argparse.Namespace: Fully populated and resolved argument namespace.
        """
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

        # Create output directory and save resolved config snapshot
        self.args.save_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

        return self.args