"""Utility functions for file and directory operations."""

from pathlib import Path


def create_output_dirs(output_dir: Path) -> Path:
    """Create directories to save EDA plots."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {output_dir}")
    plots_dir = output_dir / "plots"
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory for plots: {plots_dir}")
    else:
        # Clear existing plots
        for file in plots_dir.iterdir():
            if file.is_file():
                file.unlink()
        print(f"Cleared existing plots in: {plots_dir}")
    return plots_dir
