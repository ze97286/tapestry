"""
Configuration dataclasses for the Tapestry cfDNA methylation deconvolution pipeline.

Mirrors the YAML configuration structure with nested dataclasses for each
pipeline section. Provides loading from YAML with override support and
serialisation back to YAML.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*, mutating *base* in place."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _build_dataclass(cls, data: dict):
    """Construct a dataclass instance from a dict, ignoring unknown keys."""
    valid_keys = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Top-level project metadata."""

    name: str = "tapestry"
    output_dir: str = "./output"
    seed: int = 42
    log_level: str = "INFO"


@dataclass
class DataConfig:
    """Paths to input data and reference files."""

    manifest_path: str = ""
    cpg_index_path: Optional[str] = None
    reference_fasta: str = ""
    chromosomes: list[str] = field(default_factory=lambda: [
        f"chr{i}" for i in range(1, 23)
    ] + ["chrX"])


@dataclass
class PreprocessingConfig:
    """Read-level preprocessing parameters."""

    min_cpgs_per_read: int = 1
    n_workers: int = 8
    output_format: str = "h5"


@dataclass
class RegionsConfig:
    """Genomic windowing parameters."""

    window_size: int = 500
    step_size: int = 250
    min_cpgs_per_window: int = 5


@dataclass
class SlurmConfig:
    """HPC / Slurm job submission parameters."""

    partition: str = "short"
    account: str = "myproject"
    conda_env: str = "tapestry"


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

_SECTION_MAP: dict[str, type] = {
    "project": ProjectConfig,
    "data": DataConfig,
    "preprocessing": PreprocessingConfig,
    "regions": RegionsConfig,
    "slurm": SlurmConfig,
}


@dataclass
class Config:
    """Master configuration container for the entire Tapestry pipeline."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    regions: RegionsConfig = field(default_factory=RegionsConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # -----------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> Config:
        """Load configuration from a YAML file with optional overrides."""
        with open(path, "r") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        if overrides:
            raw = _deep_update(raw, copy.deepcopy(overrides))

        kwargs: dict[str, Any] = {}
        for section_name, dc_cls in _SECTION_MAP.items():
            section_data = raw.get(section_name, {})
            if section_data is None:
                section_data = {}
            kwargs[section_name] = _build_dataclass(dc_cls, section_data)

        return cls(**kwargs)

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert the entire config tree to a plain dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.dump(
                self.to_dict(),
                fh,
                default_flow_style=False,
                sort_keys=False,
            )
