#!/usr/bin/env python

"""Serializable class objects for storing project data.

"""

from pathlib import Path
from pydantic import BaseModel, field_validator, computed_field, Field
import numpy as np



class RNGStateModel(BaseModel):
    bit_generator: str
    state: dict

    @staticmethod
    def from_rng(rng: np.random.Generator) -> "RNGStateModel":
        """Create an RNGStateModel from a NumPy Generator."""
        return RNGStateModel(
            bit_generator=rng.bit_generator.__class__.__name__,
            state=rng.bit_generator.state
        )

    def to_rng(self) -> np.random.Generator:
        """Restore a NumPy Generator from the RNGStateModel."""
        rng = np.random.default_rng()  # Create a new Generator
        rng.bit_generator.state = self.state  # Set its state
        return rng


class Project(BaseModel):
    """..."""
    version: str

    # inputs by user
    data: Path
    name: str
    workdir: Path
    subsample_snps: bool
    weights: int = 0
    nquartets: int = 0
    random_seed: int = 0

    # checkpointing 
    bootstrap_idx: int = 0
    bootstrap_rng: RNGStateModel | None = None

    # optional
    # imap: Path | None = None
    # nspecies: int = 0
    nsnps: int = 0
    nsamples: int = 0
    nqrts: int = 0
    nqrts_total: int = 0
    samples: dict[int, str] = Field(default_factory=dict, repr=False)

    @field_validator("workdir", mode="after")
    @classmethod
    def validate_workdir(cls, value: Path | str) -> Path:
        value = Path(value).expanduser().resolve()
        value.mkdir(exist_ok=True)
        return value

    @computed_field
    def json_file(self) -> Path:
        return self.workdir / f"{self.name}.json"

    @computed_field
    def database_file(self) -> Path:
        return self.workdir / f"{self.name}.database.hdf5"

    @computed_field
    def qrts_file(self) -> Path:
        return self.workdir / f"{self.name}.quartets_{self.bootstrap_idx}.tsv"

    @computed_field
    def boots_file(self) -> Path:
        return self.workdir / f"{self.name}.boots.nwk"

    @computed_field
    def best_file(self) -> Path:
        return self.workdir / f"{self.name}.best_tree.nwk"

    @computed_field
    def consensus_file(self) -> Path:
        return self.workdir / f"{self.name}.majority_rule_consensus.nwk"

    @computed_field
    def qmc_in_file(self) -> Path:
        return self.workdir / f"{self.name}.qmc_in.txt"

    @computed_field
    def qmc_out_file(self) -> Path:
        return self.workdir / f"{self.name}.qmc_out.txt"

    def __str__(self):
        return self.model_dump_json(indent=2)

    def save_json(self) -> None:
        """Write object serialized to JSON"""
        with open(self.json_file, 'w') as out:
            json = self.model_dump_json(indent=2)
            out.write(json)

    @classmethod
    def load_json(cls, json_file: Path) -> "Project":
        """Write object serialized to JSON"""
        with open(json_file, 'r') as indata:
            try:
                proj = Project.parse_raw(indata.read())
            except ValueError as e:
                print(f"Validation Error: {e}")
        return proj




if __name__ == "__main__":
    

    proj = Project(
        version="0.1",
        data="./data.vcf",
        name="TEST",
        workdir="/tmp",
        subsample_snps=False,
    )

    rng = np.random.default_rng(123)
    x = rng.integers(0, 100)
    proj.bootstrap_rng = RNGStateModel.from_rng(rng)
    print(proj)
    rng = proj.bootstrap_rng.to_rng()
    print(rng)

