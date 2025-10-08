#!/usr/bin/env python
"""
Download quark/gluon jet dataset from EnergyFlow.

This script downloads the standard quark-gluon jet tagging benchmark dataset
using the EnergyFlow package. The dataset consists of jets generated from
Pythia8 or Herwig7, with labels for quark (1) and gluon (0) jets.

Dataset details: https://energyflow.network/docs/datasets/#quark-and-gluon-jets
"""

import argparse
from pathlib import Path

import energyflow as ef
import numpy as np


def download_qg_jets(
    num_data: int = 100_000,
    generator: str = "pythia",
    with_bc: bool = False,
    output_dir: str = "data",
    output_name: str = "qg_jets.npz",
) -> None:
    """
    Download and save quark/gluon jet dataset.

    Parameters
    ----------
    num_data : int
        Number of jets to load (default: 100,000)
    generator : str
        Generator to use: 'pythia' or 'herwig' (default: 'pythia')
    with_bc : bool
        Include bottom and charm quarks (default: False)
    output_dir : str
        Directory to save the dataset (default: 'data')
    output_name : str
        Filename for the saved dataset (default: 'qg_jets.npz')
    """
    print(f"Loading {num_data:,} jets from {generator} (with_bc={with_bc})...")
    
    # Load dataset from EnergyFlow
    X, y = ef.qg_jets.load(num_data=num_data, generator=generator, with_bc=with_bc)
    
    print(f"Dataset loaded:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Quark jets (y=1): {(y == 1).sum():,}")
    print(f"  Gluon jets (y=0): {(y == 0).sum():,}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    save_path = output_path / output_name
    np.savez(save_path, X=X, y=y)
    
    print(f"\nDataset saved to: {save_path.absolute()}")
    print(f"File size: {save_path.stat().st_size / (1024**2):.2f} MB")


def main() -> None:
    """Parse command-line arguments and download dataset."""
    parser = argparse.ArgumentParser(
        description="Download quark/gluon jet dataset from EnergyFlow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "-n",
        "--num-data",
        type=int,
        default=100_000,
        help="Number of jets to load",
    )
    
    parser.add_argument(
        "-g",
        "--generator",
        type=str,
        choices=["pythia", "herwig"],
        default="pythia",
        help="Generator to use",
    )
    
    parser.add_argument(
        "--with-bc",
        action="store_true",
        help="Include bottom and charm quarks",
    )
    
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save the dataset",
    )
    
    parser.add_argument(
        "--output-name",
        type=str,
        default="qg_jets.npz",
        help="Filename for the saved dataset",
    )
    
    args = parser.parse_args()
    
    download_qg_jets(
        num_data=args.num_data,
        generator=args.generator,
        with_bc=args.with_bc,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()

