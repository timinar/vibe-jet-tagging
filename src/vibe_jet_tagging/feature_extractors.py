"""Feature extractors for jet data."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import numpy.typing as npt


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    
    Feature extractors convert raw jet data (particles) into 
    high-level features for LLM prompts.
    """
    
    @abstractmethod
    def extract(self, jet: npt.NDArray[np.float32]) -> dict[str, float]:
        """
        Extract features from a single jet.
        
        Parameters
        ----------
        jet : np.ndarray
            Jet data with shape (n_particles, 4) where features are [pt, y, phi, pid]
        
        Returns
        -------
        dict[str, float]
            Dictionary of feature names to values
        """
        pass
    
    def extract_batch(self, jets: npt.NDArray[np.float32]) -> list[dict[str, float]]:
        """
        Extract features from multiple jets.
        
        Parameters
        ----------
        jets : np.ndarray
            Jet data with shape (n_jets, n_particles, 4)
        
        Returns
        -------
        list[dict[str, float]]
            List of feature dictionaries, one per jet
        """
        return [self.extract(jet) for jet in jets]
    
    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return list of feature names this extractor provides."""
        pass


class BasicExtractor(FeatureExtractor):
    """Extract only multiplicity (particle count)."""
    
    def extract(self, jet: npt.NDArray[np.float32]) -> dict[str, float]:
        """
        Extract multiplicity feature.
        
        Parameters
        ----------
        jet : np.ndarray
            Jet data with shape (n_particles, 4)
        
        Returns
        -------
        dict[str, float]
            Dictionary with 'multiplicity' key
        """
        # Count particles with pt > 0 (non-padded)
        multiplicity = int(np.sum(jet[:, 0] > 0))
        
        return {'multiplicity': multiplicity}
    
    @property
    def feature_names(self) -> list[str]:
        """Return feature names."""
        return ['multiplicity']


class KinematicExtractor(FeatureExtractor):
    """Extract kinematic features: multiplicity and pt statistics."""
    
    def extract(self, jet: npt.NDArray[np.float32]) -> dict[str, float]:
        """
        Extract kinematic features.
        
        Parameters
        ----------
        jet : np.ndarray
            Jet data with shape (n_particles, 4)
        
        Returns
        -------
        dict[str, float]
            Dictionary with multiplicity and pt statistics
        """
        # Get non-padded particles
        mask = jet[:, 0] > 0
        pt = jet[mask, 0]
        
        if len(pt) == 0:
            # Edge case: empty jet
            return {
                'multiplicity': 0,
                'mean_pt': 0.0,
                'std_pt': 0.0,
                'median_pt': 0.0,
                'max_pt': 0.0,
            }
        
        return {
            'multiplicity': len(pt),
            'mean_pt': float(np.mean(pt)),
            'std_pt': float(np.std(pt)),
            'median_pt': float(np.median(pt)),
            'max_pt': float(np.max(pt)),
        }
    
    @property
    def feature_names(self) -> list[str]:
        """Return feature names."""
        return ['multiplicity', 'mean_pt', 'std_pt', 'median_pt', 'max_pt']


class ConcentrationExtractor(FeatureExtractor):
    """Extract pt concentration features (top-N particle fractions)."""
    
    def extract(self, jet: npt.NDArray[np.float32]) -> dict[str, float]:
        """
        Extract pt concentration features.
        
        Parameters
        ----------
        jet : np.ndarray
            Jet data with shape (n_particles, 4)
        
        Returns
        -------
        dict[str, float]
            Dictionary with pt concentration features
        """
        # Get non-padded particles
        mask = jet[:, 0] > 0
        pt = jet[mask, 0]
        
        if len(pt) == 0:
            return {
                'lead_pt_frac': 0.0,
                'top3_pt_frac': 0.0,
                'top5_pt_frac': 0.0,
            }
        
        # Sort pt in descending order
        pt_sorted = np.sort(pt)[::-1]
        pt_sum = pt.sum()
        
        if pt_sum == 0:
            return {
                'lead_pt_frac': 0.0,
                'top3_pt_frac': 0.0,
                'top5_pt_frac': 0.0,
            }
        
        return {
            'lead_pt_frac': float(pt_sorted[0] / pt_sum),
            'top3_pt_frac': float(pt_sorted[:3].sum() / pt_sum),
            'top5_pt_frac': float(pt_sorted[:5].sum() / pt_sum),
        }
    
    @property
    def feature_names(self) -> list[str]:
        """Return feature names."""
        return ['lead_pt_frac', 'top3_pt_frac', 'top5_pt_frac']


class FullExtractor(FeatureExtractor):
    """Extract all available features by combining all extractors."""
    
    def __init__(self):
        """Initialize with all component extractors."""
        self.basic = BasicExtractor()
        self.kinematic = KinematicExtractor()
        self.concentration = ConcentrationExtractor()
    
    def extract(self, jet: npt.NDArray[np.float32]) -> dict[str, float]:
        """
        Extract all features.
        
        Parameters
        ----------
        jet : np.ndarray
            Jet data with shape (n_particles, 4)
        
        Returns
        -------
        dict[str, float]
            Dictionary with all features
        """
        features = {}
        
        # Kinematic features include multiplicity, so use those
        features.update(self.kinematic.extract(jet))
        features.update(self.concentration.extract(jet))
        
        return features
    
    @property
    def feature_names(self) -> list[str]:
        """Return all feature names."""
        # Kinematic includes multiplicity
        return self.kinematic.feature_names + self.concentration.feature_names


def get_extractor(extractor_name: str) -> FeatureExtractor:
    """
    Get feature extractor by name.
    
    Parameters
    ----------
    extractor_name : str
        Name of extractor: 'basic', 'kinematic', 'concentration', or 'full'
    
    Returns
    -------
    FeatureExtractor
        Feature extractor instance
    
    Raises
    ------
    ValueError
        If extractor_name is not recognized
    """
    extractors = {
        'basic': BasicExtractor,
        'kinematic': KinematicExtractor,
        'concentration': ConcentrationExtractor,
        'full': FullExtractor,
    }
    
    if extractor_name not in extractors:
        raise ValueError(
            f"Unknown extractor '{extractor_name}'. Valid options: {list(extractors.keys())}"
        )
    
    return extractors[extractor_name]()

