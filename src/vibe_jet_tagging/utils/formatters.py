"""Data formatters for converting jet data to text representations."""

import re
from pathlib import Path
from typing import Any


def format_jet_as_list(jet: Any) -> str:
    """
    Format jet particles as a simple list.
    
    Parameters
    ----------
    jet : array-like
        Jet data with shape (n_particles, 4) where features are [pt, y, phi, pid]
    
    Returns
    -------
    str
        Formatted particle list
    """
    lines = []
    for i, particle in enumerate(jet):
        if len(particle) == 4 and particle[0] > 0:  # Only include real particles (pt > 0)
            pt, y, phi, pid = particle
            lines.append(f"Particle {i+1}: pt={pt:.3f} GeV, y={y:.3f}, phi={phi:.3f}, pid={int(pid)}")
    
    return "\n".join(lines) if lines else "No particles"


def format_jet_as_yaml(jet: Any) -> str:
    """
    Format jet particles as YAML.
    
    Parameters
    ----------
    jet : array-like
        Jet data with shape (n_particles, 4) where features are [pt, y, phi, pid]
    
    Returns
    -------
    str
        YAML-formatted jet data
    """
    lines = ["particles:"]
    for i, particle in enumerate(jet):
        if len(particle) == 4 and particle[0] > 0:  # Only real particles
            pt, y, phi, pid = particle
            lines.append(f"  - index: {i+1}")
            lines.append(f"    pt: {pt:.3f}")
            lines.append(f"    rapidity: {y:.3f}")
            lines.append(f"    phi: {phi:.3f}")
            lines.append(f"    pid: {int(pid)}")
    
    return "\n".join(lines) if len(lines) > 1 else "particles: []"


def format_jet_as_table(jet: Any) -> str:
    """
    Format jet particles as a table.
    
    Parameters
    ----------
    jet : array-like
        Jet data with shape (n_particles, 4) where features are [pt, y, phi, pid]
    
    Returns
    -------
    str
        Table-formatted jet data
    """
    lines = ["| Index | pt (GeV) | Rapidity | Phi (rad) | PID |"]
    lines.append("|-------|----------|----------|-----------|-----|")
    
    for i, particle in enumerate(jet):
        if len(particle) == 4 and particle[0] > 0:  # Only real particles
            pt, y, phi, pid = particle
            lines.append(f"| {i+1:5d} | {pt:8.3f} | {y:8.3f} | {phi:9.3f} | {int(pid):3d} |")
    
    return "\n".join(lines)


def load_template(template_name: str, templates_dir: str = "templates") -> str:
    """
    Load a prompt template from file.
    
    Parameters
    ----------
    template_name : str
        Name of the template file (without .txt extension)
    templates_dir : str
        Directory containing templates
    
    Returns
    -------
    str
        Template content
    """
    template_path = Path(templates_dir) / f"{template_name}.txt"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    return template_path.read_text()


def fill_template(template: str, jet: Any, format_type: str = "list") -> str:
    """
    Fill a template with formatted jet data.
    
    Parameters
    ----------
    template : str
        Template string with placeholders
    jet : array-like
        Jet data
    format_type : str
        Format to use: 'list', 'yaml', or 'table'
    
    Returns
    -------
    str
        Filled template
    """
    formatters = {
        "list": format_jet_as_list,
        "yaml": format_jet_as_yaml,
        "table": format_jet_as_table,
    }
    
    if format_type not in formatters:
        raise ValueError(f"Unknown format type: {format_type}")
    
    formatter = formatters[format_type]
    formatted_jet = formatter(jet)
    
    # Replace placeholders in template
    replacements = {
        "{{jet_particles}}": formatted_jet,
        "{{jet_yaml}}": formatted_jet,
        "{{jet_table}}": formatted_jet,
    }
    
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result


def format_features_as_text(features: dict[str, float]) -> str:
    """
    Format extracted features as human-readable text for LLM prompts.
    
    Parameters
    ----------
    features : dict[str, float]
        Dictionary of feature names to values
    
    Returns
    -------
    str
        Formatted feature text
    """
    lines = ["Jet Features:"]
    
    # Format each feature with appropriate units and precision
    feature_formats = {
        'multiplicity': ('Multiplicity', '{:.0f} particles', 1),
        'mean_pt': ('Mean pT', '{:.2f} GeV', 1),
        'std_pt': ('pT Std Dev', '{:.2f} GeV', 1),
        'median_pt': ('Median pT', '{:.2f} GeV', 1),
        'max_pt': ('Max pT', '{:.2f} GeV', 1),
        'lead_pt_frac': ('Leading pT Fraction', '{:.3f}', 1),
        'top3_pt_frac': ('Top-3 pT Fraction', '{:.3f}', 1),
        'top5_pt_frac': ('Top-5 pT Fraction', '{:.3f}', 1),
    }
    
    for key, value in features.items():
        if key in feature_formats:
            name, fmt, indent = feature_formats[key]
            formatted_value = fmt.format(value)
            lines.append(f"  {name}: {formatted_value}")
        else:
            # Fallback for unknown features
            lines.append(f"  {key}: {value:.3f}")
    
    return "\n".join(lines)


def infer_required_features(template: str) -> set[str]:
    """
    Infer which features are required by parsing template placeholders.
    
    Scans for placeholders like:
    - {{jet_features}} - generic features placeholder
    - {{multiplicity}} - specific feature
    - {{mean_pt}} - specific feature
    
    Parameters
    ----------
    template : str
        Template string to parse
    
    Returns
    -------
    set[str]
        Set of required feature names, or 'features' for generic placeholder
    """
    # Find all placeholders in format {{...}}
    placeholders = re.findall(r'\{\{([^}]+)\}\}', template)
    
    required = set()
    
    for placeholder in placeholders:
        placeholder = placeholder.strip()
        
        # Generic features placeholder
        if placeholder == 'jet_features':
            required.add('features')
        # Specific feature placeholders
        elif placeholder in [
            'multiplicity', 'mean_pt', 'std_pt', 'median_pt', 'max_pt',
            'lead_pt_frac', 'top3_pt_frac', 'top5_pt_frac'
        ]:
            required.add(placeholder)
    
    return required


def select_extractor_for_template(template: str) -> str:
    """
    Auto-select appropriate feature extractor based on template requirements.
    
    Parameters
    ----------
    template : str
        Template string to analyze
    
    Returns
    -------
    str
        Recommended extractor name: 'basic', 'kinematic', 'concentration', 'full', or 'none'
    """
    required = infer_required_features(template)
    
    # If no features required, use none
    if not required:
        return 'none'
    
    # If generic features placeholder, use full extractor
    if 'features' in required:
        return 'full'
    
    # Check which specific features are needed
    basic_features = {'multiplicity'}
    kinematic_features = {'mean_pt', 'std_pt', 'median_pt', 'max_pt'}
    concentration_features = {'lead_pt_frac', 'top3_pt_frac', 'top5_pt_frac'}
    
    needs_kinematic = bool(required & kinematic_features)
    needs_concentration = bool(required & concentration_features)
    
    # Select based on what's needed
    if needs_kinematic and needs_concentration:
        return 'full'
    elif needs_kinematic:
        return 'kinematic'
    elif needs_concentration:
        return 'concentration'
    elif required & basic_features:
        return 'basic'
    else:
        return 'none'

