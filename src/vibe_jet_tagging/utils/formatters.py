"""Data formatters for converting jet data to text representations."""

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

