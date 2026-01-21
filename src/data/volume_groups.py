"""
volume_groups.py

Volume grouping logic for One-Hot encoding
Basiert auf Regex-Patterns für LEGEND detector geometry
"""

import re
from typing import Dict, List, Tuple, Optional


class VolumeGrouper:
    """
    Maps volume names to groups using explicit names and regex patterns
    """
    
    def __init__(self):
        # Explicit single-name groups
        self.explicit_groups = {
            'foot', 'tank', 'skirt', 'water',
            'innercryostat', 'outercryostat', 
            'atmosphericlar', 'undergroundlar',
            'reentrancetube', 'neutronmoderator',
            'vacuumgap', 'noVolume'
        }
        
        # Regex-based patterns with group names
        self.regex_patterns = [
            # Germanium detectors
            ('V_detector', re.compile(r'^V\d{4}$')),
            
            # SiPM modules
            ('S_module_bottom', re.compile(r'^S\d{4}B$')),
            ('S_module_top', re.compile(r'^S\d{4}T$')),
            
            # PMTs
            ('PMT_main', re.compile(r'^PMT\d{6}$')),
            ('PMT_3inch', re.compile(r'^PMT300\d{4}$')),
            
            # Detector penetrations
            ('V_penetration', re.compile(r'^V\d{4}_pen$')),
            
            # SiPM wrappings
            ('S_wrap_bottom', re.compile(r'^S\d{4}B_wrap$')),
            ('S_wrap_top', re.compile(r'^S\d{4}T_wrap$')),
            
            # Fiber system
            ('fiber_sipm', re.compile(r'^fiber_S\d{4}_s$')),
            ('fiber_core', re.compile(r'^fiber_core_l[\d.]+$')),
            ('fiber_cl1', re.compile(r'^fiber_cl1_l[\d.]+$')),
            ('fiber_cl2_tpb', re.compile(r'^fiber_cl2_l[\d.]+_tpb\d+$')),
            
            # PMT components
            ('PMT_vacuum', re.compile(r'^PMT\d{6}_vacuum$')),
            ('PMT_window', re.compile(r'^PMT\d{6}_window$')),
            ('PMT_3inch_vacuum', re.compile(r'^PMT300\d{4}_vacuum$')),
            ('PMT_3inch_window', re.compile(r'^PMT300\d{4}_window$')),
            
            # Detector mounting - consolidated (all click positions → one group)
            ('V_click_top', re.compile(r'^V\d{4}_click_top_\d+$')),
            
            # HV system - consolidated (all strings → one group per component type)
            ('V_hv_cable', re.compile(r'^V\d{4}_hv_cable_string_\d+$')),
            ('V_hv_clamp', re.compile(r'^V\d{4}_hv_clamp_string_\d+$')),
            
            # Signal readout - consolidated (all strings → one group per component type)
            ('V_signal_asic', re.compile(r'^V\d{4}_signal_asic_string_\d+$')),
            ('V_signal_cable', re.compile(r'^V\d{4}_signal_cable_string_\d+$')),
            ('V_signal_clamp', re.compile(r'^V\d{4}_signal_clamp_string_\d+$')),
            
            # Insulator holders - consolidated (all positions → one group)
            ('V_insulator', re.compile(r'^V\d{4}_insulator_du_holder_\d+$')),
            
            # Tyvek reflectors - consolidated (all types → one group)
            ('tyvek', re.compile(r'^(my_)?tyvek_(bot|pit|top|zyl|bottom|prism)_foil$')),
            
            # Copper rods - consolidated (all positions → one group)
            ('cu_rod', re.compile(r'^string_\d+_cu_rod_\d+$')),
            
            # Tristar mounting
            ('tristar_xlarge', re.compile(r'^tristar_xlarge_string_\d+$')),
            
            # String support
            ('string_support', re.compile(r'^string_support_structure_string_\d+$')),
        ]
        
        # Build group name list (for indexing)
        self.group_names = list(self.explicit_groups) + [name for name, _ in self.regex_patterns]
        self.n_groups = len(self.group_names)
    
    def match_volume(self, volume_name: str) -> Tuple[str, int]:
        """
        Match volume name to group
        
        Args:
            volume_name: Volume name string (empty string treated as "noVolume")
        
        Returns:
            (group_name, group_index)
        
        Raises:
            ValueError: If no match found
        """
        # Handle empty string as noVolume
        if volume_name == "":
            volume_name = "noVolume"
        
        # Check explicit groups first
        if volume_name in self.explicit_groups:
            group_idx = self.group_names.index(volume_name)
            return volume_name, group_idx
        
        # Check regex patterns (first match wins)
        for group_name, pattern in self.regex_patterns:
            if pattern.match(volume_name):
                group_idx = self.group_names.index(group_name)
                return group_name, group_idx
        
        # No match found
        raise ValueError(f"Volume '{volume_name}' does not match any group")
    
    def build_lookup_dict(self, volume_mapping: Dict[int, str]) -> Tuple[Dict[int, int], Dict[str, List[str]]]:
        """
        Build lookup dictionary for fast access
        
        Args:
            volume_mapping: {volID_int: volume_name_str}
        
        Returns:
            lookup: {volID_int: group_idx}
            group_contents: {group_name: [volume_names]}
        
        Raises:
            ValueError: If any volume cannot be matched
        """
        lookup = {}
        group_contents = {name: [] for name in self.group_names}
        unmatched = []
        
        for vol_id, vol_name in volume_mapping.items():
            try:
                group_name, group_idx = self.match_volume(vol_name)
                lookup[vol_id] = group_idx
                group_contents[group_name].append(vol_name)
            except ValueError:
                unmatched.append((vol_id, vol_name))
        
        if unmatched:
            unmatched_str = "\n".join([f"  volID={vid}: '{vname}'" for vid, vname in unmatched[:10]])
            if len(unmatched) > 10:
                unmatched_str += f"\n  ... and {len(unmatched) - 10} more"
            
            raise ValueError(
                f"VOLUME GROUPING FAILED: {len(unmatched)} volumes unmatched\n"
                f"{unmatched_str}\n"
                f"→ Add missing patterns to src/data/volume_groups.py"
            )
        
        return lookup, group_contents