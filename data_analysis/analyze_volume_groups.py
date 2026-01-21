#!/usr/bin/env python3
"""
analyze_volume_groups.py

Analysiert Volume-Mappings und erstellt Gruppen-Zuordnung als JSON

Usage:
    python analyze_volume_groups.py \
        --volume-mapping /path/to/globalPhysVolMappings.json \
        --output ./data/volume_group_mapping.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.volume_groups import VolumeGrouper


def main():
    parser = argparse.ArgumentParser(
        description='Analyze volume grouping for One-Hot encoding'
    )
    parser.add_argument(
        '--volume-mapping',
        required=True,
        help='Path to globalPhysVolMappings.json'
    )
    parser.add_argument(
        '--output',
        default='./data/volume_group_mapping.json',
        help='Output JSON file'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("VOLUME GROUP ANALYSIS")
    print("="*70)
    
    # Load volume mapping
    print(f"\nLoading volume mapping: {args.volume_mapping}")
    with open(args.volume_mapping, 'r') as f:
        volume_mapping_raw = json.load(f)
    
    # Invert mapping: {name: id} → {id: name}
    volume_mapping = {}
    for vol_name, vol_id in volume_mapping_raw.items():
        if isinstance(vol_id, int):
            volume_mapping[vol_id] = vol_name
    
    print(f"  Total volumes: {len(volume_mapping):,}")
    
    # Handle empty string as "noVolume"
    if "" in volume_mapping_raw:
        empty_id = volume_mapping_raw[""]
        volume_mapping[empty_id] = "noVolume"
        print(f"  Mapped empty string (ID={empty_id}) → 'noVolume'")
    
    print(f"  Total volumes: {len(volume_mapping):,}")
    
    # Initialize grouper
    grouper = VolumeGrouper()
    print(f"\n  Defined groups: {grouper.n_groups}")
    
    # Build lookup
    print("\nMatching volumes to groups...")
    try:
        lookup, group_contents = grouper.build_lookup_dict(volume_mapping)
    except ValueError as e:
        print(f"\n✗ ERROR: {e}")
        return 1
    
    # Statistics
    print("\n✓ All volumes successfully matched!")
    print(f"\nGroup statistics:")
    
    # Sort groups by volume count (descending)
    group_stats = [(name, len(volumes)) for name, volumes in group_contents.items()]
    group_stats.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Group Name':<30} {'Count':>10}")
    print("-"*42)
    for group_name, count in group_stats[:20]:  # Top 20
        if count > 0:
            print(f"{group_name:<30} {count:>10,}")
    
    # Empty groups
    empty_groups = [name for name, volumes in group_contents.items() if len(volumes) == 0]
    if empty_groups:
        print(f"\nEmpty groups ({len(empty_groups)}):")
        for name in empty_groups[:10]:
            print(f"  - {name}")
        if len(empty_groups) > 10:
            print(f"  ... and {len(empty_groups) - 10} more")
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metadata': {
            'total_volumes': len(volume_mapping),
            'total_groups': grouper.n_groups,
            'non_empty_groups': sum(1 for v in group_contents.values() if len(v) > 0)
        },
        'group_names': grouper.group_names,
        'group_contents': group_contents
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Volume group mapping saved: {output_path}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())