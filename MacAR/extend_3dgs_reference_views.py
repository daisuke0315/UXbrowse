#!/usr/bin/env python3
"""Append compatible rendered 3DGS views while preserving existing view IDs."""

import argparse
import json
import os
from pathlib import Path


COMPATIBILITY_FIELDS = (
    'format_version', 'source_type', 'ply', 'coordinate_system',
    'runtime_transform', 'up_axis', 'geometry_renderer', 'pixel_convention',
    'width', 'height', 'vertical_fov_deg', 'bounds', 'center', 'radius',
    'minimum_opacity', 'rgb_renderer',
)


def extend_views(base: Path, addition: Path, output: Path):
    base, addition, output = (path.expanduser().resolve() for path in
                              (base, addition, output))
    if addition == base or addition == output:
        raise ValueError('The additional views must be in a separate directory')
    original = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    extra = json.loads((addition / 'manifest.json').read_text(encoding='utf-8'))
    mismatches = [key for key in COMPATIBILITY_FIELDS if original.get(key) != extra.get(key)]
    if mismatches:
        raise ValueError(f'Incompatible 3DGS view settings: {", ".join(mismatches)}')
    old_views = original['views']
    new_views = extra['views']
    if any(view['view_id'] != index for index, view in enumerate(old_views)):
        raise ValueError('Base view IDs are not consecutive')
    if any(view['view_id'] != index for index, view in enumerate(new_views)):
        raise ValueError('Additional view IDs are not consecutive')
    signature = lambda view: (round(view['azimuth_deg'], 5),
                              round(view['elevation_deg'], 5),
                              round(view['distance_multiplier'], 5),
                              round(view.get('roll_deg', 0.), 5))
    existing = {signature(view) for view in old_views}
    incoming = [signature(view) for view in new_views]
    if len(set(incoming)) != len(incoming) or existing.intersection(incoming):
        raise ValueError('Additional views duplicate existing camera positions')

    in_place = output == base
    if not in_place and output.exists():
        raise ValueError(f'Output already exists: {output}')
    if not in_place:
        output.mkdir(parents=True)
    else:
        backup = output / f'manifest.before_{len(old_views)}views.json'
        if backup.exists():
            raise ValueError(f'Manifest backup already exists: {backup}')
        backup.write_bytes((base / 'manifest.json').read_bytes())
    links = []
    combined = list(old_views)
    try:
        if not in_place:
            for view in old_views:
                for key in ('image_name', 'data_name'):
                    source = base / view[key]
                    target = output / view[key]
                    os.link(source, target)
                    links.append(target)
        for view in new_views:
            index = len(combined)
            name = f'view_{index:04d}'
            appended = dict(view, view_id=index, image_name=f'{name}.png',
                            data_name=f'{name}.npz')
            for key in ('image_name', 'data_name'):
                source = addition / view[key]
                target = output / appended[key]
                os.link(source, target)
                links.append(target)
            combined.append(appended)
        merged = dict(original, views=combined,
                      additional_view_groups=list(original.get('additional_view_groups', [])) + [{
                          'first_view_id': len(old_views), 'count': len(new_views),
                          'elevations_deg': sorted({view['elevation_deg'] for view in new_views}),
                          'rolls_deg': sorted({view['roll_deg'] for view in new_views
                                               if 'roll_deg' in view}),
                      }])
        manifest = output / 'manifest.json'
        staged = output / 'manifest.next.json'
        staged.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n',
                          encoding='utf-8')
        os.replace(staged, manifest)
    except Exception:
        for path in reversed(links):
            path.unlink(missing_ok=True)
        raise
    return len(old_views), len(combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--addition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New folder, or the base folder to append in place')
    args = parser.parse_args()
    before, after = extend_views(args.base, args.addition, args.output)
    print(f'3DGS reference views: {before} + {after-before} = {after}; '
          f'manifest: {args.output / "manifest.json"}')


if __name__ == '__main__':
    main()
