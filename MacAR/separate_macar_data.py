"""One-time, non-overwriting relocation of local MacAR data. Stop all jobs first."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import secrets
import shutil

from macar_paths import CODE_DIR, DATA_DIR

DATA_NAMES = frozenset(('output', 'image', 'colmap', 'colmap_image', '260918splat',
                        '.model_cache', '.aliked_cache', '.view_retrieval_cache',
                        'registered_map_selection.json'))
METADATA_NAMES = frozenset(('summary.json', 'manifest.json', 'state.json',
                            'config.json', 'audit.json', 'registered_map_selection.json'))


def relocate_data(code_dir, data_dir, *, apply=False):
    code_dir, data_dir = Path(code_dir).resolve(), Path(data_dir).resolve()
    if not all((code_dir/n).is_file() for n in ('browser_live.py', 'realtime_pnp.py')):
        raise ValueError('Source must be the MacAR code directory')
    if data_dir.is_relative_to(code_dir) or code_dir.is_relative_to(data_dir):
        raise ValueError('Code and data roots must not contain each other')
    items = sorted(p for p in code_dir.iterdir()
                   if p.name in DATA_NAMES or (p.name.startswith('output_') and p.is_dir()))
    if any(p.is_symlink() for p in items):
        raise ValueError('Data root symlinks are not moved automatically')
    moves = [(p, data_dir/p.name) for p in items]
    for _, target in moves:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Refusing to overwrite existing data: {target}')
    result = dict(applied=False, moves=[dict(source=str(a),target=str(b)) for a,b in moves])
    if not apply or not moves:
        return result

    def update(value):
        if isinstance(value, dict): return {k:update(v) for k,v in value.items()}
        if isinstance(value, list): return [update(v) for v in value]
        if isinstance(value, str):
            for source,target in moves:
                prefix=str(source)
                if value == prefix or value.startswith(prefix+'/'):
                    return str(target)+value[len(prefix):]
        return value

    # Parse before moving anything. Research JSONL, report.json, images, NPZ
    # and numerical arrays are never rewritten.
    changes = []
    for source,_ in moves:
        files = source.rglob('*.json') if source.is_dir() else [source]
        for path in files:
            if path.name not in METADATA_NAMES or path.is_symlink(): continue
            original = path.read_bytes()
            value = json.loads(original)
            updated = update(value)
            if value != updated:
                changes.append((path.relative_to(code_dir), original,
                    (json.dumps(updated,ensure_ascii=False,indent=2)+'\n').encode(), path.stat()))
    data_dir.mkdir(parents=True,exist_ok=True)
    record_dir = data_dir/'layout-migrations'/(dt.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(3))
    record_dir.mkdir(parents=True,exist_ok=False)
    result.update(record_dir=str(record_dir), metadata_updated=len(changes))
    for relative,_,_,_ in changes:
        backup=record_dir/'original_metadata'/relative
        backup.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(code_dir/relative,backup)
    moved=[]
    try:
        for source,target in moves:
            if target.exists() or target.is_symlink(): raise FileExistsError(str(target))
            source.rename(target)
            moved.append((source,target))
        for relative,_,updated,stamp in changes:
            path=data_dir/relative
            path.write_bytes(updated)
            os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        result['applied']=True
    except Exception:
        # Restore metadata before reversing the completed atomic directory moves.
        for relative,_,_,_ in changes:
            path=data_dir/relative
            if any(path.is_relative_to(target) for _,target in moved):
                shutil.copy2(record_dir/'original_metadata'/relative,path)
        for source,target in reversed(moved): target.rename(source)
        result['rolled_back']=True
        raise
    finally:
        (record_dir/'relocation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='Move data after stopping all MacAR jobs')
    args=parser.parse_args()
    print(json.dumps(relocate_data(CODE_DIR,DATA_DIR,apply=args.apply),ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
