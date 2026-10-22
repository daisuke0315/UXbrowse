"""Local macOS picker. Fixed AppleScript, never interpolate browser input.

Only an authenticated button click invokes the dialog. Files stay on this Mac;
large maps/CT are not uploaded to or copied by the browser.
"""
from pathlib import Path
import subprocess
import sys
import threading

from map_preprocessing import audited_map

PICKERS = {
    'map_path': ('登録実写地図の object_aliked_map.npz または summary.json を選択', {'.npz', '.json'}),
    'views_path': ('3DGS参照フォルダ内の manifest.json を選択', {'.json'}),
    'volume_path': ('CT volume の .mha ファイルを選択', {'.mha'}),
    'coordinates_path': ('座標 metadata の .json ファイルを選択', {'.json'}),
    'calibration_path': ('カメラ校正の .json ファイルを選択', {'.json'}),
}
LOCK = threading.Lock()
SCRIPT = '''on run argv
    try
        activate
        set chosen to choose file with prompt (item 1 of argv)
        return POSIX path of chosen
    on error number -128
        return ""
    end try
end run'''


def choose_local_file(field):
    if field not in PICKERS:
        raise ValueError('Unknown file selection field')
    if sys.platform != 'darwin':
        raise ValueError('このファイル選択はMacで利用できます。')
    if not LOCK.acquire(blocking=False):
        raise ValueError('開いているファイル選択を完了してください。')
    try:
        prompt, extensions = PICKERS[field]
        result = subprocess.run(['/usr/bin/osascript', '-e', SCRIPT, prompt],
                                capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise ValueError('ファイル選択を開けませんでした：' + result.stderr.strip()[:500])
        if not result.stdout.strip():
            return dict(cancelled=True)
        path = Path(result.stdout.strip()).resolve()
        if not path.is_file() or path.suffix.lower() not in extensions:
            raise ValueError('選択したファイルの種類が違います。' + prompt)
        if field == 'map_path':
            if path.name not in ('object_aliked_map.npz', 'summary.json'):
                raise ValueError('object_aliked_map.npz または summary.json を選択してください。')
            folder, report = audited_map(path.parent)
            return dict(path=str(folder), cancelled=False, name=folder.name,
                        xyz_ratio=report['xyz_ratio'], xyz_observations=report['xyz_observations'])
        if field == 'views_path':
            if path.name != 'manifest.json':
                raise ValueError('3DGS参照フォルダの manifest.json を選択してください。')
            path = path.parent
        return dict(path=str(path), cancelled=False, name=path.name)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('ファイル選択がタイムアウトしました。もう一度選択してください。') from exc
    finally:
        LOCK.release()
