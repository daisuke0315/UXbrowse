#!/usr/bin/env python3
"""Browser-based local launcher for iPhone live localization and CT display QC.

The system macOS Tk build can show a blank window under VS Code/debugpy.
This launcher serves only a short-lived form on 127.0.0.1, then runs the
existing camera pipeline in the same Python process after the user starts it.
"""

from __future__ import annotations

import html
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from realtime_pnp import (DEFAULT_COORDINATES, DEFAULT_REGISTERED_MAP,
                          DEFAULT_SPLAT_VIEWS, DEFAULT_VOLUME, main)


CT_DISPLAY = 'ct-display'
MATCH_ONLY = 'match-only'
LEGACY_COMPARE = 'legacy-compare'


def build_live_arguments(mode: str, camera: str, top_k: str, map_path: str,
                         views_path: str, volume_path: str,
                         coordinates_path: str, calibration_path: str) -> list[str]:
    """Validate the selected dataset and make one unambiguous runtime preset."""
    if mode not in (CT_DISPLAY, MATCH_ONLY, LEGACY_COMPARE):
        raise ValueError('起動モードを選択してください。')
    try:
        index = int(camera)
        selected_top_k = int(top_k)
    except ValueError as exc:
        raise ValueError('カメラ番号とTop-Kには整数を入力してください。') from exc
    if index < 0 or selected_top_k < 1:
        raise ValueError('カメラ番号は0以上、Top-Kは1以上にしてください。')

    views = Path(views_path).expanduser()
    if not (views / 'manifest.json').is_file():
        raise ValueError(f'3DGS参照画像が見つかりません：\n{views}')
    registered = mode != LEGACY_COMPARE
    object_map = Path(map_path).expanduser()
    if registered and not ((object_map / 'summary.json').is_file()
                           and (object_map / 'object_aliked_map.npz').is_file()):
        raise ValueError(f'登録ALIKED地図が見つかりません：\n{object_map}')

    volume = Path(volume_path).expanduser()
    coordinates = Path(coordinates_path).expanduser()
    if mode == CT_DISPLAY and not (volume.is_file() and coordinates.is_file()):
        raise ValueError('CTボリュームと座標メタデータの両方を指定してください。')
    calibration = calibration_path.strip()
    if calibration and not Path(calibration).expanduser().is_file():
        raise ValueError(f'iPhone校正ファイルが見つかりません：\n{calibration}')

    options = [
        '--camera', str(index), '--capture-backend', 'opencv',
        '--width', '0', '--height', '0', '--process-fps', '15',
        '--max-processing-width', '1280',
        '--splat-views', str(views), '--glb-top-k', str(selected_top_k),
    ]
    if mode == CT_DISPLAY:
        options.extend([
            '--ct-validation', '--volume', str(volume),
            '--coordinates', str(coordinates), '--mpr-size', '512',
        ])
        if calibration:
            options.extend(['--calibration', str(Path(calibration).expanduser())])
    else:
        options.append('--no-mpr')
    if registered:
        options.extend([
            '--matching-mode', 'registered-aliked',
            '--registered-map', str(object_map),
            '--virtual-max-coarse-rotation', '55',
            '--ransac-threshold', '6',
        ])
    else:
        options.extend([
            '--matching-mode', 'splat-track',
            '--retrieval-mode', 'dino-global',
            '--retrieval-auto-roi',
        ])
    return options


DEFAULT_FORM = {
    'mode': CT_DISPLAY,
    'camera': '1',
    'top_k': '5',
    'map_path': str(DEFAULT_REGISTERED_MAP),
    'views_path': str(DEFAULT_SPLAT_VIEWS),
    'volume_path': str(DEFAULT_VOLUME),
    'coordinates_path': str(DEFAULT_COORDINATES),
    'calibration_path': '',
}


def render_form(values: dict[str, str], token: str, error: str = '') -> str:
    """Return the self-contained local settings page; escape every user value."""
    data = {key: html.escape(values.get(key, default), quote=True)
            for key, default in DEFAULT_FORM.items()}
    chosen = values.get('mode', CT_DISPLAY)
    checked = {key: 'checked' if chosen == key else ''
               for key in (CT_DISPLAY, MATCH_ONLY, LEGACY_COMPARE)}
    error_html = (f'<div class="error" role="alert">{html.escape(error)}</div>'
                  if error else '')
    template = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ViewSync — iPhoneライブ実験</title>
<style>
  :root { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
          color: #18202a; background: #f3f6fa; }
  body { margin: 0; padding: 32px 16px; }
  main { max-width: 760px; margin: auto; padding: 28px; background: white;
         border: 1px solid #d7e0e8; border-radius: 16px;
         box-shadow: 0 8px 28px #17314c18; }
  h1 { margin: 0 0 6px; font-size: 26px; }
  .intro { margin: 0 0 22px; color: #51606e; }
  .choice { display: flex; gap: 13px; align-items: flex-start; cursor: pointer;
            padding: 15px 16px; border: 1px solid #d7e0e8; border-radius: 11px;
            margin: 9px 0; }
  .choice:has(input:checked) { background: #eaf4ff; border-color: #3386d7; }
  .choice input { margin-top: 4px; transform: scale(1.2); }
  .choice strong { display: block; font-size: 17px; }
  .choice small { display: block; color: #536473; margin-top: 5px; line-height: 1.45; }
  .row { margin: 17px 0; }
  .row label { display: block; font-weight: 600; margin-bottom: 6px; }
  input[type=text], input[type=number] { box-sizing: border-box; width: 100%;
      padding: 10px 11px; border: 1px solid #bcc9d4; border-radius: 7px; font-size: 15px; }
  input.short { width: 100px; }
  details { margin-top: 18px; border-top: 1px solid #e0e6eb; padding-top: 13px; }
  summary { cursor: pointer; font-weight: 600; }
  .note { color: #5b4d23; background: #fff8dc; padding: 12px;
          border-radius: 8px; line-height: 1.5; margin-top: 16px; }
  .error { color: #81231e; background: #fff0ef; border: 1px solid #e5aba7;
           padding: 12px; border-radius: 8px; white-space: pre-wrap; }
  .actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 23px; }
  button { padding: 11px 20px; border: 0; border-radius: 8px;
           background: #146db2; color: white; font-size: 15px; cursor: pointer; }
  button:hover { background: #095a9a; }
  button.secondary { background: #e6ebef; color: #18202a; }
</style></head><body><main>
<h1>iPhoneライブ実験</h1>
<p class="intro">目的を選んで起動してください。通常は①のままで大丈夫です。</p>
__ERROR__
<form method="post" action="/start">
<input type="hidden" name="token" value="__TOKEN__">
<label class="choice"><input type="radio" name="mode" value="ct-display" __CT_CHECKED__>
  <span><strong>① CT重畳の表示ずれを測る（推奨）</strong>
  <small>登録実写ALIKEDでPnP → PASS後、映像画面でVを押して静止 → CT断面と比較</small></span>
</label>
<label class="choice"><input type="radio" name="mode" value="match-only" __MATCH_CHECKED__>
  <span><strong>② 位置合わせだけ試す（CTなし）</strong>
  <small>同じ登録実写ALIKED地図でPnPのPASS/FAILを確認します。</small></span>
</label>
<label class="choice"><input type="radio" name="mode" value="legacy-compare" __LEGACY_CHECKED__>
  <span><strong>③ 旧3DGS方式と比較する</strong>
  <small>仮想画像と近似XYZを使う比較用。CTの検証には使用しません。</small></span>
</label>
<div class="row"><label for="camera">カメラ番号（通常は1）</label>
  <input class="short" id="camera" name="camera" type="number" min="0" step="1"
         value="__CAMERA__" required></div>
<div class="note">①では起動直後はライブ映像のみです。PnPがPASSしたら映像画面で
<strong>V</strong>を押すと画像・姿勢が固定され、CT/MPRが表示されます。
既定のCTは260911、ALIKED地図は260918splatです。座標整合は未検証のため、
mm値を精度の証拠にしないでください。研究用・臨床使用不可。</div>
<details><summary>詳細設定：別の地図・CTを使う場合</summary>
  <div class="row"><label for="map_path">登録ALIKED地図</label>
    <input id="map_path" name="map_path" type="text" value="__MAP_PATH__"></div>
  <div class="row"><label for="views_path">3DGS参照画像</label>
    <input id="views_path" name="views_path" type="text" value="__VIEWS_PATH__"></div>
  <div class="row"><label for="volume_path">CTボリューム（.mha）</label>
    <input id="volume_path" name="volume_path" type="text" value="__VOLUME_PATH__"></div>
  <div class="row"><label for="coordinates_path">CT座標メタデータ</label>
    <input id="coordinates_path" name="coordinates_path" type="text"
           value="__COORDINATES_PATH__"></div>
  <div class="row"><label for="calibration_path">iPhone校正（任意、.npz）</label>
    <input id="calibration_path" name="calibration_path" type="text"
           value="__CALIBRATION_PATH__"></div>
  <div class="row"><label for="top_k">Top-K（通常は5）</label>
    <input class="short" id="top_k" name="top_k" type="number" min="1"
           value="__TOP_K__" required></div>
</details>
<div class="actions">
  <button class="secondary" type="submit" formaction="/cancel" formnovalidate>キャンセル</button>
  <button type="submit">選んだ目的で起動</button>
</div></form></main></body></html>"""
    replacements = {
        '__ERROR__': error_html,
        '__TOKEN__': html.escape(token, quote=True),
        '__CT_CHECKED__': checked[CT_DISPLAY],
        '__MATCH_CHECKED__': checked[MATCH_ONLY],
        '__LEGACY_CHECKED__': checked[LEGACY_COMPARE],
        '__CAMERA__': data['camera'],
        '__MAP_PATH__': data['map_path'],
        '__VIEWS_PATH__': data['views_path'],
        '__VOLUME_PATH__': data['volume_path'],
        '__COORDINATES_PATH__': data['coordinates_path'],
        '__CALIBRATION_PATH__': data['calibration_path'],
        '__TOP_K__': data['top_k'],
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def choose_settings():
    token = secrets.token_urlsafe(24)
    selection: list[str] = []
    server = None

    class SettingsHandler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            pass

        def send_html(self, content: str, status: int = 200):
            payload = content.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != '/':
                self.send_error(404)
                return
            self.send_html(render_form(DEFAULT_FORM, token))

        def do_POST(self):
            if self.path not in ('/start', '/cancel'):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 1 or length > 32_768:
                    raise ValueError('Invalid form length')
                form_data = self.rfile.read(length).decode('utf-8')
                parsed = parse_qs(form_data, keep_blank_values=True)
                values = {key: items[0] for key, items in parsed.items()}
            except (ValueError, UnicodeError):
                self.send_error(400)
                return
            if not secrets.compare_digest(values.get('token', ''), token):
                self.send_error(403)
                return
            if self.path == '/cancel':
                self.send_html('<!doctype html><meta charset="utf-8"><p>起動を中止しました。</p>')
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            try:
                options = build_live_arguments(
                    values.get('mode', ''), values.get('camera', ''),
                    values.get('top_k', ''), values.get('map_path', ''),
                    values.get('views_path', ''), values.get('volume_path', ''),
                    values.get('coordinates_path', ''),
                    values.get('calibration_path', ''))
            except ValueError as exc:
                self.send_html(render_form(values, token, str(exc)), 400)
                return
            selection.extend(options)
            self.send_html('<!doctype html><meta charset="utf-8"><title>ViewSync</title>'
                           '<p>カメラを起動します。このタブは閉じて構いません。</p>')
            threading.Thread(target=server.shutdown, daemon=True).start()

    server = ThreadingHTTPServer(('127.0.0.1', 0), SettingsHandler)
    server.daemon_threads = True
    address = f'http://127.0.0.1:{server.server_port}/'
    print(f'ViewSync settings: {address}', flush=True)
    print('If the browser does not open, paste this address into a browser.', flush=True)
    webbrowser.open(address, new=2)
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        server.server_close()
    if not selection:
        print('ViewSync settings closed without starting the camera.', flush=True)
    return selection or None


if __name__ == '__main__':
    options = choose_settings()
    if options is not None:
        print('Live test options:', ' '.join(options), flush=True)
        sys.exit(main(options))
