"""Camera-free GS comparison; optional, isolated RGB reference capture."""

from macar_paths import DATA_DIR
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
from pathlib import Path
import shutil
import secrets
from urllib.parse import urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = DATA_DIR / "output_3dgs_reference_views_72_upper_negz_v2/manifest.json"


class ReferenceCapture:
    """Save only enumerated RGBs and preserve the source geometry byte-for-byte."""
    GAINS = {'alpha10': 1.0, 'alpha15': 1.5}

    def __init__(self, manifest, output_prefix):
        self.manifest = Path(manifest).resolve()
        self.data = json.loads(self.manifest.read_text())
        ply = Path(self.data['ply'])
        if not ply.is_absolute():
            self.data['ply'] = str((self.manifest.parent / ply).resolve())
        self.outputs = {key: Path(str(output_prefix) + '_' + key).resolve()
                        for key in self.GAINS}
        if any(path.exists() for path in self.outputs.values()):
            raise FileExistsError('Reference output already exists; choose a new prefix')
        for view in self.data['views']:
            for field in ('image_name', 'data_name'):
                if Path(view[field]).name != view[field]:
                    raise ValueError('Only flat manifest filenames can be captured')
            if not (self.manifest.parent / view['data_name']).is_file():
                raise FileNotFoundError(view['data_name'])
        for path in self.outputs.values():
            path.mkdir(parents=True)
        self.saved = set()
        self.finalized = False
        self.token = secrets.token_urlsafe(32)

    def save(self, condition, index, pixels):
        from PIL import Image
        if self.finalized:
            raise ValueError('Reference capture is already finalized')
        if condition not in self.outputs or not 0 <= index < len(self.data['views']):
            raise ValueError('Unknown capture target')
        try:
            with Image.open(io.BytesIO(pixels)) as image:
                if image.format != 'PNG' or image.size != (self.data['width'], self.data['height']):
                    raise ValueError('Expected reference-sized PNG')
                image.verify()
        except Exception as error:
            raise ValueError('Invalid capture PNG') from error
        target = self.outputs[condition] / self.data['views'][index]['image_name']
        target.write_bytes(pixels)
        self.saved.add((condition, index))

    def finish(self):
        if self.finalized:
            raise ValueError('Reference capture is already finalized')
        if len(self.saved) != len(self.outputs) * len(self.data['views']):
            raise ValueError('Reference capture is incomplete')
        for key, path in self.outputs.items():
            for view in self.data['views']:
                shutil.copyfile(self.manifest.parent / view['data_name'], path / view['data_name'])
            data = dict(self.data, rgb_alpha_gain=self.GAINS[key],
                        rgb_renderer='playcanvas-browser-gaussian-rasterizer',
                        rgb_source_manifest=str(self.manifest),
                        rgb_background=[.96, .96, .96],
                        rgb_capture='settled-reference-camera-rgb-only')
            (path / 'manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
        self.finalized = True
        return list(self.outputs.values())


def comparison_config(manifest):
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    ply = Path(data["ply"])
    if not ply.is_absolute():
        ply = manifest.parent / ply
    ply = ply.resolve()
    if not ply.is_file():
        raise FileNotFoundError(f"PLY not found: {ply}")
    return dict(ply=str(ply), manifest=str(manifest), center=data["center"],
                radius=data["radius"], up=[0, 0, -1 if data.get("up_axis") == "negative-z" else 1],
                width=data.get("width", 960), height=data.get("height", 720),
                fov=data.get("vertical_fov_deg", 65), views=data["views"])


def comparison_files(manifest):
    config = comparison_config(manifest)
    directory = ROOT / "browser_ui/gs-compare"
    files = {"/": (directory / "index.html", "text/html; charset=utf-8"),
             "/asset.ply": (Path(config["ply"]), "application/octet-stream"),
             "/playcanvas.mjs": (ROOT / ".tools/gs-browser/node_modules/playcanvas/build/playcanvas.mjs",
                                  "text/javascript; charset=utf-8")}
    for name in ("compare.js", "profiles.js"):
        files["/" + name] = (directory / name, "text/javascript; charset=utf-8")
    return files


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, config, files, capture=None, **kwargs):
        self.config, self.files, self.capture = config, files, capture
        super().__init__(*args, **kwargs)

    def do_POST(self):
        # Capture is disabled in the ordinary read-only viewer, and cross-origin
        # pages cannot write: the token is sent in a non-simple request header.
        if self.capture is None or self.headers.get('X-Capture-Token') != self.capture.token:
            self.send_error(403)
            return
        try:
            route = urlsplit(self.path).path.split('/')
            if route == ['', 'capture', 'finish']:
                paths = self.capture.finish()
                body = json.dumps(dict(outputs=[str(path) for path in paths])).encode()
            elif len(route) == 4 and route[1] == 'capture':
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 8 * 1024 * 1024:
                    raise ValueError('Invalid PNG request size')
                self.capture.save(route[2], int(route[3]), self.rfile.read(size))
                body = b'{}'
            else:
                raise ValueError('Unknown capture request')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, OSError) as error:
            self.send_error(400, str(error))

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/config.json":
            body = json.dumps(self.config, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        entry = self.files.get(route)
        if not entry:
            self.send_error(404)
            return
        path, content_type = entry
        if not path.is_file():
            self.send_error(404, "Required renderer file is missing")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--reference-output", type=Path,
                        help="Opt-in RGB capture prefix (two new folders; no overwrites)")
    args = parser.parse_args()
    config, files = comparison_config(args.manifest), comparison_files(args.manifest)
    capture = ReferenceCapture(args.manifest, args.reference_output) if args.reference_output else None
    if capture:
        config['capture'] = dict(token=capture.token, outputs={key: str(path)
                                                              for key, path in capture.outputs.items()})
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                partial(Handler, config=config, files=files, capture=capture))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"GS comparison: {url}\nPLY (read-only): {config['ply']}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
