"""Run matched native/debugpy/recreation experiments sequentially on MPS.

The debugger uses the installed VS Code debugpy package and an actual DAP
client with justMyCode, no breakpoints, and no variable inspection.
"""

from macar_paths import DATA_DIR
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time


class DebugClient:
    def __init__(self, port, process):
        deadline = time.monotonic()+60
        while True:
            try:
                self.socket = socket.create_connection(('127.0.0.1', port), timeout=1)
                break
            except OSError:
                if time.monotonic() > deadline or process.poll() is not None:
                    raise RuntimeError('debugpy did not listen')
                time.sleep(.1)
        self.socket.settimeout(None)
        self.stream = self.socket.makefile('rb')
        self.messages, self.condition, self.seq = [], threading.Condition(), 0
        self.worker = threading.Thread(target=self.receive, daemon=True)
        self.worker.start()
        self.send('initialize', dict(clientID='macar-profile', adapterID='python',
            pathFormat='path', linesStartAt1=True, columnsStartAt1=True))
        self.until(lambda m: m.get('type') == 'response' and m.get('command') == 'initialize')
        self.send('attach', dict(justMyCode=True))
        self.until(lambda m: m.get('event') == 'initialized')
        self.send('configurationDone', {})
        self.until(lambda m: m.get('type') == 'response' and m.get('command') == 'attach')

    def send(self, command, arguments):
        self.seq += 1
        body = json.dumps(dict(seq=self.seq, type='request', command=command, arguments=arguments)).encode()
        self.socket.sendall(('Content-Length: %d\r\n\r\n' % len(body)).encode()+body)

    def receive(self):
        try:
            while True:
                headers = {}
                while True:
                    line = self.stream.readline()
                    if not line:
                        return
                    if line == b'\r\n':
                        break
                    k, v = line.decode().split(':', 1)
                    headers[k.lower()] = v.strip()
                message = json.loads(self.stream.read(int(headers['content-length'])))
                with self.condition:
                    self.messages.append(message)
                    self.condition.notify_all()
        except (OSError, ValueError):
            return

    def until(self, predicate):
        with self.condition:
            if not self.condition.wait_for(lambda: any(predicate(m) for m in self.messages), timeout=60):
                raise RuntimeError('DAP handshake timed out')
            message = next(m for m in self.messages if predicate(m))
            if message.get('success') is False:
                raise RuntimeError('DAP request failed: '+message.get('message', ''))

    def close(self):
        self.socket.shutdown(socket.SHUT_RDWR)
        self.stream.close()
        self.socket.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--conditions', nargs='+', default=['native-a', 'debugpy', 'native-b', 'recreate'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    photo = json.loads((DATA_DIR/'output_localization_profile_20261001/macar-profile-2594-pipeline-native.jsonl').read_text().splitlines()[0])['photo']
    args.output_directory.mkdir(parents=True, exist_ok=True)
    debug_path = Path.home()/'.vscode/extensions/ms-python.debugpy-2026.6.0-darwin-arm64/bundled/libs'
    for name in args.conditions:
        output = args.output_directory/(name+'.jsonl')
        if output.exists():
            raise RuntimeError('Output already exists: '+str(output))
        env = dict(os.environ, PYTHONPYCACHEPREFIX='/private/tmp/macar-profile-pycache',
                   TORCH_HOME=str(DATA_DIR/'.model_cache/torch'), PYTHONUNBUFFERED='1')
        command = [sys.executable]
        if name == 'debugpy':
            env['PYTHONPATH'] = str(debug_path)
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            command += ['-m', 'debugpy', '--listen', '127.0.0.1:'+str(port), '--wait-for-client']
        command += [str(root/'profile_localization.py'), photo, '--device', 'mps',
                    '--mode', 'pipeline', '--runs', str(args.runs), '--sizes', '1600', '--output', str(output)]
        if name == 'recreate':
            command.append('--recreate-models')
        print(json.dumps(dict(event='begin', condition=name)), flush=True)
        started = time.monotonic()
        with (args.output_directory/(name+'.stdout')).open('x') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            client = None
            try:
                if name == 'debugpy':
                    client = DebugClient(port, process)
                status = process.wait(timeout=600)
                if status:
                    raise RuntimeError(name+' failed; see '+str(log.name))
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                if client:
                    client.close()
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        print(json.dumps(dict(event='complete', condition=name,
            process_elapsed_ms=(time.monotonic()-started)*1000,
            trace_active=rows[0]['trace_active'],
            pipeline_ms=[round(r['total_ms'], 1) for r in rows if r['event'] == 'pipeline'],
            extraction_ms=[round(r['total_ms'], 1) for r in rows if r['event'] == 'extraction']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
