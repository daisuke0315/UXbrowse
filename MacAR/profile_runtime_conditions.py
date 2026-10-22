"""Controlled re-localizations through the existing local photo-session API.

Display is controlled separately through the browser UI. No credentials or
photo pixels are written. VM counters are host-wide, not process attribution.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request
from profile_host_memory import host_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--recreate', action='store_true')
    parser.add_argument('--idle-seconds', type=float, default=0.)
    args = parser.parse_args()
    if args.output.exists() or args.runs < 1:
        parser.error('Use a fresh output and positive runs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def get(route):
        return json.load(urllib.request.urlopen(args.base+route, timeout=10))
    config, state = get('/api/config'), get('/api/state')
    if state['status'] != 'running' or config['defaults']['input_source'] != 'photo':
        raise RuntimeError('Expected running photo session')
    def post(route, data):
        req = urllib.request.Request(args.base+route, data=json.dumps(data).encode(),
            headers={'Content-Type': 'application/json', 'X-ViewSync-Token': config['token']}, method='POST')
        urllib.request.urlopen(req, timeout=10).close()
    def packet(after):
        data = urllib.request.urlopen(args.base+'/api/frame?after='+str(after), timeout=10).read()
        if not data:
            return None
        n = int.from_bytes(data[:4], 'little')
        return json.loads(data[4:4+n])
    def emit(row):
        with args.output.open('a', encoding='utf-8') as target:
            target.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    meta = packet(-1)
    emit(dict(event='configuration', condition=args.condition, pid=args.pid,
        settings=config['defaults'], photo=meta['photo_name'], initial_result=meta['result'],
        recreate=args.recreate, native_sampling=False))
    for run in range(args.runs):
        if args.idle_seconds:
            emit(dict(event='before_idle', run=run, memory=host_memory(args.pid)))
            time.sleep(args.idle_seconds)
        before = host_memory(args.pid)
        previous_source, previous_generation = meta['source_id'], meta['generation']
        after, started = meta['sequence'], time.monotonic()
        if args.recreate:
            post('/api/stop', {})
            deadline = time.monotonic()+120
            while get('/api/state')['status'] != 'idle':
                if time.monotonic() > deadline:
                    raise RuntimeError('Stop timed out')
                time.sleep(.1)
            post('/api/start', config['defaults'])
        else:
            post('/api/control', dict(action='relocalize'))
        timeline, samples, last_sample = [], [], 0.
        while time.monotonic()-started < 180:
            row = packet(after)
            if row is None:
                continue
            meta, after = row, row['sequence']
            search = row.get('search', {})
            stage = search.get('stage')
            if not timeline or timeline[-1]['stage'] != stage:
                timeline.append(dict(stage=stage, elapsed_ms=(time.monotonic()-started)*1000))
            if time.monotonic()-last_sample >= 1:
                samples.append(dict(elapsed_ms=(time.monotonic()-started)*1000, **host_memory(args.pid)))
                last_sample = time.monotonic()
            new_source = (row['source_id'] != previous_source or row['generation'] != previous_generation)
            if new_source and row.get('result') is not None and not search.get('active'):
                emit(dict(event='complete', condition=args.condition, run=run,
                    elapsed_ms=(time.monotonic()-started)*1000,
                    localize_ms=row.get('localize_ms'), mpr_ms=row.get('mpr_ms'),
                    result=row['result'], search=search, log=row.get('diagnostic_log'),
                    generation=row['generation'], source_id=row['source_id'],
                    memory_before=before, memory_after=host_memory(args.pid),
                    memory_samples=samples, timeline=timeline))
                break
        else:
            raise RuntimeError('Localization timed out')


if __name__ == '__main__':
    main()
