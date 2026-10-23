"""Launcher, headless reconstruction, and isolated worker entry point."""
import argparse
import json
import multiprocessing
import socket
import threading
import webbrowser

from .pipeline import execute, start_run, STAGES
from .project import Project


def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='ViewSync Builder — local CT-GCP reconstruction')
    sub = parser.add_subparsers(dest='command')
    serve = sub.add_parser('serve')
    serve.add_argument('--port', type=int, default=5180)
    serve.add_argument('--no-browser', action='store_true')
    worker = sub.add_parser('worker')
    worker.add_argument('project')
    worker.add_argument('run_id')
    run = sub.add_parser('run')
    run.add_argument('project')
    run.add_argument('--settings', default='{}', help='JSON object')
    run.add_argument('--stop-after', default='export', choices=STAGES)
    sub.add_parser('doctor')
    args = parser.parse_args()
    if args.command == 'worker':
        execute(args.project, args.run_id)
    elif args.command == 'run':
        project = Project.open(args.project)
        state = start_run(project, json.loads(args.settings), args.stop_after)
        result = execute(project.root, state['id'])
        print(json.dumps({k:result[k] for k in ('id', 'status', 'stage', 'elapsed_seconds')}, indent=2))
    elif args.command == 'doctor':
        from .engine import capabilities
        print(json.dumps(capabilities(), indent=2))
    else:
        import uvicorn
        from .server import create_app
        port = getattr(args, 'port', 5180)
        # Reserve the port until uvicorn takes over, preventing a failed startup from opening another service.
        listener = socket.socket()
        listener.bind(('127.0.0.1', port))
        listener.listen(128)
        if not getattr(args, 'no_browser', False):
            threading.Timer(1., lambda: webbrowser.open(f'http://127.0.0.1:{listener.getsockname()[1]}')).start()
        config = uvicorn.Config(create_app(), log_level='warning', access_log=False)
        server = uvicorn.Server(config)
        import asyncio
        asyncio.run(server.serve(sockets=[listener]))


if __name__ == '__main__':
    main()
