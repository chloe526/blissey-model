"""Cache-aware source verification with a hard, killable whole-operation deadline."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request


def log(message):
    print('[source] ' + message, flush=True)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_files(source, cache, roots, offline, socket_timeout, file_timeout):
    """Child worker. Parent deadline also bounds DNS, TLS and executor shutdown."""
    started = time.monotonic(); lock = threading.Lock(); active = {}
    stats = dict(local_files=0, downloaded_files=0, local_bytes=0, downloaded_bytes=0)
    roots = [Path(p) for p in roots]
    if cache is not None:
        roots.insert(0, Path(cache))
    count = len(source['files'])
    log(f'Verifying {count} pinned files; offline={offline}; socket timeout={socket_timeout:g}s; file deadline={file_timeout:g}s')

    def one(f):
        name = f['path']; relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('session_03_LAHACKS', 'session_04_LAHACKS'):
            raise ValueError('Forbidden source path: ' + name)
        for root in roots:
            candidate = root / relative
            if candidate.is_file():
                if candidate.stat().st_size != f['bytes'] or file_sha(candidate) != f['sha256']:
                    raise ValueError('Local SHA-256 mismatch (not overwritten): ' + str(candidate))
                return 'local', f['bytes']
        if offline:
            raise FileNotFoundError('Offline source file missing: ' + name)
        target = None if cache is None else Path(cache) / relative
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.NamedTemporaryFile(dir=target.parent, prefix='.download-', suffix='.part', delete=False)
        else:
            temporary = None
        url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/recordings/{name}"
        tick = time.monotonic(); h = hashlib.sha256(); size = 0
        with lock:
            active[name] = tick
        log('Downloading ' + name)
        try:
            with urllib.request.urlopen(url, timeout=socket_timeout) as response:
                while True:
                    if time.monotonic() - tick >= file_timeout:
                        raise TimeoutError('File deadline exceeded: ' + name)
                    chunk = response.read1(64 * 1024)
                    if not chunk:
                        break
                    h.update(chunk); size += len(chunk)
                    if size > f['bytes']:
                        raise ValueError('Remote file exceeds pinned size: ' + name)
                    if temporary is not None:
                        temporary.write(chunk)
            if time.monotonic() - tick >= file_timeout:
                raise TimeoutError('File deadline exceeded: ' + name)
            if size != f['bytes'] or h.hexdigest() != f['sha256']:
                raise ValueError('Remote SHA-256 mismatch: ' + name)
            if temporary is not None:
                temporary.close(); Path(temporary.name).replace(target)
            return 'downloaded', size
        except Exception as exc:
            raise RuntimeError(f'{name}: {exc}') from exc
        finally:
            with lock:
                active.pop(name, None)
            if temporary is not None:
                temporary.close(); Path(temporary.name).unlink(missing_ok=True)

    # The parent process is authoritative: even DNS or a stuck thread cannot
    # keep the submitter waiting past its whole-operation timeout.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(one, f) for f in source['files']}
        completed = 0
        while pending:
            done, pending = wait(pending, timeout=5., return_when=FIRST_COMPLETED)
            if not done:
                with lock:
                    detail = ', '.join(f'{name} ({time.monotonic()-tick:.0f}s)' for name, tick in active.items())
                log(f'{completed}/{count} verified; waiting: {detail or "local hashing"}')
            for future in done:
                kind, size = future.result()
                stats[kind + '_files'] += 1; stats[kind + '_bytes'] += size; completed += 1
                if completed % 25 == 0 or completed == count:
                    log(f'{completed}/{count} verified; local={stats["local_files"]}, downloaded={stats["downloaded_files"]}')
    stats.update(seconds=round(time.monotonic() - started, 3), integrity_verified=True)
    log(f'Complete: reused {stats["local_files"]}; downloaded {stats["downloaded_files"]} ({stats["downloaded_bytes"]:,} bytes)')
    return stats


def transfer(source, cache=None, reuse_roots=(), offline=False, timeout=120., socket_timeout=10., file_timeout=45.):
    if min(timeout, socket_timeout, file_timeout) <= 0:
        raise ValueError('Timeouts must be positive')
    # A subprocess, rather than Future timeout alone, guarantees Python's
    # non-daemon HTTP threads are terminated on the overall deadline.
    with tempfile.TemporaryDirectory(prefix='blissey-source-check-') as work:
        manifest = Path(work) / 'source.json'; result = Path(work) / 'result.json'
        manifest.write_text(json.dumps(source))
        command = [sys.executable, '-B', '-m', 'lahacks_cv.transport', '--manifest', str(manifest),
                   '--result', str(result), '--socket-timeout', str(socket_timeout), '--file-timeout', str(file_timeout)]
        if cache is not None:
            command += ['--cache', str(Path(cache).resolve())]
        for root in reuse_roots:
            command += ['--reuse-root', str(Path(root).resolve())]
        if offline:
            command += ['--offline']
        log(f'Hard whole-verification deadline: {timeout:g}s')
        try:
            subprocess.run(command, check=True, timeout=timeout, cwd=Path(__file__).resolve().parents[1])
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f'Source verification exceeded {timeout:g}s; download worker stopped; no Veri submission attempted') from exc
        return json.loads(result.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--result', type=Path, required=True)
    p.add_argument('--cache', type=Path); p.add_argument('--reuse-root', action='append', default=[])
    p.add_argument('--offline', action='store_true'); p.add_argument('--socket-timeout', type=float, default=10.)
    p.add_argument('--file-timeout', type=float, default=45.)
    a = p.parse_args()
    stats = verify_files(json.loads(a.manifest.read_text()), a.cache, a.reuse_root, a.offline,
                         a.socket_timeout, a.file_timeout)
    a.result.write_text(json.dumps(stats))


if __name__ == '__main__':
    main()
