"""Single-process prototype of a bounded RAM/disk cache for immutable byte ranges.

This is a benchmark component, not an inference backend. Byte budgets cover cached
payloads only, not Python, temporary fetch buffers, OS page cache, or other programs.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib
import os
import re
import time
import urllib.request


class RangeError(RuntimeError):
    pass


class RangeSource:
    """Read ranges from a local file or an HTTP(S) URL; reject non-range responses."""

    def __init__(self, location, timeout=30):
        self.location = str(location)
        self.remote = self.location.startswith(('http://', 'https://'))
        self.timeout = timeout
        # Include source identity in filenames, never the URL itself.
        self.identity = hashlib.sha256(self.location.encode('utf-8')).hexdigest()
        if not self.remote:
            p = Path(self.location).resolve()
            st = p.stat()
            # Cache keys also change when the local file changes.
            identity = f'{p}:{st.st_size}:{st.st_mtime_ns}'
            self.identity = hashlib.sha256(identity.encode('utf-8')).hexdigest()
            self.path = p

    def read(self, offset, length):
        if offset < 0 or length <= 0:
            raise ValueError('offset must be nonnegative and length must be positive')
        if not self.remote:
            with self.path.open('rb') as f:
                f.seek(offset)
                data = f.read(length)
        else:
            end = offset + length - 1
            request = urllib.request.Request(self.location, headers={
                'Range': f'bytes={offset}-{end}', 'Accept-Encoding': 'identity',
            })
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status != 206:
                    raise RangeError('server did not honor the byte range (expected HTTP 206)')
                content_range = response.headers.get('Content-Range', '')
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', content_range)
                if not match or int(match[1]) != offset or int(match[2]) != end:
                    raise RangeError('server returned an incorrect Content-Range')
                if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                    raise RangeError('encoded range response is unsupported')
                data = response.read(length + 1)
        if len(data) != length:
            raise RangeError(f'expected {length} bytes, received {len(data)}')
        return data


class ExpertCache:
    """LRU cache; dedicated directory, one process, no concurrent writers."""

    def __init__(self, source, directory, memory_bytes, disk_bytes, max_range_bytes):
        if min(memory_bytes, disk_bytes) < 0 or max_range_bytes <= 0:
            raise ValueError('budgets must be nonnegative and max range must be positive')
        self.source, self.directory = source, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.memory_bytes, self.disk_bytes = memory_bytes, disk_bytes
        self.max_range_bytes = max_range_bytes
        self.memory = OrderedDict()
        self.memory_used = 0
        self.stats = dict(requests=0, memory_hits=0, disk_hits=0, source_fetches=0,
                          source_bytes=0, disk_evictions=0, memory_evictions=0)
        self._clock_ns = max([time.time_ns()] + [entry[0] for entry in self._entries()])
        self._trim_disk(0)

    def _touch(self, path):
        # Break ties even on filesystems with coarse timestamp resolution.
        self._clock_ns = max(time.time_ns(), self._clock_ns + 1_000_000)
        os.utime(path, ns=(self._clock_ns, self._clock_ns))

    def _entries(self):
        # Restrict eviction to files created by this prototype, not arbitrary files.
        entries = []
        for p in self.directory.iterdir():
            if re.fullmatch(r'rbtcache-[0-9a-f]{64}\.bin', p.name) and p.is_file() and not p.is_symlink():
                st = p.stat()
                entries.append((st.st_mtime_ns, p.name, st.st_size, p))
        return sorted(entries)

    def _trim_disk(self, incoming):
        entries = self._entries()
        total = sum(e[2] for e in entries)
        for _, _, size, path in entries:
            if total + incoming <= self.disk_bytes:
                break
            path.unlink()
            total -= size
            self.stats['disk_evictions'] += 1

    def get(self, offset, length):
        if offset < 0 or length <= 0 or length > self.max_range_bytes:
            raise ValueError('invalid range or range exceeds max_range_bytes')
        self.stats['requests'] += 1
        key = hashlib.sha256(f'{self.source.identity}:{offset}:{length}'.encode()).hexdigest()
        path = self.directory / f'rbtcache-{key}.bin'
        if key in self.memory:
            self.stats['memory_hits'] += 1
            self.memory.move_to_end(key)
            if path.is_file() and not path.is_symlink():
                self._touch(path)
            return self.memory[key]
        if path.is_file() and not path.is_symlink() and path.stat().st_size == length:
            data = path.read_bytes()
            self._touch(path)
            self.stats['disk_hits'] += 1
        else:
            # A wrong-sized cached entry is not usable.
            if path.is_file() and not path.is_symlink():
                path.unlink()
            data = self.source.read(offset, length)
            self.stats['source_fetches'] += 1
            self.stats['source_bytes'] += len(data)
            if length <= self.disk_bytes:
                self._trim_disk(length)
                # This cache is explicitly single-process; clean up partial writes.
                try:
                    with path.open('xb') as f:
                        f.write(data)
                    self._touch(path)
                except Exception:
                    if path.exists() and not path.is_symlink():
                        path.unlink()
                    raise
        if length <= self.memory_bytes:
            while self.memory_used + length > self.memory_bytes:
                _, old = self.memory.popitem(last=False)
                self.memory_used -= len(old)
                self.stats['memory_evictions'] += 1
            self.memory[key] = data
            self.memory_used += length
        return data

    def snapshot(self):
        return dict(self.stats, memory_payload_bytes=self.memory_used,
                    disk_payload_bytes=sum(e[2] for e in self._entries()))
