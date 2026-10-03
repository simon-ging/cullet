"""
Decode images on a thread pool and keep the most recent results in memory.

A cache entry is keyed by path and mtime, so a file that was rewritten (e.g. rotated) is
decoded again on the next request.
"""

import io
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from PIL import Image
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage

from cullet.image_io import load_image, load_thumbnail, pil_to_qimage
from cullet.thumb_store import ThumbnailStore

logger = logging.getLogger(__name__)
DecodeFn = Callable[[Path], QImage]
THUMB_JPEG_QUALITY = 88
PRIORITY_VISIBLE = 0
PRIORITY_WARMUP = -1
# decoded panoramas are hundreds of megabytes each, a count alone does not bound the memory
MAX_CACHE_BYTES = 1024**3


def decode_full(path: Path) -> QImage:
    return pil_to_qimage(load_image(path))


def make_thumbnail_decoder(size: int, store: ThumbnailStore | None = None) -> DecodeFn:
    """Thumbnail decoder that reads from and fills the on-disk store if one is given."""

    def decode(path: Path) -> QImage:
        if store is None:
            return pil_to_qimage(load_thumbnail(path, size))
        mtime = path.stat().st_mtime
        data = store.get(path, mtime)
        if data is not None:
            img = Image.open(io.BytesIO(data))
            img.load()
            return pil_to_qimage(img)
        img = load_thumbnail(path, size).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=THUMB_JPEG_QUALITY)
        store.put(path, mtime, buffer.getvalue())
        return pil_to_qimage(img)

    return decode


class _JobSignals(QObject):
    done = Signal(str, float, object)
    failed = Signal(str, str)


class _DecodeJob(QRunnable):
    def __init__(
        self, path: Path, mtime: float, decode_fn: DecodeFn, priority: int, signals: _JobSignals
    ):
        super().__init__()
        self.path = path
        self.mtime = mtime
        self.decode_fn = decode_fn
        self.priority = priority
        self.signals = signals
        self.setAutoDelete(False)

    def run(self):
        try:
            qimage = self.decode_fn(self.path)
        except Exception as e:
            # an exception inside the thread pool would only be printed and the request would
            # silently never finish, so forward it to the main thread instead
            self.signals.failed.emit(self.path.as_posix(), f"{type(e).__name__}: {e}")
            return
        self.signals.done.emit(self.path.as_posix(), self.mtime, qimage)


class ImageLoader(QObject):
    image_ready = Signal(str)  # posix path of the decoded file
    image_failed = Signal(str, str)  # posix path, error message

    def __init__(
        self,
        decode_fn: DecodeFn = decode_full,
        max_items: int = 6,
        max_bytes: int | None = MAX_CACHE_BYTES,
        threads: int = 2,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.decode_fn = decode_fn
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(threads)
        self._cache: OrderedDict[str, tuple[float, QImage]] = OrderedDict()
        self._pending: dict[str, _DecodeJob] = {}
        self._signals = _JobSignals(self)
        self._signals.done.connect(self._on_done)
        self._signals.failed.connect(self._on_failed)

    def set_decode_fn(self, decode_fn: DecodeFn) -> None:
        """Jobs already queued keep the function they were created with."""
        self.decode_fn = decode_fn

    def get(self, path: Path) -> QImage | None:
        """Return the decoded image if it is cached and current, otherwise start decoding."""
        key = path.as_posix()
        mtime = path.stat().st_mtime
        entry = self._cache.get(key)
        if entry is not None and entry[0] == mtime:
            self._cache.move_to_end(key)
            return entry[1]
        self.request(path)
        return None

    def request(self, path: Path, priority: int = PRIORITY_VISIBLE) -> None:
        key = path.as_posix()
        mtime = path.stat().st_mtime
        entry = self._cache.get(key)
        if entry is not None and entry[0] == mtime:
            return
        pending = self._pending.get(key)
        if pending is not None:
            # a visible request must not wait behind the warm-up queue: requeue if still queued
            if priority > pending.priority and self.pool.tryTake(pending):
                pending.priority = priority
                self.pool.start(pending, priority)
            return
        job = _DecodeJob(path, mtime, self.decode_fn, priority, self._signals)
        self._pending[key] = job
        self.pool.start(job, priority)

    def cancel_queued(self) -> None:
        """Drop jobs that have not started yet, e.g. the warm-up of a folder that was left."""
        for key, job in list(self._pending.items()):
            if self.pool.tryTake(job):
                del self._pending[key]

    def invalidate(self, path: Path) -> None:
        self._cache.pop(path.as_posix(), None)

    def _on_done(self, key: str, mtime: float, qimage: QImage) -> None:
        self._pending.pop(key, None)
        self._cache[key] = (mtime, qimage)
        self._cache.move_to_end(key)
        # the least recently used go first. The newest always stays, however big it is.
        while len(self._cache) > 1 and (
            len(self._cache) > self.max_items or self.cached_bytes() > self.max_bytes_or_inf()
        ):
            self._cache.popitem(last=False)
        self.image_ready.emit(key)

    def cached_bytes(self) -> int:
        return sum(qimage.sizeInBytes() for _mtime, qimage in self._cache.values())

    def max_bytes_or_inf(self) -> float:
        return float("inf") if self.max_bytes is None else self.max_bytes

    def _on_failed(self, key: str, message: str) -> None:
        self._pending.pop(key, None)
        logger.error(f"Failed to decode {key}: {message}")
        self.image_failed.emit(key, message)
