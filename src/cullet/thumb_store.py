"""
Thumbnail cache on disk: one SQLite file per image folder, holding JPEG-encoded thumbnails
keyed by the file's relative path. An entry is only reused if the mtime is unchanged and the
thumbnail size matches.
"""

import logging
import sqlite3
import threading
from pathlib import Path

from cullet.paths import folder_cache_name, get_cache_dir

logger = logging.getLogger(__name__)


def thumb_store_file(folder: Path) -> Path:
    return get_cache_dir() / "thumbs" / f"{folder_cache_name(folder)}.sqlite"


def reset_thumb_stores(folder: Path) -> list[Path]:
    """Delete the thumbnail stores of the folder and of its subfolders, so every thumbnail is
    decoded again. Needed when images were edited without their mtime changing, which is what
    the stores key on."""
    prefix = folder_cache_name(folder)
    deleted = []
    # the -wal and -shm sidecars of an open store go with it, hence the glob past .sqlite
    for path in sorted(thumb_store_file(folder).parent.glob("*.sqlite*")):
        if not (path.name.startswith(f"{prefix}.sqlite") or path.name.startswith(f"{prefix}%2F")):
            continue
        path.unlink()
        deleted.append(path)
    logger.info(f"Deleted {len(deleted)} thumbnail store files of {folder} and its subfolders")
    return deleted


class ThumbnailStore:
    def __init__(self, folder: Path, thumb_size: int):
        self.folder = Path(folder)
        self.thumb_size = thumb_size
        self.db_file = thumb_store_file(folder)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        # decode jobs write from pool threads, so one connection guarded by a lock
        self.conn = sqlite3.connect(self.db_file.as_posix(), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS thumbs "
                "(name TEXT PRIMARY KEY, mtime REAL NOT NULL, size INTEGER NOT NULL, "
                "data BLOB NOT NULL)"
            )
            self.conn.commit()

    def key(self, path: Path) -> str:
        return Path(path).absolute().relative_to(self.folder.absolute()).as_posix()

    def get(self, path: Path, mtime: float) -> bytes | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT data FROM thumbs WHERE name = ? AND mtime = ? AND size = ?",
                (self.key(path), mtime, self.thumb_size),
            ).fetchone()
        return None if row is None else row[0]

    def put(self, path: Path, mtime: float, data: bytes) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO thumbs (name, mtime, size, data) VALUES (?, ?, ?, ?)",
                (self.key(path), mtime, self.thumb_size, data),
            )
            self.conn.commit()

    def count(self) -> int:
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM thumbs").fetchone()[0]

    def prune(self, keep_paths: list[Path]) -> int:
        """Drop entries of files that are no longer in the folder. Returns how many."""
        keep = {self.key(p) for p in keep_paths}
        with self.lock:
            names = [row[0] for row in self.conn.execute("SELECT name FROM thumbs")]
            stale = [(name,) for name in names if name not in keep]
            if stale:
                self.conn.executemany("DELETE FROM thumbs WHERE name = ?", stale)
                self.conn.commit()
        if stale:
            logger.info(f"Pruned {len(stale)} stale thumbnails from {self.db_file}")
        return len(stale)
