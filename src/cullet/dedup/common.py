"""
The pieces of the deduplication that do not know about images: embedding cache, similarity
search and grouping.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from cullet.paths import folder_cache_name, get_temp_cache_dir

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Cache per-file results of one input directory, keyed by relative path. An entry is only
    reused if size and mtime of the file are unchanged.

    The cache lives in the temp dir, so it is gone after a reboot and rebuilding it costs
    seconds to minutes per input folder.
    """

    def __init__(self, kind: str, base_dir: Path, reset: bool = False):
        self.cache_file = (
            get_temp_cache_dir() / "dedup" / kind / f"{folder_cache_name(base_dir)}.pth"
        )
        self.entries: dict[str, dict[str, Any]] = {}
        if self.cache_file.is_file() and not reset:
            self.entries = torch.load(self.cache_file, map_location="cpu", weights_only=True)
            logger.info(f"Loaded {len(self.entries)} cached entries from {self.cache_file}")
        self.n_new = 0

    def get(self, rel_file: str, size: int, mtime: float) -> dict[str, Any] | None:
        entry = self.entries.get(rel_file)
        if entry is None or entry["size"] != size or entry["mtime"] != mtime:
            return None
        return _convert_float_tensors(entry, torch.float32)

    def put(self, rel_file: str, size: int, mtime: float, **data) -> dict[str, Any]:
        entry = {"size": size, "mtime": mtime, **data}
        # embeddings are stored in half precision, the cache is half the size and the
        # similarities change in the 4th digit
        self.entries[rel_file] = _convert_float_tensors(entry, torch.float16)
        self.n_new += 1
        return entry

    def save(self, keep_rel_files: list[str]):
        """Write the cache, dropping entries of files that no longer exist in the index."""
        n_before = len(self.entries)
        self.entries = {k: v for k, v in self.entries.items() if k in set(keep_rel_files)}
        if self.n_new == 0 and len(self.entries) == n_before:
            return
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        # written next to the target and renamed, so an interrupted save leaves no broken cache
        part_file = self.cache_file.with_name(f"{self.cache_file.name}.part")
        torch.save(self.entries, part_file)
        os.replace(part_file, self.cache_file)
        logger.info(f"Saved {len(self.entries)} entries ({self.n_new} new) to {self.cache_file}")
        self.n_new = 0


def _convert_float_tensors(entry: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    return {
        k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in entry.items()
    }


def find_similar_pairs(
    embeddings: torch.Tensor, threshold: float, device: str | torch.device, chunk_size: int = 1024
) -> list[tuple[int, int, float]]:
    """
    Find all pairs (i, j) with i < j whose cosine similarity is at least the threshold.

    Args:
        embeddings: (N, D) L2-normalized embeddings
        threshold: minimum cosine similarity
        device: where to compute the similarity matrix, done in row chunks to bound memory
        chunk_size: rows per chunk

    Returns:
        list of (i, j, similarity)
    """
    embeddings = embeddings.to(device)
    n_items = len(embeddings)
    all_cols = torch.arange(n_items, device=embeddings.device).unsqueeze(0)
    pairs = []
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        sim = embeddings[start:end] @ embeddings.T
        rows = torch.arange(start, end, device=embeddings.device).unsqueeze(1)
        # only keep the upper triangle, so each pair is found once and items never match themselves
        sim = sim.masked_fill(all_cols <= rows, -1.0)
        hits = torch.nonzero(sim >= threshold)
        sims = sim[hits[:, 0], hits[:, 1]].tolist()
        for (row, col), value in zip(hits.tolist(), sims):
            pairs.append((row + start, col, value))
    return pairs


def group_pairs(n_items: int, pairs: list[tuple[int, int, float]]) -> list[list[int]]:
    """Connect the pairs into groups (union-find), return only groups with at least 2 members."""
    parent = list(range(n_items))

    def find_root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _ in pairs:
        root_i, root_j = find_root(i), find_root(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)
    groups = defaultdict(list)
    for i in range(n_items):
        groups[find_root(i)].append(i)
    return [sorted(members) for members in groups.values() if len(members) > 1]


def pick_best_member(members: list[int], criteria_fn) -> int:
    """
    Pick the member of a duplicate group to keep.

    Args:
        members: item indices sorted by name, the first one wins if nothing else decides
        criteria_fn: maps an index to a list of (value, relative_tolerance) in priority order,
            e.g. [(n_pixels, 0.05), (file_size, 0.05)]. A criterion only decides if the values
            differ by more than the tolerance, so a file that is 1 byte bigger does not win.

    Returns:
        index of the member to keep
    """
    best = members[0]
    for candidate in members[1:]:
        if _is_significantly_better(criteria_fn(candidate), criteria_fn(best)):
            best = candidate
    return best


def _is_significantly_better(
    criteria_a: list[tuple[float, float]], criteria_b: list[tuple[float, float]]
) -> bool:
    assert len(criteria_a) == len(criteria_b), f"{criteria_a} vs {criteria_b}"
    for (value_a, tolerance), (value_b, _) in zip(criteria_a, criteria_b):
        if value_a > value_b * (1 + tolerance):
            return True
        if value_b > value_a * (1 + tolerance):
            return False
    return False
