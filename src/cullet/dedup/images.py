"""
Find duplicate images regardless of their size with the SSCD copy detection model.

In each group of duplicates one image is proposed to keep: the one with at least 5% more pixels, else the one
with at least 20% bigger file size (tolerance_pixels, tolerance_size), else the first in name
order. Embeddings are cached per input dir, so rerunning after adding images only embeds the
new ones.
"""

import logging
import os
from pathlib import Path
from typing import Any

import torch
from natsort import natsorted
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from cullet.dedup.common import (
    EmbeddingCache,
    find_similar_pairs,
    group_pairs,
    pick_best_member,
)
from cullet.dedup.image_groups import (
    DedupImageConfig,
    DedupImageResult,
    DuplicateGroup,
    DuplicateMember,
)
from cullet.dedup.sscd import SSCD_INPUT_SIZE, SscdEmbedder, get_default_device, get_sscd_transform
from cullet.paths import is_inside_dir

logger = logging.getLogger(__name__)

ENDINGS = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".gif", ".jfif"]
# pillow refuses images above 2x this limit as decompression bombs, panoramas are bigger than that
Image.MAX_IMAGE_PIXELS = 10 * Image.MAX_IMAGE_PIXELS


class ImageDataset(torch.utils.data.Dataset):
    """Load images for SSCD in worker processes. Unreadable images yield an error string."""

    def __init__(self, image_files: list[Path]):
        self.image_files = image_files
        self.transform = get_sscd_transform()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor | None, int, int, str | None]:
        image_file = self.image_files[idx]
        try:
            with Image.open(image_file) as opened_image:
                width, height = opened_image.size
                # jpeg decoding is much faster when the decoder is told the target size upfront
                opened_image.draft("RGB", (SSCD_INPUT_SIZE * 2, SSCD_INPUT_SIZE * 2))
                image = ImageOps.exif_transpose(opened_image).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            return idx, None, 0, 0, f"{type(e).__name__}: {e}"
        except Exception as e:
            raise ValueError(f"Unhandled exception for file {image_file}: {e}") from e
        return idx, self.transform(image), width, height, None


def _collate_as_lists(batch):
    return list(zip(*batch))


def find_duplicate_images(
    input_dir: Path,
    config: DedupImageConfig,
    exclude_dir: Path | None = None,
) -> DedupImageResult:
    """
    Index the folder, embed the images that are not cached yet, and group the similar ones.

    Args:
        input_dir: folder to search
        config: detection settings
        exclude_dir: files inside this dir are skipped, e.g. the quarantine dir of an earlier run
    """
    input_dir = Path(input_dir)
    device = get_default_device() if config.device is None else config.device
    src_index = index_files(input_dir, config.recursive)
    logger.info(f"Found {len(src_index)} files.")
    rel_files = natsorted(
        f
        for f in src_index.keys()
        if any(f.lower().endswith(a) for a in ENDINGS)
        and not is_inside_dir(f, input_dir, exclude_dir)
    )
    logger.info(f"From those detected {len(rel_files)} images.")
    if len(rel_files) == 0:
        raise ValueError(
            f"No images found in {input_dir} ({len(src_index)} files indexed, "
            f"image endings: {ENDINGS}, subdirectories searched: {config.recursive})"
        )

    # embed images that are not in the cache yet
    cache = EmbeddingCache("images", input_dir, reset=config.reset_cache)
    entries = {}
    todo = []
    for rel_file in rel_files:
        props = src_index[rel_file]
        entry = cache.get(rel_file, props.st_size, props.st_mtime)
        if entry is None:
            todo.append(rel_file)
        else:
            entries[rel_file] = entry
    logger.info(f"{len(entries)} images cached, {len(todo)} to embed.")
    broken_files = {}
    if len(todo) > 0:
        embedder = SscdEmbedder(config.model_file, device, config.batch_size)
        dataset = ImageDataset([input_dir / f for f in todo])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            collate_fn=_collate_as_lists,
        )
        pbar = tqdm(total=len(todo), desc="Embedding")
        for indices, tensors, widths, heights, errors in dataloader:
            pbar.update(len(indices))
            valid = [i for i, tensor in enumerate(tensors) if tensor is not None]
            for i in range(len(indices)):
                if tensors[i] is None:
                    rel_file = todo[indices[i]]
                    logger.error(f"Skipping {rel_file}: {errors[i]}")
                    broken_files[rel_file] = errors[i]
            if len(valid) == 0:
                continue
            embeddings = embedder.embed_tensors(torch.stack([tensors[i] for i in valid]))
            for i, embedding in zip(valid, embeddings):
                rel_file = todo[indices[i]]
                props = src_index[rel_file]
                entries[rel_file] = cache.put(
                    rel_file,
                    props.st_size,
                    props.st_mtime,
                    emb=embedding,
                    width=widths[i],
                    height=heights[i],
                )
        pbar.close()
    cache.save(keep_rel_files=rel_files)
    rel_files = [f for f in rel_files if f in entries]
    if len(rel_files) == 0:
        raise ValueError(f"None of the images in {input_dir} could be read: {broken_files}")

    embeddings = torch.stack([entries[f]["emb"] for f in rel_files])
    pairs = find_similar_pairs(embeddings, config.threshold, device)
    groups = make_groups(input_dir, rel_files, entries, pairs, config)
    return DedupImageResult(groups, len(rel_files), len(pairs), broken_files)


def index_files(input_dir: Path, recursive: bool) -> dict[str, os.stat_result]:
    """All files of the folder with their stat, keyed by the path relative to the folder."""
    index = {}
    for root, dirs, names in os.walk(input_dir):
        for name in names:
            file = Path(root) / name
            index[file.relative_to(input_dir).as_posix()] = file.stat()
        if not recursive:
            dirs.clear()
    return index


def make_groups(
    input_dir: Path,
    rel_files: list[str],
    entries: dict[str, dict[str, Any]],
    pairs: list[tuple[int, int, float]],
    config: DedupImageConfig,
) -> list[DuplicateGroup]:
    """
    Connect the similar pairs into groups and pick the member to keep in each.

    Args:
        input_dir: the files are relative to this dir
        rel_files: all images in name order, the pair indices refer to this list
        entries: per rel_file the cache entry with emb, width, height, size
        pairs: (i, j, similarity) from find_similar_pairs
        config: the tolerances decide which member is kept
    """

    def criteria(i: int) -> list[tuple[float, float]]:
        entry = entries[rel_files[i]]
        return [
            (entry["width"] * entry["height"], config.tolerance_pixels),
            (entry["size"], config.tolerance_size),
        ]

    groups = []
    for members in group_pairs(len(rel_files), pairs):
        keep = pick_best_member(members, criteria)
        group_embs = torch.stack([entries[rel_files[i]]["emb"] for i in members])
        group_members = []
        for pos, i in enumerate(members):
            entry = entries[rel_files[i]]
            sims = group_embs @ entry["emb"]
            sims[pos] = -1.0
            group_members.append(
                DuplicateMember(
                    path=(input_dir / rel_files[i]).absolute(),
                    rel_file=rel_files[i],
                    width=entry["width"],
                    height=entry["height"],
                    size=entry["size"],
                    sim=float(sims.max()),
                )
            )
        keep_pos = members.index(keep)
        rest = [m for pos, m in enumerate(group_members) if pos != keep_pos]
        groups.append(DuplicateGroup([group_members[keep_pos]] + rest))
    return groups
