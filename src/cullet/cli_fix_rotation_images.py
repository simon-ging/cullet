"""
Guess the rotation of photos and turn them upright.

A pretrained classifier predicts whether a photo is stored rotated by 90, 180 or 270 degrees,
the model is downloaded on first use. The exif orientation is applied before guessing, so only
photos that still show up wrong are reported. Photos below the confidence threshold are listed
as unsure and left alone. Predictions are cached per input dir.

With -w the rotated photos are written to input_dir/rotation_corrected_output keeping their
relative path, and for each a check_rotation/NAME.jpg with the original on the left and the
rotated photo on the right, to look through before copying the results back. Photos that are
upright are not copied. With -O the photos are rotated in place instead. Either way the
metadata (without the orientation tag) and the modification time are kept. The embedded
thumbnail then still shows the unrotated photo, use -E to drop it.

Examples:
    # dry run: list which photos would be rotated
    cullet-fix-rotation-images /path/to/photos
    # write rotated photos and check images to /path/to/photos/rotation_corrected_output
    cullet-fix-rotation-images /path/to/photos -w
    # rotate the photos in place
    cullet-fix-rotation-images /path/to/photos -O -w
    # check how well the model does on photos that are known to be upright, e.g. phone photos
    cullet-fix-rotation-images -e /path/to/upright_photos
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from attrs import define
from natsort import natsorted
from PIL import Image

from tqdm import tqdm
from typedparser import TypedParser, VerboseQuietArgs, add_argument
from cullet.dedup.common import EmbeddingCache
from cullet.dedup.files import PathSpecArgs, index_files, write_json
from cullet.images import (
    ImageMetadata,
    check_exif_survived,
    describe_exif_change,
    extra_frames_are_redundant,
    rotate_by_orientation_tag,
    load_image,
    save_image,
)

from cullet.logs import configure_logging
from cullet.paths import is_inside_dir
from cullet.rotation import (
    N_ROTATIONS,
    OrientationClassifier,
    RotationImageDataset,
    collate_as_lists,
    combine_rotation_views,
    evaluate_rotations,
    get_default_device,
    make_check_image,
    rotate_image,
)

logger = logging.getLogger(__name__)


@define
class Args(PathSpecArgs, VerboseQuietArgs):
    input_dir: Optional[Path] = add_argument(
        positional=True, type=str, nargs="?", help="Directory of photos to check"
    )
    non_recursive: bool = add_argument(
        shortcut="-r", action="store_true", help="Do not search subdirectories for images."
    )
    eval_dir: Optional[Path] = add_argument(
        shortcut="-e",
        type=str,
        help="Directory of upright photos. Rotate each by every step and report the accuracy.",
    )
    min_confidence: float = add_argument(
        shortcut="-c",
        type=float,
        default=0.9,
        help="Photos with a lower confidence are listed as unsure and never rotated.",
    )
    quality: int = add_argument(
        shortcut="-Q", type=int, default=95, help="Jpg quality for rewritten images."
    )
    clean_exif: bool = add_argument(
        shortcut="-E",
        action="store_true",
        help="Rewrite the exif so it describes the written pixels: drop the embedded thumbnail "
        "and the orientation tag and correct the image size tags. Every other tag is kept, "
        "including the vendor specific ones.",
    )
    destroy_metadata: bool = add_argument(
        action="store_true",
        help="Drop the exif and the xmp entirely, keeping the icc color profile. This also "
        "throws away the capture time and the camera model.",
    )
    output_dir_rel: Path = add_argument(
        shortcut="-o",
        type=str,
        help="Output directory relative to input dir.",
        default="rotation_corrected_output",
    )
    overwrite: bool = add_argument(
        shortcut="-O", action="store_true", help="Overwrite files instead of a separate output dir"
    )
    batch_size: int = add_argument(shortcut="-b", type=int, default=32, help="Model batch size")
    workers: int = add_argument(
        shortcut="-j", type=int, default=4, help="Parallel workers for loading images."
    )
    device: str = add_argument(type=str, default=get_default_device(), help="Torch device")
    model_file: Optional[Path] = add_argument(
        shortcut="-m", type=str, help="Model weights, default: downloaded to cache dir."
    )
    reset_cache: bool = add_argument(action="store_true", help="Recompute all predictions.")
    report_file: Optional[Path] = add_argument(
        type=str, help="Write the guessed rotation of every photo to this json file."
    )
    write: bool = add_argument(shortcut="-w", action="store_true", help="Write changes to disk")


ENDINGS = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".jfif"]
# pillow refuses images above 2x this limit as decompression bombs, panoramas are bigger than that
Image.MAX_IMAGE_PIXELS = 10 * Image.MAX_IMAGE_PIXELS
# unfinished outputs of interrupted runs
TMP_STEM_SUFFIX = ".tmp"
CHECK_DIR_NAME = "check_rotation"
CHECK_QUALITY = 85


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logging(args)
    logger.info(f"{args}")
    if args.input_dir is None and args.eval_dir is None:
        raise ValueError("Give a directory of photos to check, a directory to evaluate or both.")
    if args.clean_exif and args.destroy_metadata:
        raise ValueError("Specify either --clean_exif or --destroy_metadata but not both.")
    classifier = OrientationClassifier(args.model_file, args.device, args.batch_size)

    if args.eval_dir is not None:
        rel_files, entries = compute_predictions(Path(args.eval_dir), classifier, args)
        log_probs = torch.stack([entries[f]["logp"] for f in rel_files])
        metrics = evaluate_rotations(log_probs, args.min_confidence)
        logger.info(
            f"Evaluated {len(rel_files)} upright photos in {N_ROTATIONS} rotations each: "
            f"accuracy single view {metrics['single_acc']:.3f}, combined "
            f"{metrics['combined_acc']:.3f}. Confidence >= {args.min_confidence}: "
            f"{metrics['confident_frac']:.3f} of the photos, accuracy "
            f"{metrics['confident_acc']:.4f}"
        )

    if args.input_dir is None:
        return
    input_dir = Path(args.input_dir)
    output_dir = None if args.overwrite else input_dir / args.output_dir_rel
    rel_files, entries = compute_predictions(input_dir, classifier, args, exclude_dir=output_dir)
    log_probs = torch.stack([entries[f]["logp"] for f in rel_files])
    probs = combine_rotation_views(log_probs)

    # a photo rotated by r steps counter-clockwise is fixed by rotating it r steps clockwise
    to_rotate = {}
    unsure = []
    report = []
    for rel_file, prob in zip(rel_files, probs):
        rotation = int(prob.argmax())
        confidence = float(prob[rotation])
        entry = entries[rel_file]
        report.append({"file": rel_file, "rotation_cw": rotation * 90, "confidence": confidence})
        size_str = f"{entry['width']:5d}x{entry['height']:<5d}"
        if confidence < args.min_confidence:
            probs_str = " ".join(f"{p:.2f}" for p in prob.tolist())
            logger.warning(f"UNSURE {size_str} probs=[{probs_str}] {rel_file}")
            unsure.append(rel_file)
            continue
        if rotation == 0:
            logger.debug(f"OK     {size_str} conf={confidence:.3f} {rel_file}")
            continue
        logger.info(f"ROTATE {size_str} {rotation * 90:3d}° cw conf={confidence:.3f} {rel_file}")
        to_rotate[rel_file] = rotation
    if args.report_file is not None:
        write_json(report, Path(args.report_file))
    logger.info(
        f"Photos: {len(rel_files)}, upright: {len(rel_files) - len(to_rotate) - len(unsure)}, "
        f"to rotate: {len(to_rotate)}, unsure: {len(unsure)}"
    )
    if not args.write:
        where = "in place" if output_dir is None else f"to {output_dir}"
        logger.warning(f"Dry run. Use -w to write the rotated photos {where}.")
        return
    n_skipped = 0
    n_tagged = 0
    for rel_file, rotation in natsorted(to_rotate.items()):
        image_file = input_dir / rel_file
        out_file = image_file if output_dir is None else output_dir / rel_file
        # a multi frame jpeg loses everything but the first frame when it is rewritten, which
        # is fine for a smaller copy of the same shot and not fine for auxiliary data
        redundant, report = extra_frames_are_redundant(image_file)
        if not redundant:
            # rewriting the pixels would throw the extra frames away, so only the orientation
            # tag is changed and every other byte of the file is kept
            check_file = None if output_dir is None else check_image_file(output_dir, rel_file)
            try:
                tag = rotate_orientation_safely(
                    image_file, out_file, rotation * 90, check_file, CHECK_QUALITY
                )
            except ValueError as e:
                logger.error(f"Skipping {rel_file}: {type(e).__name__}: {e} {report}")
                n_skipped += 1
                continue
            logger.info(
                f"Rotate {rotation * 90}° cw by orientation tag {tag}, keeping the extra "
                f"frames: {image_file} -> {out_file}"
            )
            logger.debug(f"{rel_file}: {report}")
            n_tagged += 1
            continue
        # the exif orientation is applied on load and the tag removed from the metadata, so the
        # written pixels are what a viewer shows
        image, metadata = load_image(image_file)
        rotated = rotate_image(image, (N_ROTATIONS - rotation) % N_ROTATIONS)
        # the embedded thumbnail and the image size tags describe the unrotated pixels, so
        # they are stale unless the exif is rewritten or dropped
        if args.destroy_metadata:
            metadata = ImageMetadata(
                icc_profile=metadata.icc_profile, dpi=metadata.dpi, png_text=metadata.png_text
            )
        elif args.clean_exif and metadata.exif is not None:
            cleaned = metadata.with_clean_exif()
            logger.debug(
                f"clean exif {len(metadata.exif)} -> {len(cleaned.exif or b'')} bytes "
                f"[{describe_exif_change(metadata.exif, cleaned.exif)}]: {rel_file}"
            )
            metadata = cleaned
        src_stat = image_file.stat()
        if output_dir is None:
            logger.info(f"Rotate {rotation * 90}° cw in place: {image_file}")
            save_image_safely(rotated, image_file, metadata, args.quality, src_stat)
            continue
        logger.info(f"Rotate {rotation * 90}° cw: {image_file} -> {out_file}")
        save_image_safely(rotated, out_file, metadata, args.quality, src_stat)
        save_image_safely(
            make_check_image(image, rotated),
            check_image_file(output_dir, rel_file),
            None,
            CHECK_QUALITY,
        )
    logger.info(
        f"Rotated {len(to_rotate) - n_skipped - n_tagged} photos, {n_tagged} by orientation "
        f"tag, skipped {n_skipped}."
    )


def check_image_file(output_dir: Path, rel_file: str) -> Path:
    """Jpgs keep their name, other formats get .jpg appended so two stems cannot collide."""
    name = rel_file if rel_file.lower().endswith((".jpg", ".jpeg")) else f"{rel_file}.jpg"
    return output_dir / CHECK_DIR_NAME / name


def rotate_orientation_safely(
    image_file: Path, out_file: Path, rotation_cw: int, check_file: Path | None, quality: int
) -> int:
    """
    Rewrite only the orientation tag, keeping every other byte, and write a check image.

    The new file goes to a temporary path first and replaces the target afterwards, so an
    interrupted run leaves the source untouched.
    """
    src_stat = image_file.stat()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_name(f"{out_file.stem}{TMP_STEM_SUFFIX}{out_file.suffix}")
    tag = rotate_by_orientation_tag(image_file, tmp_file, rotation_cw)
    tmp_file.replace(out_file)
    os.utime(out_file, (src_stat.st_atime, src_stat.st_mtime))
    if check_file is not None:
        image, _ = load_image(image_file)
        rotated = rotate_image(image, (N_ROTATIONS - rotation_cw // 90) % N_ROTATIONS)
        save_image_safely(make_check_image(image, rotated), check_file, None, quality)
    return tag


def save_image_safely(
    image: Image.Image,
    out_file: Path,
    metadata: ImageMetadata | None,
    quality: int,
    src_stat: os.stat_result | None = None,
) -> None:
    """Write to a temporary file and rename, so interrupted runs do not leave broken files."""
    tmp_file = out_file.with_name(f"{out_file.stem}{TMP_STEM_SUFFIX}{out_file.suffix}")
    save_image(image, tmp_file, metadata, quality=quality)
    # read the written file back before it replaces anything, so metadata a format or an
    # encoder cannot carry is caught instead of disappearing quietly
    expected_exif = None if metadata is None else metadata.exif
    for drift in check_exif_survived(tmp_file, expected_exif):
        logger.warning(f"{out_file.name}: exif value changed when written: {drift}")
    tmp_file.replace(out_file)
    if src_stat is not None:
        os.utime(out_file, (src_stat.st_atime, src_stat.st_mtime))


def compute_predictions(
    input_dir: Path, classifier: OrientationClassifier, args: Args, exclude_dir: Path | None = None
) -> tuple[list[str], dict[str, dict]]:
    """
    Classify all images in the directory in all 4 rotations, reusing cached predictions.

    Args:
        input_dir: directory of images
        classifier: orientation classifier
        args: script arguments
        exclude_dir: images inside this directory are skipped, e.g. the output dir

    Returns:
        relative paths of the readable images in name order, their cache entries with
        keys logp (4, 4), width, height, size, mtime
    """
    src_index = index_files(input_dir, not args.non_recursive, args)
    logger.info(f"Found {len(src_index)} files in {input_dir}")
    rel_files = natsorted(
        f
        for f in src_index.keys()
        if any(f.lower().endswith(a) for a in ENDINGS)
        and not Path(f).stem.endswith(TMP_STEM_SUFFIX)
        and not is_inside_dir(f, input_dir, exclude_dir)
    )
    logger.info(f"From those detected {len(rel_files)} images.")
    if len(rel_files) == 0:
        raise ValueError(
            f"No images found in {input_dir} ({len(src_index)} files indexed, "
            f"image endings: {ENDINGS}, subdirectories searched: {not args.non_recursive})"
        )
    cache = EmbeddingCache("rotation", input_dir, reset=args.reset_cache)
    entries = {}
    todo = []
    for rel_file in rel_files:
        props = src_index[rel_file]
        entry = cache.get(rel_file, props.st_size, props.st_mtime)
        if entry is None:
            todo.append(rel_file)
        else:
            entries[rel_file] = entry
    logger.info(f"{len(entries)} images cached, {len(todo)} to classify.")
    broken_files = {}
    if len(todo) > 0:
        dataset = RotationImageDataset([input_dir / f for f in todo], classifier.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_as_lists,
        )
        pbar = tqdm(total=len(todo), desc="Classifying")
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
            log_probs = classifier.classify_rotations(torch.stack([tensors[i] for i in valid]))
            for i, log_prob in zip(valid, log_probs):
                rel_file = todo[indices[i]]
                props = src_index[rel_file]
                entries[rel_file] = cache.put(
                    rel_file,
                    props.st_size,
                    props.st_mtime,
                    logp=log_prob,
                    width=widths[i],
                    height=heights[i],
                )
        pbar.close()
    cache.save(keep_rel_files=rel_files)
    if len(broken_files) > 0:
        logger.warning(f"Skipped {len(broken_files)} unreadable files: {list(broken_files)[:5]}")
    rel_files = [f for f in rel_files if f in entries]
    if len(rel_files) == 0:
        raise ValueError(f"None of the images in {input_dir} could be read: {broken_files}")
    return rel_files, entries


if __name__ == "__main__":
    main()
