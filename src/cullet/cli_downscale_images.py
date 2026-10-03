"""
Downscale images, fix wrong extensions, convert formats and clean up metadata.

Images are written as jpg, except those with transparency which are written as png. Pngs
without transparency stay png unless -c is given. Only images that need a change are rewritten:
too big (-s/-b), wrong extension, unsupported format, opaque png with -c, or metadata to
rewrite with -E or --destroy_metadata. Other images are left alone (in place with -O) or
copied to the output dir.

-s and -b resize every image above the target. With -S or -B only images above that threshold
are resized, so a photo a little larger than the target keeps its pixels instead of being
re-encoded for a few percent of disk.

Re-encoded images get the exif rotation baked into the pixels. Transposing removes the exif
and xmp orientation fields and can drop the embedded thumbnail. Orientation 1 can remain
without -E. Other exif tags are kept, so size tags and any retained thumbnail can become stale.
TIFF metadata is rebuilt without fields describing its pixel storage. -E drops the thumbnail,
all exif orientation tags and the size tags while keeping the remaining metadata.
--destroy_metadata throws the exif and the xmp away completely. Images that are only renamed
or copied keep their bytes untouched, rotation tag included.

Examples:
    # downscale so the smaller side is 1080 px, into input_dir/downscaled_output
    cullet-downscale-images /path/to/images -s 1080 -w
    # the same, overwriting the input files and rewriting the exif
    cullet-downscale-images /path/to/images -s 1080 -E -O -w
"""

import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional

import filetype
from attrs import define
from PIL import Image, UnidentifiedImageError
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from cullet.files import PathSpecArgs, index_files
from cullet.images import (
    ImageMetadata,
    check_exif_survived,
    describe_exif_change,
    extra_frames_are_redundant,
    has_transparency,
    load_image,
    save_image,
    scale_to_bigger_side,
    scale_to_smaller_side,
)
from cullet.logs import configure_logging
from cullet.paths import is_inside_dir

logger = logging.getLogger(__name__)


@define
class Args(PathSpecArgs, VerboseQuietArgs):
    input_dir: Optional[Path] = add_argument(positional=True, type=str, help="Input image dir")
    non_recursive: bool = add_argument(
        shortcut="-r", action="store_true", help="Do not search subdirectories for images."
    )
    smaller_side: int | None = add_argument(
        shortcut="-s",
        type=int,
        help="Resize images so that the smaller side is this many pixels long.",
    )
    smaller_side_trigger: int | None = add_argument(
        shortcut="-S",
        type=int,
        help="Only resize when the smaller side is longer than this, so images that are only "
        "a bit above -s are left alone instead of being re-encoded for little gain. Must be "
        "larger than -s.",
    )
    bigger_side: int | None = add_argument(
        shortcut="-b",
        type=int,
        help="Resize images so that the larger side is this many pixels long.",
    )
    bigger_side_trigger: int | None = add_argument(
        shortcut="-B",
        type=int,
        help="Only resize when the larger side is longer than this, see -S. Must be larger "
        "than -b.",
    )
    quality: int = add_argument(
        shortcut="-Q", type=int, default=95, help="Jpg quality for rewritten images."
    )
    compress_pngs: bool = add_argument(
        shortcut="-c", action="store_true", help="Convert pngs without transparency to jpg."
    )
    clean_exif: bool = add_argument(
        shortcut="-E",
        action="store_true",
        help="Rewrite the exif so it describes the written pixels: drop the embedded thumbnail "
        "and remove the orientation and image size tags. Every other tag is kept, "
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
        default="downscaled_output",
    )
    overwrite: bool = add_argument(
        shortcut="-O", action="store_true", help="Overwrite files instead of a separate output dir"
    )
    write: bool = add_argument(shortcut="-w", action="store_true", help="Write changes to disk")


ENDINGS = [".jpg", ".jpeg", ".jfif", ".png", ".tiff", ".tif", ".bmp", ".webp"]
# pillow refuses images above 2x this limit as decompression bombs, panoramas are bigger than that
Image.MAX_IMAGE_PIXELS = 10 * Image.MAX_IMAGE_PIXELS
OUTPUT_EXTENSION = ".jpg"
ALPHA_EXTENSION = ".png"
FORMAT_TO_EXTENSION = {"JPEG": OUTPUT_EXTENSION, "PNG": ALPHA_EXTENSION}
# extensions that are correct for a format as pillow names it
EXTENSIONS_OF_FORMAT = {
    "JPEG": (".jpg", ".jpeg", ".jfif"),
    # a jpeg holding more than one image, written by cameras and phones with a jpg extension.
    # only the first frame is kept, so it is rewritten as a plain jpeg
    "MPO": (".jpg", ".jpeg", ".jfif"),
    "PNG": (".png",),
    "TIFF": (".tiff", ".tif"),
    "BMP": (".bmp",),
    "WEBP": (".webp",),
}
# MPO is a jpeg that holds more than one image, so writing it as a plain jpeg is not a change
# a viewer notices and never justifies a rewrite on its own. once the file is re-encoded for
# another reason the extra frames are dropped, which extra_frames_are_redundant has to allow
EQUIVALENT_FORMATS = {"MPO": "JPEG"}
# unfinished outputs of interrupted runs, never used as input
TMP_STEM_SUFFIX = ".tmp"


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logging(args)
    logger.info(f"{args}")
    if args.smaller_side is not None and args.bigger_side is not None:
        raise ValueError("Specify either --smaller_side or --bigger_side but not both.")
    for side, trigger, names in (
        (args.smaller_side, args.smaller_side_trigger, ("-s", "-S")),
        (args.bigger_side, args.bigger_side_trigger, ("-b", "-B")),
    ):
        if trigger is None:
            continue
        if side is None:
            raise ValueError(f"{names[1]} only means something together with {names[0]}.")
        if trigger <= side:
            raise ValueError(
                f"{names[1]}={trigger} must be larger than {names[0]}={side}, otherwise every "
                f"image it triggers on is already at or below the target size."
            )
    if args.clean_exif and args.destroy_metadata:
        raise ValueError("Specify either --clean_exif or --destroy_metadata but not both.")
    input_dir = Path(args.input_dir)
    # the output dir lives inside the input dir, so it has to be kept out of the index or a
    # second run would process what the first one wrote
    output_dir = input_dir if args.overwrite else input_dir / args.output_dir_rel
    exclude_dir = None if args.overwrite else output_dir

    # make index of images to process. sources are relative to input_dir
    src_index = index_files(input_dir, not args.non_recursive, args)
    logger.info(f"Found {len(src_index)} files.")
    files2 = [
        f
        for f in src_index.keys()
        if any(f.lower().endswith(a) for a in ENDINGS)
        and not Path(f).stem.endswith(TMP_STEM_SUFFIX)
        and not is_inside_dir(f, input_dir, exclude_dir)
    ]
    logger.info(f"From those detected {len(files2)} images.")
    # sources that could keep their name come first, so they get to claim it before the others
    files2.sort(key=lambda f: (not f.lower().endswith((OUTPUT_EXTENSION, ALPHA_EXTENSION)), f))
    source_files_dict = {f: None for f in files2}

    target2source = {}
    broken_files = []
    actions = Counter()
    for src_file_rel in source_files_dict.keys():
        src_file_path = input_dir / src_file_rel

        # load image and detect its format. a directory of images can contain broken files
        # or files with a wrong extension, those are skipped instead of stopping the whole run
        try:
            orig_img, metadata = load_image(src_file_path)
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Skipping {src_file_rel}: {type(e).__name__}: {e}")
            broken_files.append(src_file_rel)
            actions["unreadable"] += 1
            continue
        src_format = metadata.format
        if src_format not in EXTENSIONS_OF_FORMAT:
            logger.error(f"Skipping {src_file_rel}: unsupported format {src_format}")
            broken_files.append(src_file_rel)
            actions["unsupported_format"] += 1
            continue
        # second opinion on the format from the file header. pillow decoded the image so it
        # is trusted, but a disagreement is worth a look
        header_type = filetype.guess(src_file_path.as_posix())
        if (
            header_type is not None
            and f".{header_type.extension}" not in EXTENSIONS_OF_FORMAT[src_format]
        ):
            logger.warning(
                f"{src_file_rel}: pillow reads {src_format} but the header looks like "
                f"{header_type.extension}"
            )
        old_extension = src_file_path.suffix.lower()
        extension_ok = old_extension in EXTENSIONS_OF_FORMAT[src_format]

        # target format: jpg, png for transparency, png stays png unless -c
        if has_transparency(orig_img) or (src_format == "PNG" and not args.compress_pngs):
            target_format = "PNG"
        else:
            target_format = "JPEG"
        # an extension that already fits the target format is kept, e.g. .jpeg
        if old_extension in EXTENSIONS_OF_FORMAT[target_format]:
            extension = old_extension
        else:
            extension = FORMAT_TO_EXTENSION[target_format]
        is_new_extension = extension != old_extension

        tgt_file_rel_stem = ".".join(src_file_rel.split(".")[:-1])
        tgt_file_rel = f"{tgt_file_rel_stem}{extension}"

        # rename until no more conflicts
        for i in range(100):
            taken = False
            if tgt_file_rel in target2source:
                # same target twice, must be renamed
                taken = True
            if tgt_file_rel in source_files_dict and is_new_extension and args.overwrite:
                # target points to another source file, must be renamed
                taken = True
            if not taken:
                # current filename is fine
                break
            tgt_file_rel_new = f"{tgt_file_rel_stem}-copy{i:02d}{extension}"
            logger.warning(f"Target {tgt_file_rel} taken -> rename to {tgt_file_rel_new}")
            tgt_file_rel = tgt_file_rel_new
        else:
            raise ValueError(f"Could not find a free name for {src_file_rel} after 100 tries.")

        # store the target for checking later
        target2source[tgt_file_rel] = src_file_rel
        out_file = output_dir / tgt_file_rel

        # figure out what has to change, as (action name, message) pairs. the action names go
        # into the summary, the messages into the log line. minor_changes still cause a rewrite
        # but are not worth a line of output on their own
        changes = []
        minor_changes = []
        img = orig_img
        orig_width, orig_height = orig_img.size
        # a trigger keeps images that are only slightly too big untouched, resizing them would
        # cost a re-encode and gain almost nothing
        if args.smaller_side is not None and _over_trigger(
            min(orig_width, orig_height), args.smaller_side_trigger
        ):
            img = scale_to_smaller_side(orig_img, args.smaller_side)
        elif args.bigger_side is not None and _over_trigger(
            max(orig_width, orig_height), args.bigger_side_trigger
        ):
            img = scale_to_bigger_side(orig_img, args.bigger_side)
        width, height = img.size
        if width * height < orig_width * orig_height:
            changes.append(
                ("downscale", f"downscale {orig_width}x{orig_height} -> {width}x{height}")
            )
        else:
            # never upscale
            img = orig_img
        if EQUIVALENT_FORMATS.get(src_format, src_format) != target_format:
            changes.append(("convert", f"convert {src_format} -> {target_format}"))
        save_metadata = metadata
        if args.destroy_metadata and (metadata.exif is not None or metadata.xmp is not None):
            changes.append(("destroy_metadata", "destroy metadata"))
            save_metadata = ImageMetadata(
                icc_profile=metadata.icc_profile, dpi=metadata.dpi, png_text=metadata.png_text
            )
        elif args.clean_exif and metadata.exif is not None:
            save_metadata = metadata.with_clean_exif()
            minor_changes.append(
                (
                    "clean_exif",
                    f"clean exif {len(metadata.exif)} -> "
                    f"{len(save_metadata.exif or b'')} bytes "
                    f"[{describe_exif_change(metadata.exif, save_metadata.exif)}]",
                )
            )
        # a wrong extension is fixed by renaming, unless the format is converted anyway
        fix_extension = not extension_ok and src_format == target_format
        only_rename = len(changes) == 0 and len(minor_changes) == 0 and fix_extension
        # a multi frame jpeg loses everything but the first frame when it is re-encoded, which
        # is fine for a smaller copy of the same shot and not fine for auxiliary data. a file
        # that is only copied or renamed keeps every frame, so it is not checked at all
        if src_format == "MPO" and not only_rename and (changes or minor_changes):
            redundant, report = extra_frames_are_redundant(src_file_path)
            if not redundant:
                logger.error(
                    f"Skipping {src_file_rel}: the extra frames of this multi frame jpeg are "
                    f"not copies of the first, re-encoding it would lose them. {report}"
                )
                broken_files.append(src_file_rel)
                actions["extra_frames_differ"] += 1
                continue
            logger.debug(f"{src_file_rel}: dropping redundant extra frames. {report}")
            # the conversion rides along with the rewrite rather than causing it, so it is
            # reported here instead of in the list that decides whether to rewrite
            minor_changes.append(
                ("convert", f"convert {src_format} -> {target_format}, extra frames dropped")
            )
        if fix_extension:
            changes.append(("fix_extension", f"fix extension {old_extension} -> {extension}"))
        all_changes = changes + minor_changes

        # if the source was written under a different name it stays around, otherwise it is replaced
        delete_source = args.overwrite and tgt_file_rel != src_file_rel
        if len(all_changes) == 0:
            if args.overwrite:
                logger.debug(f"Nothing to do for {src_file_rel}")
                actions["none"] += 1
                continue
            logger.debug(f"Copy {src_file_rel} -> {out_file}")
            actions["copy"] += 1
            if args.write:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file_path, out_file)
            continue

        log_change = logger.info if changes else logger.debug
        log_change(f"{', '.join(m for _, m in all_changes)}: {src_file_rel} -> {out_file}")
        action_names = [a for a, _ in all_changes]
        if delete_source and not only_rename:
            logger.info(f"Delete {src_file_path} which was written to {out_file}")
            action_names.append("delete_source")
        actions[",".join(action_names)] += 1
        if not args.write:
            continue

        out_file.parent.mkdir(parents=True, exist_ok=True)
        src_stat = src_file_path.stat()
        if only_rename:
            # same format and pixels, so no need to re-encode
            if args.overwrite:
                src_file_path.rename(out_file)
            else:
                shutil.copy2(src_file_path, out_file)
            continue
        # write to a temporary file so interrupted runs do not leave broken outputs behind
        tmp_file = out_file.with_name(f"{out_file.stem}{TMP_STEM_SUFFIX}{extension}")
        save_image(img, tmp_file, save_metadata, quality=args.quality)
        # read the written file back before it replaces anything, so metadata a format or an
        # encoder cannot carry is caught instead of disappearing quietly
        for drift in check_exif_survived(tmp_file, save_metadata.exif):
            logger.warning(f"{src_file_rel}: exif value changed when written: {drift}")
        tmp_file.replace(out_file)
        os.utime(out_file, (src_stat.st_atime, src_stat.st_mtime))
        if delete_source:
            src_file_path.unlink()

    summary = {k: v for k, v in actions.most_common() if v}
    logger.info(f"Images: {len(files2)}, actions: {summary}")
    if len(broken_files) > 0:
        logger.warning(f"Skipped {len(broken_files)} unreadable files, e.g. {broken_files[:5]}")
    if not args.write:
        logger.warning("Dry run. Use -w to write changes to disk.")


def _over_trigger(side: int, trigger: int | None) -> bool:
    """Without a trigger every image is resized, with one only those longer than it."""
    return trigger is None or side > trigger


if __name__ == "__main__":
    main()
