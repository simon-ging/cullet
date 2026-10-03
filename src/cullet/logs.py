import logging

from typedparser import VerboseQuietArgs

VERBOSITY_LADDER = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


def configure_logging(args: VerboseQuietArgs) -> None:
    """Log to the terminal. INFO by default, every -v or -q moves one step along the ladder."""
    if args.loglevel is not None:
        assert not (args.verbose or args.quiet), "Cannot set both -v/-q and --loglevel"
        assert args.loglevel in VERBOSITY_LADDER, f"{args.loglevel=} not in {VERBOSITY_LADDER}"
        level = args.loglevel
    else:
        index = VERBOSITY_LADDER.index("INFO") + args.verbose - args.quiet
        level = VERBOSITY_LADDER[max(0, min(len(VERBOSITY_LADDER) - 1, index))]
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname).4s %(message)s", datefmt="%Y%m%d %H:%M:%S"
    )
