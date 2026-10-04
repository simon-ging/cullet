# cullet

<p align="center">
<a href="https://github.com/simon-ging/cullet/actions/workflows/build-py310.yml">
  <img alt="minimal build 3.10 status" title="minimal build 3.10 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/cullet/build-py310.yml?branch=main&label=minimal%20build%203.10" />
</a>
<a href="https://github.com/simon-ging/cullet/actions/workflows/build-py314.yml">
  <img alt="minimal build 3.14 status" title="minimal build 3.14 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/cullet/build-py314.yml?branch=main&label=minimal%20build%203.14" />
</a>
<br />
<a href="https://github.com/simon-ging/cullet/actions/workflows/build-py310-full.yml">
  <img alt="full build 3.10 status" title="full build 3.10 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/cullet/build-py310-full.yml?branch=main&label=full%20build%203.10" />
</a>
<a href="https://github.com/simon-ging/cullet/actions/workflows/build-py314-full.yml">
  <img alt="full build 3.14 status" title="full build 3.14 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/cullet/build-py314-full.yml?branch=main&label=full%20build%203.14" />
</a>
<br />
<img alt="coverage" title="coverage" src="https://raw.githubusercontent.com/simon-ging/cullet/main/docs/coverage.svg" />
<a href="https://pypi.org/project/cullet/">
  <img alt="version" title="version" src="https://img.shields.io/pypi/v/cullet?color=success" />
</a>
</p>

Keyboard-driven photo viewer for going through a folder fast: step, zoom, rotate, delete, move
into target folders, tag, and review duplicates. Press `h` in the viewer for the keys.

## Install

Requires `python>=3.10`

```bash
uv tool install cullet
```

or `pip install cullet` into an environment of your choice. Rotating JPEGs losslessly needs the
`jpegtran` binary, which comes with `libjpeg-turbo` on most distributions.

Finding duplicates and fixing rotations need torch, which comes with the full extra:

```bash
uv tool install "cullet[full]"
```

### Open with

To get cullet into the "Open with" menu of the file manager for photos and folders, install the
desktop file of this repository:

```bash
curl -o ~/.local/share/applications/cullet.desktop https://raw.githubusercontent.com/simon-ging/cullet/main/packaging/cullet.desktop
update-desktop-database ~/.local/share/applications
# optional, make it the default for photos
xdg-mime default cullet.desktop image/jpeg image/png
```

This needs the `cullet` command on the PATH of the desktop session, `uv tool install` puts it
into `~/.local/bin`.

## Usage

```bash
cullet /path/to/photos
cullet /path/to/photos/IMG_1234.jpg
cullet /path/to/photos -a ../good -a ../maybe -T sun -T portrait
cullet /path/to/photos --dedup
```

The number keys move the image into the target folders (`-a`) and toggle the tags (`-T`), in the
order they are given. Deleted files are moved to the system trash, never removed (`--trash_dir`
puts them into a folder of your choice instead), and every file operation can be undone with
`u`. `cullet --help` lists all options.

## Command line tools

They only log what they would do until `-w` is given. Each has `--help`.

| command | does | needs |
| --- | --- | --- |
| `cullet-dedup-images` | removes duplicate images, into the system trash or a folder with `-Q` | full extra |
| `cullet-dedup-videos` | the same for videos | full extra, ffmpeg |
| `cullet-fix-rotation-images` | finds photos stored sideways or upside down and rotates them | full extra |
| `cullet-fix-rotation-videos` | the same for videos | full extra, ffmpeg |
| `cullet-downscale-images` | shrinks images, keeping their metadata | |
| `cullet-downscale-videos` | shrinks videos | ffmpeg |
| `cullet-rotate-videos` | rotates the videos you name by a given angle | ffmpeg |

```bash
cullet-dedup-videos /path/to/videos -Q /path/to/quarantine -w
cullet-downscale-images /path/to/photos -s 1080 -w
```

## Install locally and run tests

Clone repository and cd into. Setup python 3.10 or higher.

Dev build: install the clone as the `cullet` command, with torch, so code changes are live:

```bash
uv tool install --editable ".[full]"
```

Note: The tests that need torch are skipped unless it is installed.

```bash
pip install -e .[full,dev]
pylint src

# run tests
python -m pytest --cov
pylint tests
```
