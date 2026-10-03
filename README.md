# cullet

<p align="center">
<a href="https://github.com/simon-ging/cullet/actions/workflows/build-py310.yml">
  <img alt="build 3.10 status" title="build 3.10 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/cullet/build-py310.yml?branch=main&label=python%203.10" />
</a>
<a href="https://github.com/simon-ging/cullet/actions/workflows/build-py314.yml">
  <img alt="build 3.14 status" title="build 3.14 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/cullet/build-py314.yml?branch=main&label=python%203.14" />
</a>
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

Finding duplicates needs torch and is an extra:

```bash
uv tool install "cullet[dedup]"
```

## Usage

```bash
cullet /path/to/photos
cullet /path/to/photos/IMG_1234.jpg
cullet /path/to/photos -a ../good -a ../maybe -T sun -T portrait
cullet /path/to/photos --dedup
```

The number keys move the image into the target folders (`-a`) and toggle the tags (`-T`), in the
order they are given. Deleted files are moved to a trash dir, never removed, and every file
operation can be undone with `u`. `cullet --help` lists all options.

## Install locally and run tests

Clone repository and cd into. Setup python 3.10 or higher.
Note: The tests of the duplicate detection are skipped unless torch is installed.

```bash
pip install -e .[dedup,dev]
pylint src

# run tests
python -m pytest --cov
pylint tests
```
