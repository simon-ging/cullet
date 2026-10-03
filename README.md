# cullet

Keyboard-driven photo viewer for going through a folder fast: step, zoom, rotate, delete, move
into target folders, tag, and review duplicates. Press `h` in the viewer for the keys.

## Install

```
uv tool install cullet
```

or `pip install cullet` into an environment of your choice. Rotating JPEGs losslessly needs the
`jpegtran` binary, which comes with `libjpeg-turbo` on most distributions.

Finding duplicates needs torch and is an extra:

```
uv tool install "cullet[dedup]"
```

## Usage

```
cullet /path/to/photos
cullet /path/to/photos/IMG_1234.jpg
cullet /path/to/photos -a ../good -a ../maybe -T sun -T portrait
cullet /path/to/photos --dedup
```

The number keys move the image into the target folders (`-a`) and toggle the tags (`-T`), in the
order they are given. Deleted files are moved to a trash dir, never removed, and every file
operation can be undone with `u`. `cullet --help` lists all options.

## Development

```
pip install -e ".[dedup,dev]"
pytest
```
