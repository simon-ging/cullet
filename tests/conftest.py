import os

# the tests open real windows, which need no display on the offscreen platform
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
