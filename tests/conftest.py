import os

# the tests open real windows, which need no display on the offscreen platform
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# set in the full build, where every optional package and binary is installed
FULL_BUILD_VARIABLE = "CULLET_FULL_BUILD"


def pytest_sessionfinish(session, exitstatus):
    """Tests skip themselves when torch, numpy, ffmpeg or jpegtran is missing, which is what the
    minimal build wants. In the full build a skip means the setup lost something and the code
    behind it went untested, so it fails the run."""
    if not os.environ.get(FULL_BUILD_VARIABLE):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", [])
    if skipped and exitstatus == 0:
        reporter.write_line(
            f"{len(skipped)} skipped tests with {FULL_BUILD_VARIABLE} set, the full build must "
            f"run all of them. Run with -rs to see why they were skipped.",
            red=True,
        )
        session.exitstatus = 1
