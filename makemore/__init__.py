import logging.config
import os

CWD = os.getcwd()
rootdir = os.path.dirname(CWD)
basename = os.path.basename(CWD)

if basename == "docs":
    basename = ""

if basename == "source":
    basename = ""
    rootdir = os.path.dirname(rootdir)

log_config_fpath = os.path.join(rootdir, basename, "configs/logging.ini")

logging.config.fileConfig(
    fname=log_config_fpath,
    disable_existing_loggers=False,
    encoding="utf-8",
)

logger = logging.getLogger(__name__)
