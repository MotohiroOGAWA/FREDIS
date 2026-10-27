"""Command line interface for FREDIS."""

import os

# UCX (loaded with torch) watches /tmp with inotify for its optional VFS
# monitoring service and prints "inotify_add_watch(/tmp) failed" when the
# per-user inotify limit is exhausted, once per worker process. FREDIS never
# uses that service; an explicit UCX_VFS_ENABLE in the environment still wins.
os.environ.setdefault("UCX_VFS_ENABLE", "n")
