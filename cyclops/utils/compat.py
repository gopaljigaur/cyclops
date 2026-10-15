"""Shims for differences between supported Python versions."""

import sys

if sys.version_info >= (3, 11):
    from asyncio import timeout as asyncio_timeout
else:
    # wait_for before 3.12 can drop a cancellation that comes as the call ends.
    from async_timeout import timeout as asyncio_timeout

__all__ = ["asyncio_timeout"]
