#!/usr/bin/env python3
"""Compatibility Adapter for the shared public-RGB redaction Implementation."""

from research.common.public_rgb_redaction import *  # noqa: F401,F403
from research.common.public_rgb_redaction import main


if __name__ == "__main__":
    raise SystemExit(main())
