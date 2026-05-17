"""Legacy entrypoint: SUPERB/s3prl downstream was removed from this release."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "ERROR: downstream SUPERB (s3prl) is not part of this release. "
        "Use: python3 -m core experiments … or python3 -m core pipeline --run-upstream-first …",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
