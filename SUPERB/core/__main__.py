"""Entry: python -m core <subcommand> ... (same as python -m core.cli ...)."""

import sys

from core.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
