"""Allow `python -m jevmlx` to invoke the CLI (the bench --ui subprocess)."""

from jevmlx.cli import main

if __name__ == "__main__":
    main()
