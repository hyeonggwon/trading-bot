"""Module entrypoint so `python -m tradingbot.cli` keeps working."""

from tradingbot.cli import app

if __name__ == "__main__":
    app()
