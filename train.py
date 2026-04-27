import argparse
from pathlib import Path

from src.config import AppConfig
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train the 180 s GNSS slip model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a JSON config file.")
    args = parser.parse_args()
    app_config = AppConfig.from_json(args.config)
    Trainer(app_config.data, app_config.model, app_config.train).fit()


if __name__ == "__main__":
    main()
