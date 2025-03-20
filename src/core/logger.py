import sys

from loguru import logger

logger.remove(0)

config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": "{time} - {message}",
            "serialize": True,
            "level": "DEBUG",
        },
    ],
}

logger.configure(**config)  # type: ignore[arg-type]