"""Shared loguru configuration used across CLI tools and services."""

from __future__ import annotations

import sys

from loguru import logger


LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[component]}</cyan> | "
    "<level>{message}</level>"
)


def configure_logging(*, verbose: bool = False, name: str | None = None):
    """Configure the shared repository logger."""

    component = name or "app"
    logger.remove()
    logger.configure(extra={"component": component})
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format=LOG_FORMAT,
        diagnose=verbose,
        backtrace=verbose,
    )
    return logger.bind(component=component)
