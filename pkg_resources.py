"""Lightweight pkg_resources shim for environments without setuptools.

This module only implements the subset of functionality required by
fyers_apiv3.FyersWebsocket.data_ws, namely ``resource_filename``.

In normal Python environments, ``pkg_resources`` is provided by the
``setuptools`` package. On some minimal platforms (like certain
serverless / cloud runtimes), that module may not be present. Creating
this shim in the project root ensures that ``from pkg_resources import
resource_filename`` continues to work.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

try:  # Python 3.9+ style APIs
    from importlib import resources as _resources  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - extremely old Python
    _resources = None  # type: ignore[assignment]


def resource_filename(package_or_requirement: Any, resource_name: str) -> str:
    """Best-effort implementation of ``pkg_resources.resource_filename``.

    Parameters
    ----------
    package_or_requirement:
        Module object or dotted-path string (e.g. "fyers_apiv3.FyersWebsocket").
    resource_name:
        Relative resource path inside that package.

    Returns
    -------
    str
        Filesystem path to the requested resource. If discovery fails,
        returns a best-effort fallback path (resource_name in CWD).
    """

    # Normalise package name
    pkg_name: str
    try:
        if hasattr(package_or_requirement, "__spec__") and getattr(
            package_or_requirement.__spec__, "name", None
        ):
            pkg_name = package_or_requirement.__spec__.name  # type: ignore[assignment]
        elif hasattr(package_or_requirement, "__name__"):
            pkg_name = package_or_requirement.__name__  # type: ignore[assignment]
        else:
            pkg_name = str(package_or_requirement)
    except Exception:
        pkg_name = str(package_or_requirement)

    # Prefer importlib.resources when available
    if _resources is not None:
        try:
            return str(_resources.files(pkg_name).joinpath(resource_name))  # type: ignore[attr-defined]
        except Exception:
            pass

    # Fallback: derive path from imported module file
    try:
        module = importlib.import_module(pkg_name)
        base = Path(module.__file__ or "").resolve().parent
        return str(base / resource_name)
    except Exception:
        # Final fallback: just return a path relative to current working dir
        return os.path.join(os.getcwd(), resource_name)


__all__ = ["resource_filename"]
