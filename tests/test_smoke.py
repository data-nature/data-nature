"""Smoke tests — verify the package imports and basics work."""

import data_nature


def test_package_imports():
    """The package can be imported."""
    assert data_nature is not None


def test_version_is_set():
    """The package exposes a version string."""
    assert hasattr(data_nature, "__version__")
    assert isinstance(data_nature.__version__, str)
    assert len(data_nature.__version__) > 0


def test_version_format():
    """Version follows semver-ish format (e.g. '0.1.0')."""
    parts = data_nature.__version__.split(".")
    assert len(parts) >= 2, f"Expected at least major.minor, got {data_nature.__version__}"
    for part in parts:
        assert part.isdigit(), f"Non-numeric version part: {part}"
