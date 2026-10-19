"""Feature extraction and caching.

Import submodules directly to avoid loading ``extract`` twice when it is also
the ``-m`` entry point:

    from train.features.extract import extract_from_manifest
"""

__all__ = ["extract"]
