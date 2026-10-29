"""Download Zurich RAW-to-DSLR test dataset samples via HTTP range requests.

Uses `remotezip` to stream selective image files directly from the published
cloud archive without downloading the full multi-gigabyte dataset archive.
Provides exponential backoff retry, thread-local connection reuse, atomic file writes,
and SHA-256 integrity manifest generation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from remotezip import RemoteZip

ZRR_DATASET_URL = (
    "https://download.ai-benchmark.com/s/GsfikXymMKHxCwr/download/Zurich-RAW-to-DSLR-Dataset.zip"
)


class ResumableZipDownloader:
    """Manages thread-safe, selective HTTP range downloads from a remote ZIP archive."""

    def __init__(self, url: str, output_dir: Path, max_workers: int = 8) -> None:
        """Initialize the downloader.

        Args:
            url: Remote URL of the ZIP archive.
            output_dir: Destination path for extracted files.
            max_workers: Number of concurrent worker threads.
        """
        self.url = url
        self.output_dir = output_dir
        self.max_workers = max_workers
        self._local = threading.local()
        self._connections: List[RemoteZip] = []
        self._lock = threading.Lock()

    def _get_archive(self) -> RemoteZip:
        """Get or initialize thread-local RemoteZip connection."""
        if not hasattr(self._local, "archive"):
            archive = RemoteZip(self.url)
            self._local.archive = archive
            with self._lock:
                self._connections.append(archive)
        return self._local.archive

    def _close_thread_archive(self) -> None:
        """Close thread-local archive on connection failure."""
        if hasattr(self._local, "archive"):
            try:
                self._local.archive.close()
            except Exception:
                pass
            del self._local.archive

    def fetch_file(self, internal_path: str) -> Dict[str, Any]:
        """Fetch a single member file from the remote zip with retry logic.

        Args:
            internal_path: Path of the target file inside the remote ZIP.

        Returns:
            Dict[str, Any]: Metadata containing output path, byte length, and SHA-256.
        """
        target_path = self.output_dir / internal_path
        if target_path.is_file():
            content = target_path.read_bytes()
        else:
            content = b""
            for attempt in range(5):
                try:
                    archive = self._get_archive()
                    content = archive.read(internal_path)
                    break
                except Exception:
                    self._close_thread_archive()
                    if attempt == 4:
                        raise
                    time.sleep(min(2**attempt, 8))

            target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = target_path.with_suffix(target_path.suffix + ".part")
            temp_path.write_bytes(content)
            temp_path.replace(target_path)

        return {
            "path": str(target_path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    def fetch_pair(self, index: int) -> Dict[str, Any]:
        """Fetch matched Huawei RAW PNG and Canon DSLR reference JPG pair.

        Args:
            index: Dataset sample index (0 to 1203).

        Returns:
            Dict[str, Any]: Pair metadata with individual file hashes.
        """
        return {
            "id": index,
            "huawei_raw": self.fetch_file(f"test/huawei_raw/{index}.png"),
            "canon": self.fetch_file(f"test/canon/{index}.jpg"),
        }

    def download_dataset(
        self,
        count: int = 1204,
        full_resolution_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute parallel batch download and write manifest JSON.

        Args:
            count: Number of paired patches to fetch (1 to 1204).
            full_resolution_id: Optional sample ID to fetch full Bayer sensor crop.

        Returns:
            Dict[str, Any]: Full download manifest dictionary.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, Any] = {
            "source": self.url,
            "split": "test",
            "pairs": [],
        }

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for pair_meta in pool.map(self.fetch_pair, range(count)):
                    manifest["pairs"].append(pair_meta)
                    done = len(manifest["pairs"])
                    if done % 50 == 0 or done == count:
                        print(f"Prepared {done}/{count} matched pairs", flush=True)

            if full_resolution_id is not None:
                manifest["full_raw"] = self.fetch_file(
                    f"full_resolution/huawei_raw/{full_resolution_id}.png"
                )
        finally:
            with self._lock:
                for conn in self._connections:
                    try:
                        conn.close()
                    except Exception:
                        pass
                self._connections.clear()

        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Prepared {len(manifest['pairs'])} RAW/DSLR pairs in {self.output_dir}", flush=True)
        return manifest


def main() -> None:
    """CLI entry point for downloading Zurich RAW-to-DSLR dataset samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=1204,
        help="Number of test patches to download (1..1204, default: 1204)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download worker threads (default: 8)",
    )
    parser.add_argument(
        "--full-id",
        type=int,
        help="Optional full-resolution Bayer image ID (e.g. 1050)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/zrr_test_samples"),
        help="Destination directory (default: data/zrr_test_samples)",
    )

    args = parser.parse_args()
    if not (1 <= args.count <= 1204) or not (1 <= args.workers <= 16):
        parser.error("Expected 1..1204 pairs and 1..16 download workers")

    downloader = ResumableZipDownloader(
        url=ZRR_DATASET_URL,
        output_dir=args.output,
        max_workers=args.workers,
    )
    downloader.download_dataset(count=args.count, full_resolution_id=args.full_id)


if __name__ == "__main__":
    main()
