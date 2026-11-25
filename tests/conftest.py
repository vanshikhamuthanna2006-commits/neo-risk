from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.config import SBDB_OBJECT_URL, SBDB_QUERY_URL, SENTRY_URL

FIX = Path(__file__).parent / "fixtures"


class FixtureGetter:
    """Deterministic stand-in for JPLClient.get_json, backed by real captured JPL payloads.

    Honours SBDB-query ``limit`` / ``limit-from`` so paging logic is exercised.
    """

    def __init__(self) -> None:
        self.sbdb = json.loads((FIX / "sbdb_query_sample.json").read_text())
        self.sentry = json.loads((FIX / "sentry_sample.json").read_text())
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((url, dict(params)))
        if url == SBDB_QUERY_URL:
            off = int(params.get("limit-from", 0))
            lim = int(params.get("limit", len(self.sbdb["data"])))
            return {**self.sbdb, "data": self.sbdb["data"][off:off + lim],
                    "count": len(self.sbdb["data"])}
        if url == SENTRY_URL:
            return self.sentry
        if url == SBDB_OBJECT_URL:
            fp = FIX / f"sbdb_cov_{params['sstr'].replace(' ', '_')}.json"
            if not fp.exists():
                raise RuntimeError(f"no fixture for {params['sstr']}")
            return json.loads(fp.read_text())
        raise ValueError(url)


@pytest.fixture
def getter() -> FixtureGetter:
    return FixtureGetter()
