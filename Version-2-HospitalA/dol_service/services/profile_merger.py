"""
Utilities to merge local and remote portable profiles.
"""

from __future__ import annotations

from typing import Any


def merge_profiles(
    local_profile: dict[str, Any], remote_profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Merge remote profiles into the local payload.
    """

    merged = {
        "patient": local_profile["patient"],
        "timeline": list(local_profile.get("timeline", [])),
        "summaries": list(local_profile.get("summaries", [])),
        "sources": [local_profile.get("source_hospital")],
    }

    for profile in remote_profiles:
        merged["timeline"].extend(profile.get("timeline", []))
        merged["summaries"].extend(profile.get("summaries", []))
        source = profile.get("source_hospital")
        if source:
            merged["sources"].append(source)

    merged["timeline"].sort(key=lambda event: event["timestamp"])
    merged["sources"] = sorted(set(filter(None, merged["sources"])))
    return merged
