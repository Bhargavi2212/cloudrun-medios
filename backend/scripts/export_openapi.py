from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app


def export_openapi(path: Path) -> None:
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
    path.write_text(json.dumps(schema, indent=2))
    print(f"Wrote OpenAPI schema to {path}")


if __name__ == "__main__":
    output = Path("openapi-schema.json")
    export_openapi(output)

