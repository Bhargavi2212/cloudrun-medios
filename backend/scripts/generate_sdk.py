#!/usr/bin/env python3
"""
SDK Generation Script for Medi OS API

This script generates client SDKs from the OpenAPI specification.
Supports multiple languages: TypeScript, Python, JavaScript, Go, etc.

Requirements:
    - openapi-generator-cli (install via npm: npm install -g @openapitools/openapi-generator-cli)
    - Or use Docker: docker pull openapitools/openapi-generator-cli

Usage:
    python backend/scripts/generate_sdk.py --language typescript --output sdk/typescript
    python backend/scripts/generate_sdk.py --language python --output sdk/python
    python backend/scripts/generate_sdk.py --language javascript --output sdk/javascript
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app


def export_openapi_schema(output_path: Path) -> dict:
    """Export OpenAPI schema from FastAPI app."""
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2))
    print(f"✅ Exported OpenAPI schema to {output_path}")
    return schema


def generate_sdk_with_openapi_generator(
    openapi_schema_path: Path,
    language: str,
    output_dir: Path,
    package_name: str = "medios-api-client",
    package_version: str = "2.0.0",
) -> bool:
    """Generate SDK using openapi-generator-cli.

    Args:
        openapi_schema_path: Path to OpenAPI schema JSON
        language: Target language (typescript, python, javascript, go, etc.)
        output_dir: Output directory for generated SDK
        package_name: Package name for the generated SDK
        package_version: Package version

    Returns:
        True if generation succeeded, False otherwise
    """
    # Check if openapi-generator-cli is available
    try:
        result = subprocess.run(
            ["openapi-generator-cli", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print("❌ openapi-generator-cli not found. Trying Docker...")
            return generate_sdk_with_docker(openapi_schema_path, language, output_dir, package_name, package_version)
    except FileNotFoundError:
        print("❌ openapi-generator-cli not found. Trying Docker...")
        return generate_sdk_with_docker(openapi_schema_path, language, output_dir, package_name, package_version)

    # Generate SDK using openapi-generator-cli
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "openapi-generator-cli",
        "generate",
        "-i",
        str(openapi_schema_path.absolute()),
        "-g",
        language,
        "-o",
        str(output_dir.absolute()),
        "--package-name",
        package_name,
        "--package-version",
        package_version,
        "--additional-properties",
        (f"npmName={package_name},npmVersion={package_version}" if language in ("typescript", "javascript") else ""),
    ]

    # Remove empty additional-properties if not needed
    if language not in ("typescript", "javascript"):
        cmd = [c for c in cmd if c != "--additional-properties" or (c == "--additional-properties" and cmd[cmd.index(c) + 1])]
        # Simplified: just remove the npm properties
        cmd_filtered = []
        skip_next = False
        for i, arg in enumerate(cmd):
            if skip_next:
                skip_next = False
                continue
            if arg == "--additional-properties" and language not in (
                "typescript",
                "javascript",
            ):
                skip_next = True
                continue
            cmd_filtered.append(arg)
        cmd = cmd_filtered

    print(f"🚀 Generating {language} SDK...")
    print(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ SDK generated successfully to {output_dir}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ SDK generation failed: {exc}")
        print(exc.stderr)
        return False


def generate_sdk_with_docker(
    openapi_schema_path: Path,
    language: str,
    output_dir: Path,
    package_name: str = "medios-api-client",
    package_version: str = "2.0.0",
) -> bool:
    """Generate SDK using Docker openapi-generator.

    Args:
        openapi_schema_path: Path to OpenAPI schema JSON
        language: Target language
        output_dir: Output directory for generated SDK
        package_name: Package name
        package_version: Package version

    Returns:
        True if generation succeeded, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use absolute paths for Docker volume mounting
    schema_abs = openapi_schema_path.absolute()
    output_abs = output_dir.absolute()

    # Docker command
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{schema_abs.parent}:/local/input",
        "-v",
        f"{output_abs.parent}:/local/output",
        "openapitools/openapi-generator-cli",
        "generate",
        "-i",
        f"/local/input/{schema_abs.name}",
        "-g",
        language,
        "-o",
        f"/local/output/{output_abs.name}",
        "--package-name",
        package_name,
    ]

    if language in ("typescript", "javascript"):
        docker_cmd.extend(
            [
                "--additional-properties",
                f"npmName={package_name},npmVersion={package_version}",
            ]
        )

    print(f"🚀 Generating {language} SDK using Docker...")
    print(f"   Command: {' '.join(docker_cmd)}")

    try:
        result = subprocess.run(docker_cmd, check=True, capture_output=True, text=True)
        print(f"✅ SDK generated successfully to {output_dir}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ SDK generation failed: {exc}")
        print(exc.stderr)
        return False
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker or openapi-generator-cli.")
        print("\n📦 Installation options:")
        print("   1. Install openapi-generator-cli: npm install -g @openapitools/openapi-generator-cli")
        print("   2. Install Docker: https://docs.docker.com/get-docker/")
        return False


def generate_typescript_sdk_simple(openapi_schema_path: Path, output_dir: Path) -> bool:
    """Generate a simple TypeScript SDK using openapi-typescript-codegen.

    This is a lighter-weight alternative that doesn't require openapi-generator-cli.
    """
    try:
        import subprocess

        # Check if openapi-typescript-codegen is available
        result = subprocess.run(
            ["npx", "--yes", "openapi-typescript-codegen", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "npx",
            "--yes",
            "openapi-typescript-codegen",
            "--input",
            str(openapi_schema_path.absolute()),
            "--output",
            str(output_dir.absolute()),
            "--client",
            "fetch",
        ]

        print(f"🚀 Generating TypeScript SDK using openapi-typescript-codegen...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ TypeScript SDK generated successfully to {output_dir}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ TypeScript SDK generation failed: {exc}")
        print(exc.stderr)
        return False
    except Exception as exc:
        print(f"❌ Error generating TypeScript SDK: {exc}")
        return False


def main():
    """Main entry point for SDK generation."""
    parser = argparse.ArgumentParser(description="Generate SDKs from Medi OS OpenAPI specification")
    parser.add_argument(
        "--language",
        type=str,
        default="typescript",
        choices=[
            "typescript",
            "python",
            "javascript",
            "go",
            "java",
            "php",
            "ruby",
            "swift",
            "kotlin",
        ],
        help="Target language for SDK generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sdk",
        help="Output directory for generated SDK",
    )
    parser.add_argument(
        "--schema-output",
        type=str,
        default="openapi-schema.json",
        help="Output path for OpenAPI schema JSON",
    )
    parser.add_argument(
        "--package-name",
        type=str,
        default="medios-api-client",
        help="Package name for generated SDK",
    )
    parser.add_argument(
        "--package-version",
        type=str,
        default="2.0.0",
        help="Package version for generated SDK",
    )
    parser.add_argument(
        "--use-simple-typescript",
        action="store_true",
        help="Use openapi-typescript-codegen for TypeScript (lighter weight)",
    )
    parser.add_argument(
        "--export-schema-only",
        action="store_true",
        help="Only export OpenAPI schema, don't generate SDK",
    )

    args = parser.parse_args()

    # Export OpenAPI schema
    schema_path = Path(args.schema_output)
    schema = export_openapi_schema(schema_path)

    if args.export_schema_only:
        print("✅ OpenAPI schema exported. Use it with openapi-generator or other tools.")
        print(f"   Schema: {schema_path}")
        print(f"   API Version: {schema.get('info', {}).get('version', 'unknown')}")
        return 0

    # Generate SDK
    output_dir = Path(args.output) / args.language
    success = False

    if args.language == "typescript" and args.use_simple_typescript:
        success = generate_typescript_sdk_simple(schema_path, output_dir)
    else:
        success = generate_sdk_with_openapi_generator(
            schema_path,
            args.language,
            output_dir,
            args.package_name,
            args.package_version,
        )

    if success:
        print(f"\n✅ SDK generation completed!")
        print(f"   Language: {args.language}")
        print(f"   Output: {output_dir}")
        print(f"\n📝 Next steps:")
        print(f"   1. Navigate to {output_dir}")
        print(f"   2. Review the generated code")
        print(f"   3. Install dependencies and build the SDK")
        return 0
    else:
        print(f"\n❌ SDK generation failed. See errors above.")
        print(f"\n💡 Alternative: Use the exported OpenAPI schema with your preferred tool:")
        print(f"   Schema: {schema_path}")
        print(f"   Tools: openapi-generator, swagger-codegen, openapi-typescript-codegen, etc.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
