"""
AI Scribe model orchestration for transcription, entity extraction, and note generation.

This hardened implementation removes heavyweight local dependencies and relies on:
  - Whisper (openai-whisper) for speech-to-text, cached locally via `model_manager`.
  - Lightweight keyword/regex entity extraction.
  - Gemini 1.5 Pro for SOAP note generation (with graceful fallback).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]

from .config import get_settings
from .model_manager import get_whisper_model

logger = logging.getLogger(__name__)

VALID_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}

SYMPTOM_KEYWORDS = {
    "chest pain",
    "shortness of breath",
    "dyspnea",
    "fever",
    "cough",
    "fatigue",
    "headache",
    "nausea",
    "vomiting",
    "dizziness",
    "palpitations",
}

MEDICATION_KEYWORDS = {
    "aspirin",
    "acetaminophen",
    "ibuprofen",
    "nitroglycerin",
    "metformin",
    "insulin",
    "lisinopril",
    "atorvastatin",
    "albuterol",
    "amoxicillin",
}

DIAGNOSIS_KEYWORDS = {
    "hypertension",
    "diabetes",
    "pneumonia",
    "asthma",
    "myocardial infarction",
    "stroke",
    "infection",
    "covid",
    "heart failure",
}

VITAL_PATTERNS = {
    "blood_pressure": re.compile(r"(?:bp|blood pressure)[^\d]*(\d{2,3}\/\d{2,3})", re.IGNORECASE),
    "heart_rate": re.compile(r"(?:hr|heart rate)[^\d]*(\d{2,3})", re.IGNORECASE),
    "temperature": re.compile(r"(?:temp|temperature)[^\d]*(\d{2}\.\d|\d{2})", re.IGNORECASE),
    "respiratory_rate": re.compile(r"(?:rr|respiratory rate)[^\d]*(\d{2})", re.IGNORECASE),
    "oxygen_saturation": re.compile(r"(?:sat|oxygen|spo2)[^\d]*(\d{2})", re.IGNORECASE),
}


class AIModelsService:
    """Service wrapper that coordinates transcription, entity extraction, and note generation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._gemini_model = None
        self._gemini_enabled = False
        self._initialise_gemini()
        self._check_ffmpeg_availability()

    def _list_available_models(self) -> List[str]:
        """List available Gemini models for the API key."""
        if genai is None:
            return []
        try:
            models = genai.list_models()
            available = [m.name.replace("models/", "") for m in models if "generateContent" in m.supported_generation_methods]
            return available
        except Exception as exc:
            logger.warning(f"Failed to list available models: {exc}")
            return []

    def _initialise_gemini(self) -> None:
        if not self.settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY missing. Falling back to template note generation.")
            logger.warning("To enable Gemini, set GEMINI_API_KEY in your .env file.")
            logger.warning("Get your API key from: https://makersuite.google.com/app/apikey")
            return
        if genai is None:
            logger.warning("google-generativeai not installed. Run `pip install google-generativeai`.")
            return
        try:
            genai.configure(api_key=self.settings.gemini_api_key)

            # List available models to help with debugging
            available_models = self._list_available_models()
            if available_models:
                logger.info(
                    f"Available Gemini models: {', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}"
                )
            else:
                logger.warning("Could not list available models. This might indicate an API key issue.")

            # Try different model name formats
            model_name = self.settings.gemini_model.strip()

            # Remove "models/" prefix if present (SDK adds it internally)
            if model_name.startswith("models/"):
                model_name = model_name.replace("models/", "", 1)
                logger.info(f"Removed 'models/' prefix from GEMINI_MODEL. Using: {model_name}")

            # Try the configured model first
            logger.info(f"Initializing Gemini model: {model_name}")
            try:
                self._gemini_model = genai.GenerativeModel(model_name)
                self._gemini_enabled = True
                logger.info(f"Gemini model '{model_name}' initialized successfully")
                return
            except Exception as model_exc:
                logger.warning(f"Failed to initialize model '{model_name}': {model_exc}")

                # Try alternative models (newer versions)
                alternative_models = [
                    "gemini-2.0-flash-exp",  # Latest experimental
                    "gemini-2.0-flash-thinking-exp",
                    "gemini-1.5-pro-latest",  # Latest stable
                    "gemini-1.5-flash-latest",
                    "gemini-pro",  # Fallback
                ]

                # Also try models from available list
                if available_models:
                    # Prefer models with "flash" or "pro" in the name
                    preferred = [m for m in available_models if any(keyword in m.lower() for keyword in ["flash", "pro"])]
                    if preferred:
                        alternative_models = preferred[:3] + alternative_models

                for alt_model in alternative_models:
                    try:
                        logger.info(f"Trying alternative model: {alt_model}")
                        self._gemini_model = genai.GenerativeModel(alt_model)
                        self._gemini_enabled = True
                        logger.info(f"Successfully initialized alternative Gemini model: {alt_model}")
                        # Update settings to use the working model
                        self.settings.gemini_model = alt_model
                        return
                    except Exception:
                        continue

                # If all models fail, disable Gemini
                raise model_exc

        except Exception as exc:  # pragma: no cover - external API
            error_msg = str(exc)
            logger.error(f"Failed to initialise Gemini: {exc}")

            # Provide helpful error messages
            if "API key" in error_msg.lower() or "invalid" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                logger.error("Gemini API key appears to be invalid or unauthorized.")
                logger.error("Please verify your GEMINI_API_KEY in the .env file.")
                logger.error("Get your API key from: https://makersuite.google.com/app/apikey")
            elif "404" in error_msg or "not found" in error_msg.lower():
                logger.error("Gemini model not found. The model may have been deprecated.")
                logger.error(f"Configured model: {self.settings.gemini_model}")
                logger.error("Try updating GEMINI_MODEL in .env to a newer model like 'gemini-2.0-flash-exp'")
                if available_models:
                    logger.error(f"Available models: {', '.join(available_models[:10])}")
            else:
                logger.error("Unknown error initializing Gemini. Check your API key and internet connection.")

            logger.warning("Falling back to template-based note generation. Gemini will be disabled.")
            self._gemini_model = None
            self._gemini_enabled = False

    def _check_ffmpeg_availability(self) -> None:
        """Check if ffmpeg is available and attempt to install it if not found."""
        import os
        import platform
        import shutil
        import subprocess
        from pathlib import Path

        # First, check for local installation in project directory (fastest check)
        project_root = Path(__file__).parent.parent.parent
        local_ffmpeg = project_root / "ffmpeg" / "bin" / "ffmpeg.exe"
        if local_ffmpeg.exists():
            ffmpeg_dir = str(local_ffmpeg.parent)
            current_path = os.environ.get("PATH", "")
            if ffmpeg_dir not in current_path:
                os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{current_path}"
            logger.info(f"Using locally installed ffmpeg at: {local_ffmpeg}")
            return

        # Try to find ffmpeg in system PATH
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            logger.info(f"ffmpeg is available at: {ffmpeg_path}")
            return

        # Check common installation locations on Windows
        if not ffmpeg_path:
            common_paths = [
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
                os.path.join(
                    os.environ.get("LOCALAPPDATA", ""),
                    "Microsoft",
                    "WindowsApps",
                    "ffmpeg.exe",
                ),
                os.path.join(os.environ.get("ProgramFiles", ""), "ffmpeg", "bin", "ffmpeg.exe"),
            ]
            for path in common_paths:
                if path and os.path.exists(path):
                    ffmpeg_path = path
                    # Add ffmpeg directory to PATH for this process
                    ffmpeg_dir = os.path.dirname(path)
                    current_path = os.environ.get("PATH", "")
                    if ffmpeg_dir not in current_path:
                        os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{current_path}"
                    logger.info(f"Found ffmpeg at: {ffmpeg_path}, added to PATH")
                    return

        # Try to find ffmpeg.exe in PATH directories
        if not ffmpeg_path:
            path_dirs = os.environ.get("PATH", "").split(os.pathsep)
            for path_dir in path_dirs:
                if path_dir:
                    potential_ffmpeg = os.path.join(path_dir, "ffmpeg.exe")
                    if os.path.exists(potential_ffmpeg):
                        ffmpeg_path = potential_ffmpeg
                        logger.info(f"Found ffmpeg at: {ffmpeg_path}")
                        return

        # If still not found, log warning but don't block startup
        logger.warning(
            "ffmpeg is not found. Audio transcription will not work until ffmpeg is installed. "
            "To install manually: https://ffmpeg.org/download.html or run 'winget install Gyan.FFmpeg'"
        )

    def _try_install_ffmpeg(self) -> None:
        """Attempt to install ffmpeg automatically using available package managers."""
        import os
        import platform
        import shutil
        import subprocess

        system = platform.system().lower()
        installation_successful = False

        if system == "windows":
            # Try winget first (Windows 10/11)
            logger.info("Attempting to install ffmpeg using winget...")
            try:
                result = subprocess.run(
                    [
                        "winget",
                        "install",
                        "Gyan.FFmpeg",
                        "--silent",
                        "--accept-package-agreements",
                        "--accept-source-agreements",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                logger.info(f"winget result: returncode={result.returncode}")
                if result.stdout:
                    logger.info(f"winget stdout: {result.stdout[:200]}")
                if result.stderr:
                    logger.info(f"winget stderr: {result.stderr[:200]}")
                if result.returncode == 0:
                    logger.info("Successfully installed ffmpeg using winget")
                    # Refresh PATH - may need to restart to pick up system PATH changes
                    ffmpeg_path = shutil.which("ffmpeg")
                    if ffmpeg_path:
                        logger.info(f"ffmpeg is now available at: {ffmpeg_path}")
                        installation_successful = True
                    else:
                        logger.warning("ffmpeg installed but not yet in PATH. May need to restart the server.")
                        installation_successful = True  # Consider it successful even if not in PATH yet
                else:
                    logger.warning(f"winget installation returned non-zero exit code: {result.returncode}")
            except FileNotFoundError:
                logger.info("winget not found on system - skipping")
            except subprocess.TimeoutExpired:
                logger.warning("winget installation timed out after 120 seconds")
            except Exception as e:
                logger.warning(f"winget installation failed: {e}")

            # Try chocolatey if winget failed
            if not installation_successful:
                logger.info("Attempting to install ffmpeg using chocolatey...")
                try:
                    result = subprocess.run(
                        ["choco", "install", "ffmpeg", "-y"],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    logger.info(f"chocolatey result: returncode={result.returncode}")
                    if result.stdout:
                        logger.info(f"chocolatey stdout: {result.stdout[:200]}")
                    if result.stderr:
                        logger.info(f"chocolatey stderr: {result.stderr[:200]}")
                    if result.returncode == 0:
                        logger.info("Successfully installed ffmpeg using chocolatey")
                        # Refresh PATH
                        ffmpeg_path = shutil.which("ffmpeg")
                        if ffmpeg_path:
                            logger.info(f"ffmpeg is now available at: {ffmpeg_path}")
                            installation_successful = True
                        else:
                            logger.warning("ffmpeg installed but not yet in PATH. May need to restart the server.")
                            installation_successful = True
                except FileNotFoundError:
                    logger.info("chocolatey not found on system - skipping")
                except subprocess.TimeoutExpired:
                    logger.warning("chocolatey installation timed out after 120 seconds")
                except Exception as e:
                    logger.warning(f"chocolatey installation failed: {e}")

            # Try downloading and extracting ffmpeg to a local directory if package managers failed
            if not installation_successful:
                logger.info("Attempting to download ffmpeg binaries directly...")
                try:
                    self._download_ffmpeg_windows()
                    installation_successful = True
                except Exception as e:
                    logger.error(f"Failed to download ffmpeg: {e}", exc_info=True)

        elif system == "linux":
            # Try apt (Debian/Ubuntu)
            logger.info("Attempting to install ffmpeg using apt...")
            try:
                result = subprocess.run(
                    ["sudo", "apt-get", "update", "-qq"],
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
                result = subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "ffmpeg"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode == 0:
                    logger.info("Successfully installed ffmpeg using apt")
                    installation_successful = True
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                logger.debug(f"apt installation failed: {e}")

        elif system == "darwin":  # macOS
            # Try homebrew
            logger.info("Attempting to install ffmpeg using homebrew...")
            try:
                result = subprocess.run(
                    ["brew", "install", "ffmpeg"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                if result.returncode == 0:
                    logger.info("Successfully installed ffmpeg using homebrew")
                    installation_successful = True
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                logger.debug(f"homebrew installation failed: {e}")

        if not installation_successful:
            logger.warning(
                "Could not automatically install ffmpeg using any method. "
                "Please install ffmpeg manually and ensure it's in your PATH. "
                "For Windows: https://ffmpeg.org/download.html or use 'winget install Gyan.FFmpeg'"
            )

    def _download_ffmpeg_windows(self) -> None:
        """Download and extract ffmpeg for Windows to a local directory."""
        import os
        import shutil
        import tempfile
        import urllib.request
        import zipfile
        from pathlib import Path

        # Create local ffmpeg directory in the project
        project_root = Path(__file__).parent.parent.parent
        ffmpeg_dir = project_root / "ffmpeg"
        ffmpeg_bin_dir = ffmpeg_dir / "bin"
        ffmpeg_exe = ffmpeg_bin_dir / "ffmpeg.exe"

        # If already downloaded, use it
        if ffmpeg_exe.exists():
            logger.info(f"Using existing ffmpeg at: {ffmpeg_exe}")
            ffmpeg_bin_dir_str = str(ffmpeg_bin_dir)
            current_path = os.environ.get("PATH", "")
            if ffmpeg_bin_dir_str not in current_path:
                os.environ["PATH"] = f"{ffmpeg_bin_dir_str}{os.pathsep}{current_path}"
            return

        logger.info("Downloading ffmpeg for Windows...")
        # Try multiple download sources in case one is unavailable
        # Using the essentials build which is smaller (~70MB)
        download_urls = [
            "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        ]
        download_url = download_urls[0]  # Start with the first one

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "ffmpeg.zip")

                # Download with progress and retry logic
                logger.info(f"Downloading ffmpeg from {download_url} (this may take a few minutes, ~70MB)...")
                download_successful = False
                last_error = None

                for url in download_urls:
                    try:
                        logger.info(f"Attempting download from: {url}")
                        # Create a request with a user agent to avoid blocking
                        req = urllib.request.Request(url)
                        req.add_header(
                            "User-Agent",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        )

                        with urllib.request.urlopen(req, timeout=300) as response:
                            total_size = int(response.headers.get("Content-Length", 0))
                            if total_size > 0:
                                logger.info(f"Download size: {total_size / (1024*1024):.1f} MB")

                            with open(zip_path, "wb") as out_file:
                                downloaded = 0
                                while True:
                                    chunk = response.read(8192)
                                    if not chunk:
                                        break
                                    out_file.write(chunk)
                                    downloaded += len(chunk)
                                    if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                                        logger.info(
                                            f"Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB"
                                        )

                        logger.info("Download complete")
                        download_successful = True
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Failed to download from {url}: {e}")
                        continue

                if not download_successful:
                    raise Exception(f"Failed to download ffmpeg from all sources. Last error: {last_error}") from last_error

                # Verify zip file
                if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
                    raise Exception(f"Downloaded file is too small or doesn't exist: {zip_path}")

                # Extract
                logger.info("Extracting ffmpeg...")
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        # Get all members
                        members = zip_ref.namelist()
                        if not members:
                            raise Exception("Zip file is empty")

                        # Find the root directory - look for bin/ffmpeg.exe
                        root_dir = None
                        for member in members:
                            # Normalize path separators
                            normalized = member.replace("\\", "/")
                            parts = normalized.split("/")
                            # Look for pattern like "ffmpeg-X.X.X-essentials/bin/ffmpeg.exe"
                            if len(parts) >= 3 and "bin" in parts and "ffmpeg.exe" in parts:
                                root_dir = parts[0]
                                break

                        if not root_dir:
                            # Fallback: use the first directory
                            for member in members:
                                parts = member.replace("\\", "/").split("/")
                                if len(parts) > 1 and parts[0]:
                                    root_dir = parts[0]
                                    break

                        if not root_dir:
                            raise Exception(f"Could not determine root directory. Zip contents: {members[:10]}")

                        logger.info(f"Found root directory in zip: {root_dir}")

                        # Extract the entire zip
                        extract_dir = os.path.join(tmpdir, "extracted")
                        zip_ref.extractall(extract_dir)
                        logger.info(f"Extracted to: {extract_dir}")

                        # Find the bin directory
                        possible_bin_paths = [
                            os.path.join(extract_dir, root_dir, "bin"),
                            os.path.join(extract_dir, "bin"),
                        ]

                        src_bin = None
                        for possible_path in possible_bin_paths:
                            if os.path.exists(possible_path) and os.path.exists(os.path.join(possible_path, "ffmpeg.exe")):
                                src_bin = possible_path
                                break

                        if not src_bin:
                            # Try to find it by walking the directory
                            for root, dirs, files in os.walk(extract_dir):
                                if "ffmpeg.exe" in files and "bin" in root:
                                    src_bin = root
                                    break

                        if not src_bin or not os.path.exists(os.path.join(src_bin, "ffmpeg.exe")):
                            raise Exception(f"bin directory with ffmpeg.exe not found. Searched: {possible_bin_paths}")

                        logger.info(f"Found ffmpeg.exe in: {src_bin}")

                        # Copy to project directory
                        ffmpeg_dir.mkdir(parents=True, exist_ok=True)
                        if ffmpeg_bin_dir.exists():
                            shutil.rmtree(ffmpeg_bin_dir)
                        shutil.copytree(src_bin, ffmpeg_bin_dir)
                        logger.info(f"Copied ffmpeg to: {ffmpeg_bin_dir}")

                except zipfile.BadZipFile as e:
                    raise Exception(f"Invalid zip file: {e}") from e
                except Exception as e:
                    raise Exception(f"Failed to extract ffmpeg: {e}") from e

                # Verify installation
                if not ffmpeg_exe.exists():
                    raise Exception(f"ffmpeg.exe not found at expected location: {ffmpeg_exe}")

                # Add to PATH
                logger.info(f"Successfully downloaded ffmpeg to: {ffmpeg_exe}")
                ffmpeg_bin_dir_str = str(ffmpeg_bin_dir)
                current_path = os.environ.get("PATH", "")
                if ffmpeg_bin_dir_str not in current_path:
                    os.environ["PATH"] = f"{ffmpeg_bin_dir_str}{os.pathsep}{current_path}"
                logger.info("ffmpeg has been added to PATH for this session")

        except Exception as e:
            error_msg = f"Failed to download ffmpeg: {e}"
            logger.error(error_msg, exc_info=True)
            logger.info("You can manually install ffmpeg by running: winget install Gyan.FFmpeg")
            raise Exception(error_msg) from e

    async def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Transcribe audio using Whisper. Returns structured result with success/error/is_stub flags."""
        if not os.path.exists(audio_file_path):
            return {
                "success": False,
                "transcription": "",
                "confidence": 0.0,
                "is_stub": False,
                "error": f"Audio file not found: {audio_file_path}",
            }

        if os.path.getsize(audio_file_path) < 2000:  # <2KB likely empty/corrupt
            return {
                "success": False,
                "transcription": "",
                "confidence": 0.0,
                "is_stub": False,
                "error": "Audio file too small; ensure the recording contains speech.",
            }

        extension = os.path.splitext(audio_file_path)[1].lower()
        if extension not in VALID_AUDIO_EXTENSIONS:
            return {
                "success": False,
                "transcription": "",
                "confidence": 0.0,
                "is_stub": False,
                "error": f"Unsupported audio format '{extension}'. Supported: {', '.join(sorted(VALID_AUDIO_EXTENSIONS))}",
            }

        try:
            model = get_whisper_model()
        except RuntimeError as exc:
            warning = f"Whisper unavailable: {exc}"
            logger.error(warning)
            return {
                "success": False,
                "transcription": "",
                "confidence": 0.0,
                "is_stub": True,
                "error": warning,
            }

        def _run_transcription() -> Dict[str, Any]:
            import os
            import shutil
            from pathlib import Path

            # Try to find ffmpeg in common locations or PATH
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                # Check common installation locations on Windows
                common_paths = [
                    r"C:\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
                    os.path.join(
                        os.environ.get("LOCALAPPDATA", ""),
                        "Microsoft",
                        "WindowsApps",
                        "ffmpeg.exe",
                    ),
                    os.path.join(
                        os.environ.get("ProgramFiles", ""),
                        "ffmpeg",
                        "bin",
                        "ffmpeg.exe",
                    ),
                ]
                for path in common_paths:
                    if path and os.path.exists(path):
                        ffmpeg_path = path
                        # Add ffmpeg directory to PATH for this process
                        ffmpeg_dir = os.path.dirname(path)
                        current_path = os.environ.get("PATH", "")
                        if ffmpeg_dir not in current_path:
                            os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{current_path}"
                        logger.info(f"Found ffmpeg at: {ffmpeg_path}, added to PATH")
                        break

            if not ffmpeg_path:
                # Try to find ffmpeg.exe in PATH directories
                path_dirs = os.environ.get("PATH", "").split(os.pathsep)
                for path_dir in path_dirs:
                    if path_dir:
                        potential_ffmpeg = os.path.join(path_dir, "ffmpeg.exe")
                        if os.path.exists(potential_ffmpeg):
                            ffmpeg_path = potential_ffmpeg
                            logger.info(f"Found ffmpeg at: {ffmpeg_path}")
                            break

            # Ensure the audio file path is absolute
            audio_path = Path(audio_file_path).resolve()
            if not audio_path.exists():
                return {
                    "success": False,
                    "transcription": "",
                    "confidence": 0.0,
                    "is_stub": True,
                    "error": f"Audio file not found: {audio_path}",
                }

            # Verify audio file exists with absolute path first
            if not audio_path.exists():
                logger.error(f"Audio file does not exist: {audio_path}")
                return {
                    "success": False,
                    "transcription": "",
                    "confidence": 0.0,
                    "is_stub": True,
                    "error": f"Audio file not found: {audio_path}",
                }

            # Ensure ffmpeg is in PATH before calling Whisper
            # Re-check PATH in case it was updated during initialization
            if not ffmpeg_path:
                ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                ffmpeg_dir = os.path.dirname(ffmpeg_path)
                current_path = os.environ.get("PATH", "")
                if ffmpeg_dir and ffmpeg_dir not in current_path:
                    os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{current_path}"
                    logger.info(f"Added ffmpeg directory to PATH: {ffmpeg_dir}")
                # Verify it's accessible
                if not shutil.which("ffmpeg"):
                    logger.error(f"ffmpeg at {ffmpeg_path} is not accessible via PATH")
            else:
                logger.error("ffmpeg is not found in PATH or common locations")

            # Log the paths for debugging
            logger.info(
                f"Transcribing audio: {audio_path} (absolute path, exists: {audio_path.exists()}, size: {audio_path.stat().st_size if audio_path.exists() else 0} bytes)"
            )
            final_ffmpeg_check = shutil.which("ffmpeg")
            if final_ffmpeg_check:
                logger.info(f"ffmpeg is available at: {final_ffmpeg_check}")
            else:
                logger.error("ffmpeg is NOT available - transcription will fail")
                return {
                    "success": False,
                    "transcription": "",
                    "confidence": 0.0,
                    "is_stub": True,
                    "error": "ffmpeg is not installed or not in PATH. Whisper requires ffmpeg to process audio files. Please install ffmpeg and add it to your system PATH, then restart the server.",
                }

            try:
                result = model.transcribe(str(audio_path), language="en")
                text = result.get("text", "").strip()
                segments = result.get("segments", [])
                if segments:
                    avg_logprob = sum(seg.get("avg_logprob", -5.0) for seg in segments) / len(segments)
                    confidence = max(min(1.0 + avg_logprob / 5.0, 0.99), 0.0)
                else:
                    confidence = 0.8
                return {
                    "success": True,
                    "transcription": text,
                    "confidence": confidence,
                    "is_stub": False,
                    "error": None,
                }
            except FileNotFoundError as exc:
                # This usually means ffmpeg is not found
                error_msg = str(exc)
                logger.error(f"FileNotFoundError during transcription: {exc}")
                logger.error(f"Current PATH: {os.environ.get('PATH', '')[:500]}")
                logger.error(f"ffmpeg which: {shutil.which('ffmpeg')}")
                if (
                    "winerror 2" in error_msg.lower()
                    or "cannot find the file" in error_msg.lower()
                    or "ffmpeg" in error_msg.lower()
                ):
                    return {
                        "success": False,
                        "transcription": "",
                        "confidence": 0.0,
                        "is_stub": True,
                        "error": f"ffmpeg is not available. Whisper requires ffmpeg to process audio files. Please install ffmpeg and ensure it's accessible. Current PATH: {os.environ.get('PATH', '')[:200]}... Original error: {exc}",
                    }
                raise
            except Exception as exc:
                # Catch other potential errors and provide context
                error_msg = str(exc)
                logger.error(f"Exception during transcription: {exc}")
                if "ffmpeg" in error_msg.lower():
                    return {
                        "success": False,
                        "transcription": "",
                        "confidence": 0.0,
                        "is_stub": True,
                        "error": f"ffmpeg error: {error_msg}. Please ensure ffmpeg is installed and working correctly.",
                    }
                raise

        try:
            return await asyncio.to_thread(_run_transcription)
        except Exception as exc:  # pragma: no cover - whisper runtime issues
            logger.exception("Whisper transcription failed: %s", exc)
            return {
                "success": False,
                "transcription": "",
                "confidence": 0.0,
                "is_stub": False,
                "error": f"Transcription failed: {exc}",
            }

    async def extract_entities(self, transcription: str) -> Dict[str, Any]:
        """Lightweight pattern-based entity extraction with improved matching."""
        text = transcription.strip()
        if not text:
            return {
                "success": True,
                "entities": self._empty_entities(),
                "confidence": 0.0,
                "is_stub": False,
                "warning": "No transcription text supplied.",
            }

        lowered = text.lower()

        # Extract symptoms - improved matching with multiple strategies
        symptoms = set()

        # Strategy 1: Direct keyword matching (case-insensitive)
        for keyword in SYMPTOM_KEYWORDS:
            keyword_lower = keyword.lower()
            if keyword_lower in lowered:
                symptoms.add(keyword)

        # Strategy 2: Pattern-based extraction for common symptom phrases
        symptom_patterns = {
            "chest pain": [
                r"pain\s+(?:in|at|around)\s+(?:my\s+)?chest",
                r"chest\s+(?:pain|discomfort|tightness|pressure)",
                r"pain\s+at\s+my\s+chest",
            ],
            "shortness of breath": [
                r"shortness\s+of\s+breath",
                r"difficulty\s+breathing",
                r"trouble\s+breathing",
                r"can't\s+breathe",
                r"hard\s+to\s+breathe",
            ],
            "dyspnea": [
                r"dyspnea",
                r"short\s+of\s+breath",
            ],
        }

        for symptom_name, patterns in symptom_patterns.items():
            for pattern in patterns:
                if re.search(pattern, lowered, re.IGNORECASE):
                    symptoms.add(symptom_name)
                    break

        # Strategy 3: Extract from patient statements
        patient_symptom_patterns = [
            r"experiencing\s+(?:some\s+)?(?:pain|discomfort|symptoms?)\s+(?:in|at|with)\s+([^.,;!?]+)",
            r"feeling\s+([^.,;!?]+?)(?:\.|,|;|for|when)",
            r"having\s+([^.,;!?]+?)(?:\.|,|;|for|when)",
            r"(?:have|got|am\s+having)\s+([^.,;!?]+?)(?:\s+pain|\s+discomfort|\.|,|;|for)",
        ]

        for pattern in patient_symptom_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match_lower = match.strip().lower()
                # Check if it contains known symptom keywords
                for symptom in SYMPTOM_KEYWORDS:
                    if symptom.lower() in match_lower:
                        symptoms.add(symptom)
                    # Also check if match contains words that suggest symptoms
                    elif any(word in match_lower for word in ["chest", "breath", "pain", "discomfort"]):
                        # Try to map to known symptoms
                        if "chest" in match_lower and "pain" in match_lower:
                            symptoms.add("chest pain")
                        elif "breath" in match_lower or "breathing" in match_lower:
                            symptoms.add("shortness of breath")

        # Extract medications
        medications = sorted({kw for kw in MEDICATION_KEYWORDS if kw.lower() in lowered})

        # Extract diagnoses
        diagnoses = sorted({kw for kw in DIAGNOSIS_KEYWORDS if kw.lower() in lowered})

        # Extract vitals with improved patterns
        vitals = {}
        for key, pattern in VITAL_PATTERNS.items():
            match = pattern.search(text)
            if match:
                vitals[key] = match.group(1)

        # Extract additional context from transcription
        # Look for duration mentions (e.g., "for a few weeks", "for 3 days", "bothering me for")
        duration_patterns = [
            r"for\s+(?:a\s+)?(?:few\s+)?(?:several\s+)?(?:weeks?|days?|months?|hours?)",
            r"bothering\s+(?:me\s+)?for\s+([^.,;!?]+)",
            r"(?:weeks?|days?|months?)\s+(?:now|ago)",
        ]
        duration = None
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                duration = match.group(0)
                break

        # Look for activity triggers (e.g., "when climbing stairs", "during exercise")
        activity_patterns = [
            r"when\s+[^.,;!?]*?(?:climbing|walking|exercising|physical|push|strenuous|fast)",
            r"during\s+[^.,;!?]*?(?:climbing|walking|exercising|physical|activity)",
            r"(?:climbing|walking)\s+[^.,;!?]*?(?:stairs|fast)",
        ]
        activity_trigger = None
        for pattern in activity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                activity_trigger = "; ".join(matches[:2])
                break

        # Extract recommended tests/procedures
        test_keywords = [
            "ecg",
            "ekg",
            "electrocardiogram",
            "x-ray",
            "xray",
            "ct scan",
            "mri",
            "blood test",
            "lab test",
            "laboratory",
        ]
        tests_mentioned = []
        for kw in test_keywords:
            if kw in lowered:
                tests_mentioned.append(kw)

        entities = {
            "symptoms": sorted(symptoms),
            "medications": medications,
            "diagnoses": diagnoses,
            "vitals": vitals,
            "duration": duration,
            "activity_trigger": activity_trigger,
            "tests_mentioned": tests_mentioned,
        }

        # Log extracted entities for debugging
        logger.debug(
            f"Extracted entities: symptoms={sorted(symptoms)}, duration={duration}, activity_trigger={activity_trigger}, tests={tests_mentioned}"
        )

        return {
            "success": True,
            "entities": entities,
            "confidence": 0.65 if any(entities.values()) else 0.0,
            "is_stub": False,
            "warning": None,
        }

    async def generate_note(self, transcription: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SOAP note via Gemini (if available) with rule-based fallback."""
        transcription = transcription.strip()
        if not transcription:
            return {
                "success": True,
                "generated_note": "No speech detected in audio input. Please provide a clear recording.",
                "confidence": 0.0,
                "is_stub": True,
                "model": "template",
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "cost_cents": 0.0,
                "warning": "Empty transcription supplied.",
            }

        if self._gemini_enabled and self._gemini_model is not None:
            prompt = self._build_gemini_prompt(transcription, entities)

            def _run_gemini() -> Dict[str, Any]:
                response = self._gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.settings.gemini_temperature,
                        "max_output_tokens": self.settings.gemini_max_tokens,
                    },
                )
                text = (response.text or "").strip()
                usage = getattr(response, "usage_metadata", None)
                tokens_prompt = getattr(usage, "prompt_token_count", 0) if usage else 0
                tokens_completion = getattr(usage, "candidates_token_count", 0) if usage else 0
                return {
                    "success": True,
                    "generated_note": text,
                    "confidence": 0.9,
                    "is_stub": False,
                    "model": self.settings.gemini_model,
                    "tokens_prompt": tokens_prompt,
                    "tokens_completion": tokens_completion,
                    "cost_cents": 0.0,
                    "warning": None,
                }

            try:
                return await asyncio.to_thread(_run_gemini)
            except Exception as exc:  # pragma: no cover - external API
                error_msg = str(exc)
                logger.warning(f"Gemini generation failed: {error_msg}")

                # If it's a 404 or model not found error, try to find and use an available model
                if "404" in error_msg or "not found" in error_msg.lower() or "not supported" in error_msg.lower():
                    logger.error(f"Gemini model '{self.settings.gemini_model}' is not available (deprecated or invalid).")

                    # Try to find and use an available model
                    available_models = self._list_available_models()
                    if available_models:
                        logger.info(f"Attempting to use an available model from: {available_models[:3]}")
                        for alt_model in available_models[:3]:
                            try:
                                logger.info(f"Trying model: {alt_model}")
                                self._gemini_model = genai.GenerativeModel(alt_model)
                                self.settings.gemini_model = alt_model
                                logger.info(f"Successfully switched to model: {alt_model}")
                                # Retry the generation with the new model
                                try:
                                    # Rebuild prompt with new model context
                                    prompt = self._build_gemini_prompt(transcription, entities)

                                    def _run_gemini_retry() -> Dict[str, Any]:
                                        response = self._gemini_model.generate_content(
                                            prompt,
                                            generation_config={
                                                "temperature": self.settings.gemini_temperature,
                                                "max_output_tokens": self.settings.gemini_max_tokens,
                                            },
                                        )
                                        text = (response.text or "").strip()
                                        usage = getattr(response, "usage_metadata", None)
                                        tokens_prompt = getattr(usage, "prompt_token_count", 0) if usage else 0
                                        tokens_completion = getattr(usage, "candidates_token_count", 0) if usage else 0
                                        return {
                                            "success": True,
                                            "generated_note": text,
                                            "confidence": 0.9,
                                            "is_stub": False,
                                            "model": alt_model,
                                            "tokens_prompt": tokens_prompt,
                                            "tokens_completion": tokens_completion,
                                            "cost_cents": 0.0,
                                            "warning": None,
                                        }

                                    return await asyncio.to_thread(_run_gemini_retry)
                                except Exception as retry_exc:
                                    logger.warning(f"Retry with {alt_model} also failed: {retry_exc}")
                                    continue
                            except Exception:
                                continue

                    # If we couldn't find a working model, disable Gemini
                    logger.error("Could not find a working Gemini model. Disabling Gemini and using template notes.")
                    self._gemini_enabled = False
                    self._gemini_model = None
                    warning = "AI model unavailable (model deprecated) - using template-based extraction"
                elif "429" in error_msg or "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                    logger.warning("Gemini API quota exceeded. Using template-based note generation.")
                    warning = "AI model quota exceeded - using template-based extraction"
                    # Don't disable Gemini for quota errors - it might work later
                    # Just fall back to template for this request
                elif (
                    "API key" in error_msg.lower()
                    or "invalid" in error_msg.lower()
                    or "401" in error_msg
                    or "403" in error_msg
                ):
                    logger.error("Gemini API key appears to be invalid or unauthorized.")
                    logger.error("Please verify your GEMINI_API_KEY in the .env file.")
                    warning = "AI model unavailable (API key invalid) - using template-based extraction"
                else:
                    warning = f"AI generation failed: {error_msg[:100]}"

                return self._template_note(transcription, entities, warning=warning)

        return self._template_note(
            transcription,
            entities,
            warning="Gemini disabled. Returning template-based note.",
        )

    async def summarize_document(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize an uploaded document with Gemini (if available) and provide key highlights."""
        text = text.strip()
        if not text:
            return {
                "success": False,
                "summary": "",
                "highlights": [],
                "confidence": 0.0,
                "is_stub": True,
                "model": "template",
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "cost_cents": 0.0,
                "warning": "Document contained no extractable text.",
            }

        metadata = metadata or {}

        if self._gemini_enabled and self._gemini_model is not None:
            prompt = self._build_document_summary_prompt(text, metadata)

            def _run_gemini() -> Dict[str, Any]:
                response = self._gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.settings.gemini_temperature,
                        "max_output_tokens": self.settings.gemini_max_tokens,
                    },
                )
                summary_text = (response.text or "").strip()
                highlights = self._extract_highlights(summary_text)
                usage = getattr(response, "usage_metadata", None)
                tokens_prompt = getattr(usage, "prompt_token_count", 0) if usage else 0
                tokens_completion = getattr(usage, "candidates_token_count", 0) if usage else 0
                return {
                    "success": True,
                    "summary": summary_text,
                    "highlights": highlights,
                    "confidence": 0.85 if summary_text else 0.5,
                    "is_stub": False,
                    "model": self.settings.gemini_model,
                    "tokens_prompt": tokens_prompt,
                    "tokens_completion": tokens_completion,
                    "cost_cents": 0.0,
                    "warning": None,
                }

            try:
                return await asyncio.to_thread(_run_gemini)
            except Exception as exc:  # pragma: no cover - external API
                logger.exception("Gemini document summary failed: %s", exc)
                return self._fallback_document_summary(text, metadata, warning=str(exc))

        return self._fallback_document_summary(
            text,
            metadata,
            warning="Gemini disabled. Returning heuristic summary.",
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _empty_entities() -> Dict[str, Any]:
        return {"symptoms": [], "medications": [], "diagnoses": [], "vitals": {}}

    def _build_gemini_prompt(self, transcription: str, entities: Dict[str, Any]) -> str:
        entity_block = json.dumps(entities, indent=2, ensure_ascii=False)
        return (
            "You are a clinical AI scribe. Generate a SOAP-formatted medical note.\n\n"
            f"Transcript:\n{transcription}\n\n"
            f"Extracted Entities:\n{entity_block}\n\n"
            "Required format:\n"
            "S: ...\n"
            "O: ...\n"
            "A: ...\n"
            "P: ...\n"
            "Keep the tone clinical, concise, and factual. Highlight risk factors and abnormal vitals."
        )

    def _template_note(
        self,
        transcription: str,
        entities: Dict[str, Any],
        warning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a template SOAP note from transcription and entities."""
        lowered_transcription = transcription.lower()

        # Extract symptoms - try entities first, then parse transcription directly
        symptoms_list = entities.get("symptoms", [])

        # If no symptoms found in entities, try to extract from transcription directly
        if not symptoms_list:
            # Look for common symptom patterns in patient speech
            symptom_patterns = [
                r"experiencing\s+(?:some\s+)?(?:pain|discomfort|symptoms?)\s+(?:in|at|with)\s+([^.,;!?]+)",
                r"feeling\s+([^.,;!?]+?)(?:\.|,|;|for)",
                r"having\s+([^.,;!?]+?)(?:\.|,|;|for)",
                r"pain\s+(?:in|at)\s+(?:my\s+)?([^.,;!?]+)",
                r"shortness\s+of\s+breath",
                r"chest\s+pain",
            ]
            extracted_symptoms = []
            for pattern in symptom_patterns:
                matches = re.findall(pattern, transcription, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = " ".join(match)
                    match_clean = match.strip()
                    if len(match_clean) > 3 and len(match_clean) < 50:
                        # Check if it's a known symptom or contains symptom keywords
                        for known_symptom in SYMPTOM_KEYWORDS:
                            if known_symptom.lower() in match_clean.lower() or match_clean.lower() in known_symptom.lower():
                                extracted_symptoms.append(known_symptom)
                                break
                        else:
                            # Add as-is if it sounds like a symptom
                            if any(
                                word in match_clean.lower()
                                for word in [
                                    "pain",
                                    "ache",
                                    "discomfort",
                                    "trouble",
                                    "difficulty",
                                ]
                            ):
                                extracted_symptoms.append(match_clean)

            symptoms_list = list(set(extracted_symptoms)) if extracted_symptoms else []

            # Fallback: look for key phrases - MORE AGGRESSIVE MATCHING
            if not symptoms_list:
                # Check for chest pain variations
                if (
                    re.search(
                        r"pain\s+(?:in|at|around)\s+(?:my\s+)?chest",
                        lowered_transcription,
                    )
                    or re.search(r"chest\s+(?:pain|discomfort)", lowered_transcription)
                    or ("chest" in lowered_transcription and "pain" in lowered_transcription)
                ):
                    symptoms_list.append("chest pain")

                # Check for shortness of breath variations
                if (
                    "shortness of breath" in lowered_transcription
                    or "difficulty breathing" in lowered_transcription
                    or "trouble breathing" in lowered_transcription
                    or (
                        "breath" in lowered_transcription
                        and ("short" in lowered_transcription or "difficulty" in lowered_transcription)
                    )
                ):
                    symptoms_list.append("shortness of breath")

        # If still no symptoms from entities, extract directly from transcription with more aggressive parsing
        if not symptoms_list:
            # Try to extract from transcription using the helper method
            extracted_subjective = self._extract_subjective_from_transcription(transcription)
            if (
                extracted_subjective
                and extracted_subjective
                != "Patient concerns and symptoms discussed during visit. See transcription for details."
            ):
                symptoms = extracted_subjective
            else:
                # Last resort: extract key phrases from first few sentences
                sentences = re.split(r"[.!?]+", transcription)
                symptom_phrases = []
                for sentence in sentences[:3]:
                    sentence_lower = sentence.lower().strip()
                    if any(
                        word in sentence_lower
                        for word in [
                            "pain",
                            "discomfort",
                            "symptom",
                            "feeling",
                            "experiencing",
                            "having",
                            "chest",
                            "breath",
                        ]
                    ):
                        # Extract the relevant part
                        if "chest" in sentence_lower:
                            symptom_phrases.append("chest pain")
                        if "breath" in sentence_lower or "breathing" in sentence_lower:
                            symptom_phrases.append("shortness of breath")
                        if "pain" in sentence_lower and "chest" not in sentence_lower:
                            # Try to extract the location
                            pain_match = re.search(
                                r"pain\s+(?:in|at)\s+([^.,;!?]+)",
                                sentence,
                                re.IGNORECASE,
                            )
                            if pain_match:
                                symptom_phrases.append(f"pain in {pain_match.group(1).strip()}")

                if symptom_phrases:
                    symptoms = ", ".join(list(set(symptom_phrases)))
                else:
                    symptoms = "Patient concerns and symptoms discussed during visit. See transcription for details."
        else:
            symptoms = ", ".join(symptoms_list)

        logger.debug(f"Template note: extracted symptoms='{symptoms}'")

        # Add duration if available (check both entities and transcription directly)
        duration = entities.get("duration")
        if not duration:
            # Try to extract duration directly from transcription - multiple patterns
            duration_patterns = [
                r"for\s+(?:a\s+)?(?:few\s+)?(?:several\s+)?(?:weeks?|days?|months?|hours?)",
                r"bothering\s+(?:me\s+)?for\s+([^.,;!?]+)",
                r"(?:weeks?|days?|months?)\s+(?:now|ago)",
            ]
            for pattern in duration_patterns:
                duration_match = re.search(pattern, transcription, re.IGNORECASE)
                if duration_match:
                    duration = duration_match.group(0)
                    break
        if duration and symptoms and "Patient concerns" not in symptoms and "Not documented" not in symptoms:
            symptoms = f"{symptoms} ({duration})"

        # Add activity trigger if available (check both entities and transcription directly)
        activity_trigger = entities.get("activity_trigger")
        if not activity_trigger:
            # Try to extract activity trigger directly from transcription - multiple patterns
            activity_patterns = [
                r"when\s+[^.,;!?]*?(?:climbing|walking|exercising|physical|push|strenuous|fast)",
                r"during\s+[^.,;!?]*?(?:climbing|walking|exercising|physical|activity)",
                r"(?:climbing|walking)\s+[^.,;!?]*?(?:stairs|fast)",
            ]
            for pattern in activity_patterns:
                activity_match = re.search(pattern, transcription, re.IGNORECASE)
                if activity_match:
                    activity_trigger = activity_match.group(0)
                    break
        if activity_trigger and symptoms and "Patient concerns" not in symptoms and "Not documented" not in symptoms:
            # Clean up activity trigger
            activity_trigger_clean = activity_trigger.strip().lower()
            symptoms = f"{symptoms}, particularly {activity_trigger_clean}"

        logger.debug(f"Template note: final symptoms='{symptoms}', duration={duration}, activity_trigger={activity_trigger}")

        # Extract medications
        meds_list = entities.get("medications", [])
        meds = ", ".join(meds_list) if meds_list else "None reported"

        # Extract diagnoses
        diagnoses_list = entities.get("diagnoses", [])
        diagnoses = ", ".join(diagnoses_list) if diagnoses_list else "To be determined"

        # Extract vitals
        vitals = entities.get("vitals", {})
        vitals_lines = []
        for label, value in vitals.items():
            pretty_label = label.replace("_", " ").title()
            vitals_lines.append(f"{pretty_label}: {value}")

        # Extract key clinical findings from transcription
        objective_findings = []
        if vitals_lines:
            objective_findings.extend(vitals_lines)

        # Look for physical examination findings
        exam_keywords = [
            "examination",
            "exam",
            "observed",
            "noted",
            "found",
            "appears",
            "presents",
            "assessment",
        ]
        exam_sentences = []
        sentences = re.split(r"[.!?]+", transcription)
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(keyword in sentence_lower for keyword in exam_keywords) and len(sentence.strip()) > 15:
                # Clean up the sentence
                sentence_clean = sentence.strip()
                if len(sentence_clean) < 200:  # Not too long
                    exam_sentences.append(sentence_clean)

        if exam_sentences:
            objective_findings.append("Clinical findings: " + "; ".join(exam_sentences[:2]))

        # Extract tests/procedures mentioned (check both entities and transcription directly)
        tests_mentioned = entities.get("tests_mentioned", [])
        # Also check transcription directly for tests if not found in entities
        if not tests_mentioned:
            test_keywords = [
                "ecg",
                "ekg",
                "electrocardiogram",
                "x-ray",
                "xray",
                "ct scan",
                "mri",
                "blood test",
                "lab test",
                "laboratory",
            ]
            for kw in test_keywords:
                if kw in lowered_transcription:
                    tests_mentioned.append(kw)

        if tests_mentioned:
            test_names = []
            for t in tests_mentioned:
                if t == "ecg" or t == "ekg":
                    test_names.append("ECG")
                elif t == "electrocardiogram":
                    test_names.append("ECG (electrocardiogram)")
                else:
                    test_names.append(t.replace("xray", "X-ray").title())
            objective_findings.append(f"Tests/Procedures discussed: {', '.join(set(test_names))}")

        # Extract from doctor's statements about recommendations
        if not objective_findings or len(objective_findings) == 0:
            doctor_recommendations = []
            doctor_patterns = [
                r"I'd like to\s+([^.,;!?]+)",
                r"I recommend\s+([^.,;!?]+)",
                r"I suggest\s+([^.,;!?]+)",
                r"order\s+(?:an\s+)?([^.,;!?]+)",
                r"perform\s+(?:an\s+)?([^.,;!?]+)",
            ]
            for pattern in doctor_patterns:
                matches = re.findall(pattern, transcription, re.IGNORECASE)
                if matches:
                    doctor_recommendations.extend([m.strip() for m in matches[:2]])

            if doctor_recommendations:
                objective_findings.append(f"Clinical recommendations discussed: {', '.join(doctor_recommendations)}")

        objective = (
            "\n".join(objective_findings)
            if objective_findings
            else "Physical examination and objective findings not fully documented in this transcript."
        )

        # Build assessment based on symptoms and context
        assessment = diagnoses
        symptoms_lower = [s.lower() for s in symptoms_list] if symptoms_list else []
        transcription_lower = lowered_transcription

        if diagnoses == "To be determined":
            # Create a preliminary assessment based on symptoms or transcription content
            if any(s in transcription_lower for s in ["chest pain", "chest discomfort"]) or any(
                s in symptoms_lower for s in ["chest pain"]
            ):
                if (
                    "shortness of breath" in transcription_lower
                    or "dyspnea" in transcription_lower
                    or any(s in symptoms_lower for s in ["shortness of breath"])
                ):
                    assessment = "Chest pain and dyspnea - rule out cardiac causes. Consider ECG and further cardiac workup."
                else:
                    assessment = "Chest pain - evaluate for cardiac and non-cardiac causes. Consider ECG and appropriate diagnostic workup."
            elif (
                "shortness of breath" in transcription_lower
                or "dyspnea" in transcription_lower
                or any(s in symptoms_lower for s in ["shortness of breath"])
            ):
                assessment = (
                    "Dyspnea - evaluate for cardiac, pulmonary, or other causes. Consider appropriate diagnostic studies."
                )
            elif "fever" in transcription_lower and "cough" in transcription_lower:
                assessment = "Respiratory symptoms with fever - rule out infection. Consider appropriate diagnostic studies."
            elif symptoms_list:
                assessment = f"Patient presents with {', '.join(symptoms_list).lower()}. Further evaluation needed to determine underlying cause."
            else:
                assessment = "Patient presentation noted. Further clinical evaluation and diagnostic workup indicated."

        # Build plan
        plan_items = []
        if tests_mentioned:
            test_names = [t.replace("ecg", "ECG").replace("ekg", "EKG").title() for t in tests_mentioned]
            plan_items.append(f"Order {', '.join(test_names)}")
        else:
            # Suggest tests based on symptoms or transcription
            if any(s in transcription_lower for s in ["chest", "heart", "cardiac"]) or any(
                s in symptoms_lower for s in ["chest pain"]
            ):
                if "ecg" in transcription_lower or "electrocardiogram" in transcription_lower:
                    plan_items.append("Order ECG (electrocardiogram) to assess cardiac function")
                else:
                    plan_items.append("Consider ECG to assess cardiac function if clinically indicated")
            if "fever" in transcription_lower:
                plan_items.append("Consider laboratory studies to rule out infection")

        if meds_list:
            plan_items.append(f"Review current medications: {meds}")
        else:
            plan_items.append("Review medications as needed")

        # Extract lifestyle recommendations from transcription
        if "lifestyle" in transcription_lower or "diet" in transcription_lower or "exercise" in transcription_lower:
            plan_items.append("Provide lifestyle modification counseling (diet, exercise)")

        plan_items.append("Follow up as clinically indicated")
        plan = "\n".join(f"- {item}" for item in plan_items)

        # Build the SOAP note
        note = f"S: {symptoms}\n\n" f"O: {objective}\n\n" f"A: {assessment}\n\n" f"P: {plan}"

        # Add warning if this is a fallback template (but make it less intrusive)
        # Only add warning in the response data, not in the note itself for cleaner output
        # The warning will be in the response metadata

        return {
            "success": True,
            "generated_note": note,
            "confidence": 0.6 if symptoms_list or vitals or tests_mentioned else 0.4,
            "is_stub": True,
            "model": "template",
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "cost_cents": 0.0,
            "warning": warning,
        }

    def _extract_subjective_from_transcription(self, transcription: str) -> str:
        """Extract subjective information directly from transcription when entities are empty."""
        # Look for patient's chief complaint or main symptoms
        patterns = [
            r"Patient\s+(?:reports|presents\s+with|complains\s+of)\s+([^.,;!?]+)",
            r"I've\s+been\s+experiencing\s+([^.,;!?]+)",
            r"I\s+(?:have|am\s+having|am\s+experiencing)\s+([^.,;!?]+)",
            r"(?:pain|discomfort|symptoms?)\s+(?:in|at|with)\s+([^.,;!?]+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, transcription, re.IGNORECASE)
            if matches:
                return matches[0].strip()

        # Fallback: look for first sentence that mentions symptoms
        sentences = re.split(r"[.!?]+", transcription)
        for sentence in sentences[:5]:  # Check first 5 sentences
            sentence_lower = sentence.lower()
            if any(
                word in sentence_lower
                for word in [
                    "pain",
                    "discomfort",
                    "symptom",
                    "feeling",
                    "experiencing",
                    "having",
                ]
            ):
                # Clean up the sentence
                sentence_clean = sentence.strip()
                if 10 < len(sentence_clean) < 150:
                    return sentence_clean

        return "Patient concerns and symptoms discussed during visit. See transcription for details."

    def _build_document_summary_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        metadata_block = json.dumps(metadata, indent=2, ensure_ascii=False)
        truncated_text = textwrap.shorten(text, width=8000, placeholder=" …")
        return (
            "You are assisting a clinical operations team by reviewing uploaded patient documents.\n"
            "Summarize the medically relevant details and extract key timeline events.\n\n"
            f"Document metadata:\n{metadata_block}\n\n"
            "Provide your response as Markdown with a short summary paragraph followed by a bullet list "
            "of clinically significant findings.\n"
            "Keep the language concise and professional. Highlight dates, vitals, labs, medications, and follow-up actions when present.\n\n"
            f"Document text:\n{truncated_text}\n"
        )

    def _fallback_document_summary(
        self,
        text: str,
        metadata: Dict[str, Any],
        warning: Optional[str] = None,
    ) -> Dict[str, Any]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        summary = " ".join(sentences[:3]).strip()
        if not summary:
            summary = textwrap.shorten(text, width=600, placeholder=" …")
        highlights = self._extract_highlights(summary)
        return {
            "success": True,
            "summary": summary,
            "highlights": highlights,
            "confidence": 0.4,
            "is_stub": True,
            "model": "heuristic",
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "cost_cents": 0.0,
            "warning": warning,
        }

    @staticmethod
    def _extract_highlights(summary_text: str) -> List[str]:
        lines = []
        for line in summary_text.splitlines():
            stripped = line.strip("-• ").strip()
            if stripped:
                lines.append(stripped)
        if not lines:
            sentences = re.split(r"(?<=[.!?])\s+", summary_text)
            lines = [sentence.strip() for sentence in sentences if sentence.strip()]
        return lines[:5]
