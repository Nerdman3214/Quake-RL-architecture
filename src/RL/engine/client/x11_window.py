"""Direct X11 window capture and RGB policy preprocessing."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class X11Window:
    """One discovered X11 window."""

    window_id: int
    title: str

    def __post_init__(self) -> None:
        if self.window_id <= 0:
            raise ValueError(
                "window id must be greater than zero"
            )

    @property
    def selector(self) -> str:
        """Return the hexadecimal selector used by X11 tools."""

        return f"0x{self.window_id:x}"


class X11WindowCapture:
    """Capture a visible window directly through X11."""

    def __init__(
        self,
        *,
        window_name_pattern: str = "^Xonotic$",
        xdotool_command: str = "xdotool",
        import_command: str = "import",
    ) -> None:
        if not window_name_pattern:
            raise ValueError(
                "window name pattern must not be empty"
            )

        self.window_name_pattern = window_name_pattern
        self.xdotool_command = xdotool_command
        self.import_command = import_command

    @staticmethod
    def _command_error(
        command: list[str],
        error: subprocess.CalledProcessError,
    ) -> RuntimeError:
        stderr = error.stderr

        if isinstance(stderr, bytes):
            detail = stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()
        else:
            detail = str(stderr or "").strip()

        message = (
            f"command failed: {' '.join(command)}"
        )

        if detail:
            message += f": {detail}"

        return RuntimeError(message)

    def _run_text(
        self,
        command: list[str],
    ) -> str:
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "required command is unavailable: "
                f"{command[0]}"
            ) from error
        except subprocess.CalledProcessError as error:
            raise self._command_error(
                command,
                error,
            ) from error

        return result.stdout.strip()

    def find_window(self) -> X11Window:
        """Find the most recently listed visible matching window."""

        command = [
            self.xdotool_command,
            "search",
            "--onlyvisible",
            "--name",
            self.window_name_pattern,
        ]

        output = self._run_text(command)

        window_ids: list[int] = []

        for line in output.splitlines():
            value = line.strip()

            if not value:
                continue

            try:
                window_ids.append(int(value))
            except ValueError as error:
                raise RuntimeError(
                    "xdotool returned an invalid window id: "
                    f"{value!r}"
                ) from error

        if not window_ids:
            raise RuntimeError(
                "no visible matching X11 window was found"
            )

        window_id = window_ids[-1]

        title = self._run_text(
            [
                self.xdotool_command,
                "getwindowname",
                str(window_id),
            ]
        )

        return X11Window(
            window_id=window_id,
            title=title,
        )

    def capture_rgb(
        self,
        window: X11Window,
    ) -> np.ndarray:
        """Capture a window directly and return H×W×3 uint8 RGB."""

        if not isinstance(window, X11Window):
            raise TypeError(
                "window must be an X11Window instance"
            )

        command = [
            self.import_command,
            "-silent",
            "-window",
            window.selector,
            "png:-",
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "required command is unavailable: "
                f"{command[0]}"
            ) from error
        except subprocess.CalledProcessError as error:
            raise self._command_error(
                command,
                error,
            ) from error

        if not result.stdout:
            raise RuntimeError(
                "direct X11 capture returned no image data"
            )

        try:
            with Image.open(
                BytesIO(result.stdout)
            ) as image:
                rgb = np.asarray(
                    image.convert("RGB"),
                    dtype=np.uint8,
                ).copy()
        except Exception as error:
            raise RuntimeError(
                "direct X11 capture returned an invalid image"
            ) from error

        if (
            rgb.ndim != 3
            or rgb.shape[2] != 3
            or rgb.size == 0
        ):
            raise RuntimeError(
                "captured image is not a valid RGB frame"
            )

        return rgb


def preprocess_rgb_frame(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert H×W×3 uint8 RGB to normalized 3×H×W."""

    if width <= 0:
        raise ValueError(
            "policy frame width must be greater than zero"
        )

    if height <= 0:
        raise ValueError(
            "policy frame height must be greater than zero"
        )

    if not isinstance(frame, np.ndarray):
        raise TypeError(
            "captured frame must be a NumPy array"
        )

    if (
        frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError(
            "captured frame must have shape H×W×3"
        )

    if frame.dtype != np.uint8:
        raise TypeError(
            "captured frame must use uint8 RGB values"
        )

    image = Image.fromarray(frame)

    resized = image.resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    resized_hwc = np.asarray(
        resized,
        dtype=np.uint8,
    )

    resized_chw = np.transpose(
        resized_hwc,
        (2, 0, 1),
    )

    normalized = (
        resized_chw.astype(np.float32) / 255.0
    )

    return np.ascontiguousarray(normalized)
