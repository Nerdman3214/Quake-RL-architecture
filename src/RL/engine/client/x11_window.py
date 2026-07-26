"""Direct X11 window capture and RGB policy preprocessing."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
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
    """Capture a visible X11 window as an RGB array."""

    _SUPPORTED_BACKENDS = {
        "mss",
        "imagemagick",
    }

    def __init__(
        self,
        *,
        window_name_pattern: str = "^Xonotic$",
        xdotool_command: str = "xdotool",
        import_command: str = "import",
        backend: str = "mss",
        mss_factory: (
            Callable[[], object] | None
        ) = None,
    ) -> None:
        if not window_name_pattern:
            raise ValueError(
                "window name pattern must not be empty"
            )

        if backend not in self._SUPPORTED_BACKENDS:
            raise ValueError(
                "capture backend must be one of: "
                + ", ".join(
                    sorted(self._SUPPORTED_BACKENDS)
                )
            )

        self.window_name_pattern = window_name_pattern
        self.xdotool_command = xdotool_command
        self.import_command = import_command
        self.backend = backend
        self.mss_factory = mss_factory

        self._mss_capture: object | None = None
        self._window_monitors: dict[
            int,
            dict[str, int],
        ] = {}

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

    def _window_monitor(
        self,
        window: X11Window,
    ) -> dict[str, int]:
        cached = self._window_monitors.get(
            window.window_id
        )

        if cached is not None:
            return cached

        output = self._run_text(
            [
                self.xdotool_command,
                "getwindowgeometry",
                "--shell",
                str(window.window_id),
            ]
        )

        geometry: dict[str, int] = {}

        for line in output.splitlines():
            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            if key not in {
                "X",
                "Y",
                "WIDTH",
                "HEIGHT",
            }:
                continue

            try:
                geometry[key] = int(value)
            except ValueError as error:
                raise RuntimeError(
                    "xdotool returned invalid window "
                    f"geometry: {line!r}"
                ) from error

        required = {
            "X",
            "Y",
            "WIDTH",
            "HEIGHT",
        }

        if not required.issubset(geometry):
            raise RuntimeError(
                "xdotool returned incomplete window "
                f"geometry: {geometry}"
            )

        if (
            geometry["WIDTH"] <= 0
            or geometry["HEIGHT"] <= 0
        ):
            raise RuntimeError(
                "xdotool returned nonpositive window "
                "dimensions"
            )

        monitor = {
            "left": geometry["X"],
            "top": geometry["Y"],
            "width": geometry["WIDTH"],
            "height": geometry["HEIGHT"],
        }

        self._window_monitors[
            window.window_id
        ] = monitor

        return monitor

    def _create_mss_capture(self) -> object:
        factory = self.mss_factory

        if factory is None:
            try:
                from mss import mss
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "required Python package is "
                    "unavailable: mss"
                ) from error

            factory = mss

        try:
            capture = factory()
        except Exception as error:
            raise RuntimeError(
                "failed to initialize MSS capture"
            ) from error

        if not callable(
            getattr(
                capture,
                "grab",
                None,
            )
        ):
            close = getattr(
                capture,
                "close",
                None,
            )

            if callable(close):
                close()

            raise RuntimeError(
                "MSS capture backend does not "
                "provide grab()"
            )

        return capture

    def _require_mss_capture(self) -> object:
        if self._mss_capture is None:
            self._mss_capture = (
                self._create_mss_capture()
            )

        return self._mss_capture

    def _capture_rgb_mss(
        self,
        window: X11Window,
    ) -> np.ndarray:
        monitor = self._window_monitor(window)
        capture = self._require_mss_capture()

        grab = getattr(
            capture,
            "grab",
        )

        try:
            shot = grab(monitor)
            bgra = np.asarray(
                shot,
                dtype=np.uint8,
            )
        except Exception as error:
            raise RuntimeError(
                "MSS window capture failed"
            ) from error

        if (
            bgra.ndim != 3
            or bgra.shape[2] < 3
            or bgra.size == 0
        ):
            raise RuntimeError(
                "MSS returned an invalid BGRA frame"
            )

        rgb = np.ascontiguousarray(
            bgra[:, :, :3][:, :, ::-1]
        )

        if (
            rgb.ndim != 3
            or rgb.shape[2] != 3
            or rgb.dtype != np.uint8
            or rgb.size == 0
        ):
            raise RuntimeError(
                "captured image is not a valid RGB frame"
            )

        return rgb

    def _capture_rgb_imagemagick(
        self,
        window: X11Window,
    ) -> np.ndarray:
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

    def capture_rgb(
        self,
        window: X11Window,
    ) -> np.ndarray:
        """Capture a window and return H×W×3 uint8 RGB."""

        if not isinstance(window, X11Window):
            raise TypeError(
                "window must be an X11Window instance"
            )

        if self.backend == "mss":
            return self._capture_rgb_mss(window)

        return self._capture_rgb_imagemagick(
            window
        )

    def close(self) -> None:
        """Release persistent capture resources."""

        capture = self._mss_capture

        self._mss_capture = None
        self._window_monitors.clear()

        if capture is None:
            return

        close = getattr(
            capture,
            "close",
            None,
        )

        if callable(close):
            close()

    def __enter__(
        self,
    ) -> "X11WindowCapture":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


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
