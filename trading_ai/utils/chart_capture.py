"""
Chart image capture and CNN feature extraction.

Captures the IQ Option chart from the screen (or browser window),
resizes it, passes it through a lightweight CNN to extract a compact
feature vector that the RL agent can learn from.
"""
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import screen-capture libraries (not available in headless environments)
try:
    import mss
    import mss.tools
    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class _ChartCNN(nn.Module):
    """
    Lightweight CNN that maps a grayscale 84×84 chart image → 256-dim vector.
    Architecture follows the Nature DQN paper encoder.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # → 32×20×20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # → 64×9×9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # → 64×7×7
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, 256)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 84, 84)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.relu(self.fc(h))


class ChartCapture:
    """
    Captures the IQ Option chart and encodes it to a 256-dim feature vector.

    In headless / no-GUI environments it returns zeros gracefully.
    The CNN weights are trained jointly with the PPO agent via backprop
    through the actor/critic networks – ChartCapture just holds the encoder.
    """

    # Region of the screen (in pixels) where the IQ Option chart lives.
    # Adjust these to match your monitor layout.
    CHART_REGION = {"top": 100, "left": 200, "width": 900, "height": 500}

    def __init__(self, img_size: int = 84, device: Optional[torch.device] = None):
        self.img_size = img_size
        self.device = device or torch.device("cpu")
        self.cnn = _ChartCNN().to(self.device)
        self.cnn.eval()
        self._headless = not (_MSS_AVAILABLE and _PIL_AVAILABLE)
        if self._headless:
            logger.info(
                "ChartCapture running in headless mode – "
                "chart image features will be zeros."
            )

    @property
    def output_dim(self) -> int:
        return 256

    def get_features(self) -> np.ndarray:
        """
        Returns a (256,) float32 numpy array.
        Falls back to zeros when screen capture is unavailable.
        """
        if self._headless:
            return np.zeros(self.output_dim, dtype=np.float32)

        img_tensor = self._capture()
        if img_tensor is None:
            return np.zeros(self.output_dim, dtype=np.float32)

        with torch.no_grad():
            features = self.cnn(img_tensor.to(self.device))  # (1, 256)
        return features.squeeze(0).cpu().numpy()

    def get_cnn(self) -> _ChartCNN:
        """Return the CNN module (so the PPO agent can include it in its graph)."""
        return self.cnn

    def _capture(self) -> Optional[torch.Tensor]:
        """Capture the chart region and return a (1, 1, 84, 84) tensor."""
        try:
            with mss.mss() as sct:
                screenshot = sct.grab(self.CHART_REGION)
                img = Image.frombytes(
                    "RGB",
                    screenshot.size,
                    screenshot.bgra,
                    "raw", "BGRX",
                )
            img = img.convert("L")  # grayscale
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,84,84)
            return tensor
        except Exception as exc:
            logger.debug("Chart capture failed: %s", exc)
            return None
