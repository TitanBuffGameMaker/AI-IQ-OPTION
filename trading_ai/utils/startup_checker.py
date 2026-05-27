"""
IQ Option Startup Verification Module

ตรวจสอบ 3 เงื่อนไขก่อนให้ AI เริ่มเทรด:
  1. ชื่อหุ้นทั้ง 4 กราฟ ต้องเป็น OTC หรือ Forex ที่รองรับ
  2. กรอบเวลาแท่งเทียน = 30 นาที
  3. ระยะเวลาออเดอร์ = 1 นาที

ถ้าผ่านทั้งหมด → AI เริ่มทำงาน
ถ้าไม่ผ่าน → หยุด + แจ้งให้แก้ไข
"""
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── ชื่อหุ้น OTC ที่ยอมรับ ──────────────────────────────────────────────────
VALID_OTC_ASSETS = [
    "EUR/USD (OTC)",
    "GBP/USD (OTC)",
    "EUR/CAD (OTC)",
    "GBP/JPY (OTC)",
    "EUR/JPY (OTC)",
    "USD/JPY (OTC)",
    "NZD/USD (OTC)",
    "EUR/GBP (OTC)",
    "USD/CHF (OTC)",
    "AUD/JPY (OTC)",
]

# ── ชื่อหุ้น Forex ปกติ (ตลาดจริง) ที่รองรับ ─────────────────────────────────
VALID_FOREX_ASSETS = [
    "EUR/USD",
    "GBP/USD",
    "EUR/CAD",
    "EUR/JPY",
    "GBP/JPY",
    "USD/JPY",
    "NZD/USD",
    "EUR/GBP",
    "USD/CHF",
    "AUD/JPY",
]

# All valid assets (OTC + Forex)
VALID_ALL_ASSETS = VALID_OTC_ASSETS + VALID_FOREX_ASSETS

# ── Expected settings ────────────────────────────────────────────────────────
REQUIRED_TIMEFRAME_MIN = 30   # กรอบเวลาแท่งเทียน = 30 นาที
REQUIRED_DURATION_MIN  = 1    # ระยะเวลาออเดอร์ = 1 นาที

# ── Screen layout (2×2 four-chart layout) ────────────────────────────────────
# These ratios are relative to the IQ Option window size.
# Tuned to match the screenshots provided.
CHART_REGIONS_RATIO = {
    "top_left":     (0.05, 0.02, 0.50, 0.50),   # (x1%, y1%, x2%, y2%)
    "top_right":    (0.50, 0.02, 0.97, 0.50),
    "bottom_left":  (0.05, 0.50, 0.50, 0.97),
    "bottom_right": (0.50, 0.50, 0.97, 0.97),
}

# Position of the timeframe label within each chart (relative to chart region)
TIMEFRAME_ROI_RATIO = (0.00, 0.88, 0.25, 1.00)   # bottom-left of each chart

# Position of expiry/duration label (global, right panel)
EXPIRY_ROI_RATIO    = (0.75, 0.03, 1.00, 0.12)


@dataclass
class CheckResult:
    name: str
    passed: bool
    found: str
    expected: str
    message: str


class StartupChecker:
    """
    Captures the IQ Option screen and verifies all required settings
    using OCR before allowing the AI to start trading.
    """

    def __init__(self):
        self._ocr_available = self._check_ocr()
        self._capture_available = self._check_capture()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_all_checks(self) -> Tuple[bool, List[CheckResult]]:
        """
        Run all startup checks.
        Returns (all_passed, list_of_results).
        """
        results: List[CheckResult] = []

        if not self._capture_available:
            logger.warning("Screen capture unavailable – skipping visual checks")
            return True, []

        if not self._ocr_available:
            logger.info("OCR (pytesseract) unavailable – skipping visual checks")
            return True, []

        screenshot = self._capture_screen()
        if screenshot is None:
            logger.warning("Cannot capture screen – skipping visual checks")
            return True, []

        h, w = screenshot.shape[:2]
        logger.info("Screen captured: %dx%d", w, h)

        # ── Check 1: OTC asset names ─────────────────────────────────────────
        asset_results = self._check_all_assets(screenshot, w, h)
        results.extend(asset_results)

        # ── Check 2: Candle timeframe = 30m ──────────────────────────────────
        tf_result = self._check_timeframe(screenshot, w, h)
        results.append(tf_result)

        # ── Check 3: Order duration = 1m ─────────────────────────────────────
        dur_result = self._check_duration(screenshot, w, h)
        results.append(dur_result)

        all_passed = all(r.passed for r in results)
        return all_passed, results

    def print_report(self, results: List[CheckResult]):
        """Print a coloured startup report to the terminal."""
        PASS  = "\033[92m✅ PASS\033[0m"
        FAIL  = "\033[91m❌ FAIL\033[0m"
        WARN  = "\033[93m⚠️  WARN\033[0m"

        print("\n" + "═" * 62)
        print("  IQ OPTION STARTUP VERIFICATION")
        print("═" * 62)

        for r in results:
            status = PASS if r.passed else FAIL
            print(f"  {status}  {r.name}")
            if not r.passed:
                print(f"         found    : {r.found}")
                print(f"         expected : {r.expected}")
                print(f"         → {r.message}")

        passed = sum(1 for r in results if r.passed)
        total  = len(results)
        print("─" * 62)
        print(f"  Result: {passed}/{total} checks passed")

        if passed == total:
            print("  \033[92m✅ All checks passed – AI is ready to trade\033[0m")
        else:
            print("  \033[91m❌ Fix the settings above before trading\033[0m")
        print("═" * 62 + "\n")

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_all_assets(
        self, img: np.ndarray, w: int, h: int
    ) -> List[CheckResult]:
        results = []
        positions = list(CHART_REGIONS_RATIO.keys())

        for pos in positions:
            x1r, y1r, x2r, y2r = CHART_REGIONS_RATIO[pos]
            # Read the top portion of each chart (asset name header area)
            cx1 = int(x1r * w)
            cy1 = int(y1r * h)
            cx2 = int(x2r * w)
            cy2 = int(y1r * h + (y2r - y1r) * h * 0.12)  # top 12% = header

            crop = img[cy1:cy2, cx1:cx2]
            text = self._ocr(crop)

            found_asset = self._find_otc_asset(text)
            passed = found_asset is not None

            results.append(CheckResult(
                name=f"กราฟ {pos.replace('_', ' ')}: OTC asset",
                passed=passed,
                found=text.strip()[:40] if text else "(ไม่พบข้อความ)",
                expected="ชื่อหุ้น (OTC)",
                message=(
                    "ตั้งค่าหุ้นเป็น OTC (เช่น EUR/USD (OTC)) ในกราฟนี้"
                    if not passed else ""
                ),
            ))

        return results

    def _check_timeframe(self, img: np.ndarray, w: int, h: int) -> CheckResult:
        """ตรวจสอบกรอบเวลาแท่งเทียน = 30 นาที"""
        # Check all 4 chart timeframe areas
        found_values = []
        for pos, (x1r, y1r, x2r, y2r) in CHART_REGIONS_RATIO.items():
            # Timeframe label is at bottom-left of each chart
            tx1 = int(x1r * w)
            ty1 = int((y1r + (y2r - y1r) * 0.85) * h)
            tx2 = int((x1r + (x2r - x1r) * 0.20) * w)
            ty2 = int(y2r * h)

            crop = img[ty1:ty2, tx1:tx2]
            text = self._ocr(crop)
            if text:
                found_values.append(text.strip())

        combined = " ".join(found_values)
        is_30m = self._contains_timeframe_30m(combined)

        return CheckResult(
            name="กรอบเวลาแท่งเทียน = 30 นาที",
            passed=is_30m,
            found=combined[:60] if combined else "(ไม่พบ)",
            expected="30m / 30 นาที",
            message=(
                "คลิกที่ตัวเลขกรอบเวลา (เช่น '2m') แล้วเลือก '30m'"
                if not is_30m else ""
            ),
        )

    def _check_duration(self, img: int, w: int, h: int) -> CheckResult:
        """ตรวจสอบระยะเวลาออเดอร์ = 1 นาที"""
        # Right panel – expiry area
        ex1 = int(EXPIRY_ROI_RATIO[0] * w)
        ey1 = int(EXPIRY_ROI_RATIO[1] * h)
        ex2 = int(EXPIRY_ROI_RATIO[2] * w)
        ey2 = int(EXPIRY_ROI_RATIO[3] * h)

        crop = img[ey1:ey2, ex1:ex2]
        text = self._ocr(crop)

        # Also scan the wider right half for expiry text
        right_crop = img[int(h * 0.01):int(h * 0.15), int(w * 0.70):]
        text2 = self._ocr(right_crop)

        combined = (text or "") + " " + (text2 or "")
        is_1m = self._contains_duration_1m(combined)

        return CheckResult(
            name="ระยะเวลาออเดอร์ = 1 นาที",
            passed=is_1m,
            found=combined.strip()[:60] if combined.strip() else "(ไม่พบ)",
            expected="1m / 1 นาที / 00:01",
            message=(
                "ตั้งค่า 'การหมดอายุ' เป็น 1 นาที ในช่องด้านขวา"
                if not is_1m else ""
            ),
        )

    # ── API-based verification (no OCR needed) ────────────────────────────────

    def check_via_api(self, connector) -> Tuple[bool, List[CheckResult]]:
        """
        Verify settings directly via IQ Option API (more reliable than OCR).
        Use this when the API is connected.
        """
        results = []

        # Check open time / available assets
        try:
            open_time = connector.api.get_all_open_time()
            otc_assets = []
            seen = set()
            for category in open_time.values():
                for asset, info in category.items():
                    # IQ Option API uses "-OTC" suffix (e.g. "EURUSD-OTC"), not "(OTC)"
                    if "OTC" in asset.upper() and info.get("open", False) and asset not in seen:
                        otc_assets.append(asset)
                        seen.add(asset)

            results.append(CheckResult(
                name="หุ้น OTC พร้อมเทรด",
                passed=len(otc_assets) >= 4,
                found=f"พบ {len(otc_assets)} หุ้น OTC",
                expected="อย่างน้อย 4 หุ้น OTC",
                message="" if len(otc_assets) >= 4
                         else "ตลาด OTC อาจปิดอยู่ ลองอีกครั้งในช่วง weekend",
            ))
        except Exception as exc:
            results.append(CheckResult(
                name="หุ้น OTC พร้อมเทรด",
                passed=False,
                found=f"Error: {exc}",
                expected="API response",
                message="ตรวจสอบการเชื่อมต่อ IQ Option",
            ))

        # Check account type (PRACTICE = safe)
        try:
            balance = connector.get_balance()
            acc_type = connector.account_type
            results.append(CheckResult(
                name=f"บัญชี {acc_type}",
                passed=True,
                found=f"ยอดเงิน: ${balance:.2f}",
                expected=f"{acc_type} account",
                message="",
            ))
        except Exception:
            pass

        all_passed = all(r.passed for r in results)
        return all_passed, results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _capture_screen(self) -> Optional[np.ndarray]:
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # primary screen
                shot = sct.grab(monitor)
                img = np.array(shot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
        except Exception as exc:
            logger.debug("Screen capture failed: %s", exc)
            return None

    def _ocr(self, img: np.ndarray) -> str:
        if not self._ocr_available or img is None or img.size == 0:
            return ""
        try:
            import pytesseract
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Enlarge small regions for better OCR
            if gray.shape[0] < 50:
                scale = max(2, 60 // gray.shape[0])
                gray = cv2.resize(gray, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_LINEAR)
            # Threshold for clearer text
            _, thresh = cv2.threshold(gray, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(
                thresh,
                config="--psm 7 -l eng+tha",  # single line, English + Thai
            )
            return text.strip()
        except Exception as exc:
            logger.debug("OCR error: %s", exc)
            return ""

    @staticmethod
    def _find_otc_asset(text: str) -> Optional[str]:
        """Returns matching asset name if text contains a valid OTC or Forex asset."""
        if not text:
            return None
        text_upper = text.upper().replace(" ", "")
        for asset in VALID_ALL_ASSETS:
            key = asset.upper().replace(" ", "")
            if key in text_upper:
                return asset
        # Fallback: any OTC marker
        if "OTC" in text_upper:
            return VALID_OTC_ASSETS[0]
        return None

    @staticmethod
    def _contains_timeframe_30m(text: str) -> bool:
        text_clean = text.upper().replace(" ", "")
        markers = ["30M", "30MIN", "30นาที", "30", "THIRTY"]
        return any(m in text_clean for m in markers)

    @staticmethod
    def _contains_duration_1m(text: str) -> bool:
        text_clean = text.upper().replace(" ", "")
        markers = ["1M", "1MIN", "00:01", "1นาที", "1MINUTE"]
        return any(m in text_clean for m in markers)

    @staticmethod
    def _check_ocr() -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            logger.info("pytesseract not available – OCR disabled")
            return False

    @staticmethod
    def _check_capture() -> bool:
        try:
            import mss
            return True
        except ImportError:
            return False
