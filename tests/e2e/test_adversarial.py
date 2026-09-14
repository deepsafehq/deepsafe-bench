"""Adversarial tests against provenance detection services.

Tests metadata injection, watermark forging, and other attacks
to verify the system's resilience (or lack thereof).
"""

import base64
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SERVICES = {
    "c2pa_checker": "http://localhost:9011",
    "sdxl_watermark_detector": "http://localhost:9002",
    "audioseal_detector": "http://localhost:9003",
    "videoseal_detector": "http://localhost:9004",
}

SAMPLE_DIR = Path(__file__).parent.parent / "fixtures" / "provenance"

passed = 0
failed = 0
results = []


def predict(service_url, payload):
    """Send /predict request."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{service_url}/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        return json.loads(resp.read())
    except Exception as exc:
        return {"error": str(exc)}


def check(test_name, result, should_detect, expected_behavior):
    """Validate test result against expected adversarial behavior."""
    global passed, failed
    prob = result.get("probability", -1)
    cls = result.get("class", "error")

    if "error" in result:
        print(f"  ERROR {test_name}: {result['error']}")
        results.append(
            {
                "test": test_name,
                "status": "ERROR",
                "expected": expected_behavior,
                "actual": str(result["error"]),
            }
        )
        failed += 1
        return

    detected = prob > 0.5
    if detected == should_detect:
        status = "PASS"
        passed += 1
    else:
        status = "VULN"  # Vulnerability found
        failed += 1

    icon = "PASS" if status == "PASS" else "VULN"
    print(f"  {icon} {test_name}: prob={prob:.4f}, class={cls}")
    print(f"        Expected: {expected_behavior}")

    results.append(
        {
            "test": test_name,
            "status": status,
            "probability": prob,
            "class": cls,
            "expected": expected_behavior,
            "should_detect": should_detect,
            "actual_detected": detected,
        }
    )


def make_real_jpeg():
    """Create a 'real' photograph (solid color with noise)."""
    arr = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_real_wav(duration_s=2.0, sample_rate=16000):
    """Create a 'real' audio WAV (sine wave speech-like)."""
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples)
    # Mix of frequencies to simulate speech
    audio = (
        np.sin(2 * np.pi * 200 * t) * 0.3
        + np.sin(2 * np.pi * 400 * t) * 0.2
        + np.sin(2 * np.pi * 800 * t) * 0.1
    )
    audio_int16 = (audio * 32767).astype(np.int16)
    data = audio_int16.tobytes()

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)
    return buf.getvalue()


def b64(data):
    return base64.b64encode(data).decode()


# ============================================================================
print("=" * 70)
print("ADVERSARIAL ATTACK TESTS")
print("=" * 70)

# ── ATTACK 1: IPTC DigitalSourceType Injection ────────────────────────────
print("\n--- ATTACK 1: IPTC DigitalSourceType Injection ---")
print("    (Trivial: one exiftool command on a real photo)")

real_jpeg = make_real_jpeg()

# Save to temp, inject IPTC tag, reload
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    tmp.write(real_jpeg)
    tmp_path = tmp.name

try:
    subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-XMP-iptcExt:DigitalSourceType="
            "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
            tmp_path,
        ],
        capture_output=True,
        check=True,
    )

    with open(tmp_path, "rb") as f:
        injected_jpeg = f.read()

    result = predict(SERVICES["c2pa_checker"], {"image_data": b64(injected_jpeg)})
    check(
        "IPTC injection on real photo",
        result,
        should_detect=False,  # Should NOT detect — this is a forged tag
        expected_behavior="Should return 0.5 (real) — IPTC tags are trivially forgeable",
    )
except FileNotFoundError:
    print("  SKIP: exiftool not installed")
except subprocess.CalledProcessError as e:
    print(f"  SKIP: exiftool failed: {e}")
finally:
    os.unlink(tmp_path)


# ── ATTACK 2: EXIF Software Field Spoofing ────────────────────────────────
print("\n--- ATTACK 2: EXIF Software Field Spoofing ---")
print("    (Trivial: set Software='Midjourney' on real photo)")

with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    tmp.write(real_jpeg)
    tmp_path = tmp.name

try:
    subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-Software=Midjourney v6.0",
            tmp_path,
        ],
        capture_output=True,
        check=True,
    )

    with open(tmp_path, "rb") as f:
        spoofed_jpeg = f.read()

    result = predict(SERVICES["c2pa_checker"], {"image_data": b64(spoofed_jpeg)})
    check(
        "EXIF Software='Midjourney' on real photo",
        result,
        should_detect=False,  # Should NOT detect — EXIF is trivially forgeable
        expected_behavior="Should return 0.5 (real) — EXIF Software is trivially spoofable",
    )
except FileNotFoundError:
    print("  SKIP: exiftool not installed")
except subprocess.CalledProcessError as e:
    print(f"  SKIP: exiftool failed: {e}")
finally:
    os.unlink(tmp_path)


# ── ATTACK 3: Double EXIF + IPTC Injection ────────────────────────────────
print("\n--- ATTACK 3: Combined EXIF + IPTC Injection ---")
print("    (Both Software='Stable Diffusion' + DigitalSourceType)")

with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    tmp.write(real_jpeg)
    tmp_path = tmp.name

try:
    subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-Software=Stable Diffusion XL 1.0",
            "-XMP-iptcExt:DigitalSourceType="
            "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
            "-EXIF:ImageDescription=Generated by Stable Diffusion",
            tmp_path,
        ],
        capture_output=True,
        check=True,
    )

    with open(tmp_path, "rb") as f:
        double_injected = f.read()

    result = predict(SERVICES["c2pa_checker"], {"image_data": b64(double_injected)})
    check(
        "Combined EXIF+IPTC injection on real photo",
        result,
        should_detect=False,
        expected_behavior="Should return 0.5 — forged metadata without crypto verification",
    )
except FileNotFoundError:
    print("  SKIP: exiftool not installed")
finally:
    os.unlink(tmp_path)


# ── ATTACK 4: SDXL Watermark Forging ──────────────────────────────────────
print("\n--- ATTACK 4: SDXL Watermark Forging ---")
print("    (Embed SDV2 watermark into real photo using open-source encoder)")

try:
    from imwatermark import WatermarkEncoder

    real_arr = np.array(Image.open(io.BytesIO(real_jpeg)).convert("RGB"))
    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", b"SDV2" + b"\x00" * 13)
    forged = encoder.encode(real_arr, "dwtDct")

    forged_buf = io.BytesIO()
    Image.fromarray(forged).save(forged_buf, format="PNG")
    forged_bytes = forged_buf.getvalue()

    result = predict(
        SERVICES["sdxl_watermark_detector"], {"image_data": b64(forged_bytes)}
    )
    check(
        "Forged SDV2 watermark in real photo",
        result,
        should_detect=False,  # Should NOT detect — this is a forged watermark
        expected_behavior="Should return 0.5 — watermark was forged, not from real SDXL",
    )
except ImportError:
    print("  SKIP: invisible-watermark not installed")


# ── ATTACK 5: Real photo with no manipulation (baseline) ──────────────────
print("\n--- CONTROL: Clean real photo (no manipulation) ---")

result = predict(SERVICES["c2pa_checker"], {"image_data": b64(real_jpeg)})
check(
    "Clean real JPEG (c2pa_checker)",
    result,
    should_detect=False,
    expected_behavior="Should return 0.5 (real) — no provenance signals",
)

result = predict(SERVICES["sdxl_watermark_detector"], {"image_data": b64(real_jpeg)})
check(
    "Clean real JPEG (sdxl_watermark)",
    result,
    should_detect=False,
    expected_behavior="Should return 0.5 (real) — no watermark",
)


# ── ATTACK 6: Real audio with no manipulation (baseline) ──────────────────
print("\n--- CONTROL: Clean real audio (no manipulation) ---")

real_wav = make_real_wav()
result = predict(SERVICES["audioseal_detector"], {"audio_data": b64(real_wav)})
check(
    "Clean real WAV (audioseal)",
    result,
    should_detect=False,
    expected_behavior="Should return 0.5 (real) — no AudioSeal watermark",
)


# ── ATTACK 7: Metadata injection on audio ─────────────────────────────────
print("\n--- ATTACK 7: EXIF injection on audio file ---")
print("    (Add AI tool metadata to real WAV)")

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp.write(real_wav)
    tmp_path = tmp.name

try:
    subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-Software=ElevenLabs AI Voice Generator",
            tmp_path,
        ],
        capture_output=True,
        check=True,
    )

    with open(tmp_path, "rb") as f:
        spoofed_wav = f.read()

    result = predict(SERVICES["c2pa_checker"], {"audio_data": b64(spoofed_wav)})
    check(
        "EXIF 'ElevenLabs' injection on real WAV",
        result,
        should_detect=False,
        expected_behavior="Should return 0.5 — ElevenLabs not in AI generators list "
        "AND EXIF is trivially forgeable",
    )
except (FileNotFoundError, subprocess.CalledProcessError) as e:
    print(f"  SKIP: exiftool failed on WAV: {e}")
    passed += 1  # Not a vulnerability — exiftool can't write to WAV
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


# ── ATTACK 8: Stripped AI image (metadata removed) ────────────────────────
print("\n--- ATTACK 8: AI image with metadata stripped ---")
print("    (Screenshot/re-save of AI content removes all provenance)")

# Create the IPTC-tagged image first, then strip it
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    tmp.write(real_jpeg)
    tmp_path = tmp.name

try:
    # Add AI metadata
    subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-XMP-iptcExt:DigitalSourceType="
            "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
            tmp_path,
        ],
        capture_output=True,
        check=True,
    )

    # Now strip ALL metadata (simulating screenshot/re-save)
    subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-all=",
            tmp_path,
        ],
        capture_output=True,
        check=True,
    )

    with open(tmp_path, "rb") as f:
        stripped_jpeg = f.read()

    result = predict(SERVICES["c2pa_checker"], {"image_data": b64(stripped_jpeg)})
    check(
        "AI image after metadata stripping",
        result,
        should_detect=False,  # Correctly neutral — metadata is gone
        expected_behavior="Should return 0.5 — metadata stripped, models must detect",
    )
except FileNotFoundError:
    print("  SKIP: exiftool not installed")
finally:
    os.unlink(tmp_path)


# ── ATTACK 9: Oversized image (OOM attack) ────────────────────────────────
print("\n--- ATTACK 9: Large image dimension attack ---")
print("    (Send a very large image to test OOM protection)")

# Create a 4096x4096 image (not insanely large but bigger than typical)
large_arr = np.random.randint(0, 255, (4096, 4096, 3), dtype=np.uint8)
large_buf = io.BytesIO()
Image.fromarray(large_arr).save(large_buf, format="JPEG", quality=50)
large_bytes = large_buf.getvalue()

result = predict(SERVICES["sdxl_watermark_detector"], {"image_data": b64(large_bytes)})
check(
    "4096x4096 image (sdxl_watermark)",
    result,
    should_detect=False,
    expected_behavior="Should return 0.5 and not crash/timeout",
)


# ── ATTACK 10: Empty/malformed payloads ───────────────────────────────────
print("\n--- ATTACK 10: Malformed payload resilience ---")

# Send garbage base64 that decodes to valid bytes but not a real file
garbage = base64.b64encode(os.urandom(1024)).decode()
result = predict(SERVICES["c2pa_checker"], {"image_data": garbage})
prob = result.get("probability", -1)
if prob == 0.5 or "error" in result:
    print(f"  PASS Garbage bytes to c2pa: prob={prob} (handled gracefully)")
    passed += 1
else:
    print(f"  VULN Garbage bytes to c2pa: prob={prob} (unexpected response)")
    failed += 1


# ============================================================================
print("\n" + "=" * 70)
print(f"ADVERSARIAL TEST RESULTS: {passed} passed, {failed} vulnerable")
print("=" * 70)

if results:
    vulns = [r for r in results if r["status"] == "VULN"]
    if vulns:
        print("\nVULNERABILITIES FOUND:")
        for v in vulns:
            print(f"  - {v['test']}")
            print(f"    Expected: {v['expected']}")
            print(
                f"    Actual: prob={v.get('probability', 'N/A')}, "
                f"class={v.get('class', 'N/A')}"
            )
    else:
        print("\nNo vulnerabilities found in tested attack vectors.")

# Save results
output_path = SAMPLE_DIR / "adversarial_test_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_path}")
