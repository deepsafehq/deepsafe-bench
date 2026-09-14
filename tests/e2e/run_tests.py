"""End-to-end test for all 4 provenance detection services.

Sends real AI-generated and control media to each running service
and validates the responses.
"""

import base64
import json
import os
import sys
import urllib.request

SERVICES = {
    "c2pa_checker": "http://localhost:9011",
    "sdxl_watermark_detector": "http://localhost:9002",
    "audioseal_detector": "http://localhost:9003",
    "videoseal_detector": "http://localhost:9004",
}

SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "provenance"
)

passed = 0
failed = 0
errors = []


def predict(service_url, payload):
    """Send a /predict request and return the JSON response."""
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


def load_b64(filename):
    """Load a file as base64 string."""
    path = os.path.join(SAMPLE_DIR, filename)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def check(test_name, result, expected_min, expected_max, expected_class=None):
    """Validate a test result."""
    global passed, failed
    prob = result.get("probability")
    cls = result.get("class")

    if "error" in result:
        failed += 1
        errors.append(f"  FAIL {test_name}: {result['error']}")
        return

    ok = True
    msgs = []

    if prob is None:
        ok = False
        msgs.append("missing probability")
    elif prob < expected_min or prob > expected_max:
        ok = False
        msgs.append(
            f"probability={prob:.4f}, expected [{expected_min}, {expected_max}]"
        )

    if expected_class and cls != expected_class:
        ok = False
        msgs.append(f"class='{cls}', expected '{expected_class}'")

    if ok:
        passed += 1
        print(f"  PASS {test_name} (prob={prob:.4f}, class={cls})")
    else:
        failed += 1
        detail = ", ".join(msgs)
        errors.append(f"  FAIL {test_name}: {detail}")
        print(f"  FAIL {test_name}: {detail}")


# ── Test Suite ──────────────────────────────────────────────────────────────

print("=" * 70)
print("PROVENANCE SERVICE END-TO-END TESTS")
print("=" * 70)

# --- C2PA Checker (port 9001) ---
print("\n--- c2pa_checker (port 9001) ---")

# C2PA-signed images (Adobe test fixtures — have C2PA manifest but may not
# have AI generator assertions, so they test C2PA parsing)
b64 = load_b64("c2pa_test_C.jpg")
result = predict(SERVICES["c2pa_checker"], {"image_data": b64})
check("C2PA signed image (C)", result, 0.5, 1.0)

b64 = load_b64("c2pa_test_CA.jpg")
result = predict(SERVICES["c2pa_checker"], {"image_data": b64})
check("C2PA signed image (CA)", result, 0.5, 1.0)

b64 = load_b64("c2pa_test_CACA.jpg")
result = predict(SERVICES["c2pa_checker"], {"image_data": b64})
check("C2PA signed image (CACA)", result, 0.5, 1.0)

# IPTC AI-tagged image (no C2PA backing — should be neutral after hardening)
b64 = load_b64("iptc_ai_tagged.jpg")
result = predict(SERVICES["c2pa_checker"], {"image_data": b64})
check("IPTC AI-tagged image (no C2PA)", result, 0.5, 0.5, "real")

# Control: clean image (no metadata)
b64 = load_b64("control_image.jpg")
result = predict(SERVICES["c2pa_checker"], {"image_data": b64})
check("Control image (no metadata)", result, 0.5, 0.5, "real")

# Control: clean audio
b64 = load_b64("control_audio.wav")
result = predict(SERVICES["c2pa_checker"], {"audio_data": b64})
check("Control audio (no metadata)", result, 0.5, 0.5, "real")

# Control: clean video
b64 = load_b64("control_video.mp4")
result = predict(SERVICES["c2pa_checker"], {"video_data": b64})
check("Control video (no metadata)", result, 0.5, 0.5, "real")


# --- SDXL Watermark Detector (port 9002) ---
print("\n--- sdxl_watermark_detector (port 9002) ---")

# SDXL watermarked image
b64 = load_b64("sdxl_watermarked.png")
result = predict(SERVICES["sdxl_watermark_detector"], {"image_data": b64})
check("SDXL watermarked image", result, 0.7, 1.0, "fake")

# Control: clean random image
b64 = load_b64("clean_random.png")
result = predict(SERVICES["sdxl_watermark_detector"], {"image_data": b64})
check("Control random image (no watermark)", result, 0.5, 0.5, "real")

# Control: real photograph
b64 = load_b64("control_image.jpg")
result = predict(SERVICES["sdxl_watermark_detector"], {"image_data": b64})
check("Control photograph (no watermark)", result, 0.45, 0.55)


# --- AudioSeal Detector (port 9003) ---
print("\n--- audioseal_detector (port 9003) ---")

# AudioSeal watermarked audio
b64 = load_b64("audioseal_watermarked.wav")
result = predict(SERVICES["audioseal_detector"], {"audio_data": b64})
check("AudioSeal watermarked audio", result, 0.7, 1.0, "fake")

# Control: clean speech
b64 = load_b64("control_audio.wav")
result = predict(SERVICES["audioseal_detector"], {"audio_data": b64})
check("Control speech (no watermark)", result, 0.45, 0.55)


# --- VideoSeal Detector (port 9004) ---
print("\n--- videoseal_detector (port 9004) ---")

# Control: clean image
b64 = load_b64("control_image.jpg")
result = predict(SERVICES["videoseal_detector"], {"image_data": b64})
check("Control image (no VideoSeal)", result, 0.45, 0.55)

# Control: clean video
b64 = load_b64("control_video.mp4")
result = predict(SERVICES["videoseal_detector"], {"video_data": b64})
check("Control video (no VideoSeal)", result, 0.45, 0.55)


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed")
if errors:
    print("\nFailures:")
    for e in errors:
        print(e)
print("=" * 70)

sys.exit(1 if failed > 0 else 0)
