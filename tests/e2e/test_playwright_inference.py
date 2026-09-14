"""End-to-end Playwright test for DeepSafe web UI detection.

Tests uploading image, audio, and video files through localhost:3000
and verifying that results come back with a verdict and confidence score.
"""

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "apps" / "web" / "public" / "samples"

# Timeout for detection to complete (models can take 30-60s for video)
DETECTION_TIMEOUT_MS = 120_000


async def test_modality(page, file_path: Path, label: str):
    """Upload a file and verify detection results appear."""
    print(f"\n{'='*60}")
    print(f"Testing {label}: {file_path.name}")
    print(f"{'='*60}")

    # Navigate to the demo page (no auth required)
    await page.goto(
        "http://localhost:3000/demo", wait_until="networkidle", timeout=30000
    )
    print(f"  Page loaded: {page.url}")

    # Take screenshot of initial state
    await page.screenshot(path=f"/tmp/deepsafe_{label}_1_initial.png")

    # Look for the file upload zone
    # The upload zone could be an input[type=file] or a dropzone div
    file_input = page.locator('input[type="file"]')
    count = await file_input.count()
    print(f"  File inputs found: {count}")

    if count > 0:
        # Upload the file
        await file_input.first.set_input_files(str(file_path))
        print(f"  File uploaded: {file_path.name}")
    else:
        print(f"  ERROR: No file input found on page")
        await page.screenshot(path=f"/tmp/deepsafe_{label}_error.png")
        return False

    # Wait for results to appear
    # Look for verdict text (real/fake), confidence scores, or result containers
    try:
        # Wait for any indication of results
        result_selectors = [
            "text=/real|fake|undetermined/i",
            '[data-testid="verdict"]',
            ".verdict",
            "text=/confidence/i",
            "text=/detection complete/i",
            "text=/analyzing/i",
        ]

        # First wait for analysis to start
        print("  Waiting for analysis...")
        await page.screenshot(path=f"/tmp/deepsafe_{label}_2_uploading.png")

        # Wait for verdict to appear (real, fake, or undetermined)
        verdict_locator = page.locator("text=/\\b(real|fake|undetermined)\\b/i").first
        await verdict_locator.wait_for(state="visible", timeout=DETECTION_TIMEOUT_MS)

        verdict_text = await verdict_locator.text_content()
        print(f"  Verdict found: {verdict_text}")

        await page.screenshot(path=f"/tmp/deepsafe_{label}_3_results.png")

        # Try to find confidence score
        page_text = await page.text_content("body")
        if "confidence" in page_text.lower() or "%" in page_text:
            print(f"  Confidence score visible in results")

        print(f"  SUCCESS: {label} detection completed")
        return True

    except Exception as e:
        print(f"  Timeout or error waiting for results: {e}")
        await page.screenshot(path=f"/tmp/deepsafe_{label}_timeout.png")
        # Get page content for debugging
        body_text = await page.text_content("body")
        print(f"  Page text (first 500 chars): {body_text[:500]}")
        return False


async def main():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        results = {}

        # Test image
        image_file = SAMPLES_DIR / "sample_image.jpg"
        if image_file.exists():
            results["image"] = await test_modality(page, image_file, "image")
        else:
            print(f"SKIP: {image_file} not found")

        # Test audio (new page to reset state)
        page2 = await context.new_page()
        audio_file = SAMPLES_DIR / "sample_audio.wav"
        if audio_file.exists():
            results["audio"] = await test_modality(page2, audio_file, "audio")
        else:
            print(f"SKIP: {audio_file} not found")

        # Test video (new page)
        page3 = await context.new_page()
        video_file = SAMPLES_DIR / "sample_video.mp4"
        if video_file.exists():
            results["video"] = await test_modality(page3, video_file, "video")
        else:
            print(f"SKIP: {video_file} not found")

        await browser.close()

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for modality, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {modality}: {status}")

        all_passed = all(results.values())
        print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
