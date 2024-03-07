"""Record docs/demo.gif (45 s) by driving the real Streamlit chat UI with headless Chromium.

Dev-only helper (not needed to run the project):
    pip install playwright pillow && playwright install chromium
    streamlit run app/streamlit_app.py --server.port 8501 --server.headless true &
    python scripts/record_demo.py http://localhost:8501 docs/demo.gif
"""
import io
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

TOTAL_SECONDS = 45.0
QUESTIONS = [
    "How do I split a check by seat?",
    "How much does the Pro plan cost per month?",
    "What is the capital of France?",
]


def main(url: str, out: str) -> None:
    shots: list[tuple[bytes, float]] = []  # (png, relative weight)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 860})
        page.goto(url)
        page.wait_for_selector("[data-testid=stChatInputTextArea]", timeout=60000)
        page.wait_for_timeout(1000)
        shots.append((page.screenshot(), 3))
        box = page.locator("[data-testid=stChatInputTextArea]")
        for i, q in enumerate(QUESTIONS):
            box.click()
            box.type(q, delay=25)
            shots.append((page.screenshot(), 2))
            box.press("Enter")
            page.wait_for_function(
                "n => document.querySelectorAll('[data-testid=stChatMessage]').length >= n",
                arg=2 * (i + 1), timeout=30000)
            page.wait_for_timeout(1200)
            shots.append((page.screenshot(), 4))
            if i < 2:  # open the first cited source to show the evidence
                page.locator("[data-testid=stExpander] summary").last.scroll_into_view_if_needed()
                page.locator("[data-testid=stExpander] summary", has_text="cited").first.click()
                page.wait_for_timeout(600)
                shots.append((page.screenshot(), 5))
        browser.close()
    total_w = sum(w for _, w in shots)
    frames, durations = [], []
    for png, w in shots:
        img = Image.open(io.BytesIO(png)).convert("RGB")
        img = img.resize((960, int(img.height * 960 / img.width)), Image.LANCZOS)
        frames.append(img.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
        durations.append(int(TOTAL_SECONDS * 1000 * w / total_w))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {out}: {len(frames)} frames, {sum(durations) / 1000:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
