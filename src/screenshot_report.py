"""Screenshot the HTML report across viewport × theme × language (Stage-9 visual gate).

Renders the report with a headless Chromium and writes 8 PNGs:
  {desktop,mobile} × {light,dark} × {ko,en}
into an output dir (default: <report_dir>/_review/). Use these to eyeball the
prose-driven essay layout: no element overlap, consistent spacing rhythm, prose
column vs breakout-figure widths read intentionally, mobile collapses to one
column, EN (runs longer than KO) doesn't break layout, both themes legible.

It drives the report's own toggles: sets localStorage `ddr-theme` / `ddr-lang`
and reloads so the page applies them exactly as a user would.

Usage:
    python screenshot_report.py path/to/report.html [--out DIR]

Requires: playwright  (`uv add --dev playwright && uv run playwright install chromium`).
"""
import argparse
import os
import sys

from playwright.sync_api import sync_playwright

VIEWPORTS = {"desktop": (1440, 900), "mobile": (390, 844)}
THEMES = ("light", "dark")
LANGS = ("ko", "en")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--out", default=None, help="output dir (default: <report_dir>/_review)")
    args = ap.parse_args()

    path = os.path.abspath(args.report)
    if not os.path.exists(path):
        print(f"report not found: {path}")
        sys.exit(1)
    out = os.path.abspath(args.out or os.path.join(os.path.dirname(path), "_review"))
    os.makedirs(out, exist_ok=True)
    url = "file://" + path

    shots = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for vp_name, (w, h) in VIEWPORTS.items():
            ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=2)
            page = ctx.new_page()
            # domcontentloaded, not load: external CDN <link>/<script> (fonts, lucide,
            # footer.js) can hang the "load" event offline or on a slow network.
            page.goto(url, wait_until="domcontentloaded")
            for theme in THEMES:
                for lang in LANGS:
                    page.evaluate(
                        "([t,l])=>{localStorage.setItem('ddr-theme',t);localStorage.setItem('ddr-lang',l);}",
                        [theme, lang],
                    )
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_timeout(1200)  # let fonts + lucide icons settle
                    dest = os.path.join(out, f"{vp_name}_{theme}_{lang}.png")
                    page.screenshot(path=dest, full_page=True)
                    shots += 1
                    print(f"  wrote {os.path.relpath(dest)}")
            ctx.close()
        browser.close()

    print(f"{shots} screenshots → {out}")


if __name__ == "__main__":
    main()
