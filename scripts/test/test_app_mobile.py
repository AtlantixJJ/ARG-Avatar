#!/usr/bin/env python
"""Browser regression checks against a running `python -m app --no_load` server.

Install Playwright in the test environment, then `python -m playwright install chromium webkit`.
Run: python scripts/test/test_app_mobile.py --url http://127.0.0.1:7868
"""
from __future__ import annotations

import argparse
import json
import re
import time
from urllib.request import urlopen
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

PHONE_SIZES = [(320, 568), (375, 667), (390, 844), (430, 932), (667, 375), (844, 390), (932, 430)]


def check_layout(page, stacked: bool) -> None:
    layout = page.locator(".workflow-row:visible").evaluate("""row => {
        const columns = [...row.querySelectorAll(':scope > .workflow-column')]
            .map(e => e.getBoundingClientRect());
        return { viewport: innerWidth, scroll: document.body.scrollWidth,
            docScroll: document.documentElement.scrollWidth, docClient: document.documentElement.clientWidth,
            columns: columns.map(r => ({x: r.x, y: r.y, width: r.width, bottom: r.bottom})) };
    }""")
    assert layout["scroll"] <= layout["viewport"] + 1, layout
    assert layout["docScroll"] <= layout["docClient"] + 1, layout
    a, b = layout["columns"]
    for column in (a, b):
        assert column["x"] >= -1 and column["x"] + column["width"] <= layout["viewport"] + 1, layout
    if stacked:
        assert abs(a["x"] - b["x"]) <= 1, layout
        assert abs(a["width"] - b["width"]) <= 1, layout
        assert b["y"] >= a["bottom"] - 1, layout
    else:
        assert b["x"] > a["x"] + a["width"] - 1, layout
    for button in page.locator(".primary-action:visible").all() if stacked else []:
        box = button.bounding_box()
        assert box["height"] >= 44, box


def run(browser, engine: str, args) -> None:
    for width, height in PHONE_SIZES + [(1280, 900)]:
        phone = width != 1280
        context = browser.new_context(viewport={"width": width, "height": height},
                                      is_mobile=phone, has_touch=phone)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.locator(".splat-viewer-placeholder")).to_be_visible()
        library_button = page.locator("#subject_library button.label-wrap")
        if phone:
            expect(library_button).not_to_have_class(re.compile(r"\bopen\b"))
        else:
            expect(library_button).to_have_class(re.compile(r"\bopen\b"))
        check_layout(page, phone)
        viewer = page.locator(".splat-viewer-placeholder").bounding_box()
        assert 180 <= viewer["height"] <= 640, viewer
        page.get_by_role("tab", name="2. Animate", exact=True).click()
        expect(page.get_by_role("button", name="Render Animation", exact=True)).to_be_visible()
        check_layout(page, phone)
        page.get_by_text("Custom Video Upload", exact=True).click()
        expect(page.get_by_text("Upload a driving video", exact=False)).to_be_visible()
        check_layout(page, phone)
        assert not errors, errors
        context.close()
        print(f"PASS {engine}: {width}x{height}, both tabs and custom video", flush=True)

    # Full touch workflow, plus rotation without resetting an opened library.
    context = browser.new_context(viewport={"width": 390, "height": 844},
                                  is_mobile=True, has_touch=True)
    page = context.new_page()
    page.goto(args.url)
    library_button = page.locator("#subject_library button.label-wrap")
    expect(library_button).not_to_have_class(re.compile(r"\bopen\b"))
    library_button.tap()
    thumbnail = page.locator("#subject_library button.thumbnail-item:visible").first
    expect(thumbnail).to_be_visible(timeout=15000)
    grid = page.locator("#subject_library .grid-container:visible").first
    assert len(grid.evaluate("e => getComputedStyle(e).gridTemplateColumns.split(' ')") ) == 2
    thumbnail.tap()
    delete = page.locator("#upload_box .delete-button").first
    expect(delete).to_be_visible()
    assert delete.evaluate("e => getComputedStyle(e).opacity") == '1'
    box = delete.bounding_box()
    assert box['width'] >= 44 and box['height'] >= 44, box
    delete.tap()
    expect(page.locator("#upload_box .delete-button")).to_have_count(0)
    thumbnail.tap()
    expect(page.locator("#upload_box .delete-button")).to_have_count(1)
    if args.photo:
        page.locator('#upload_box input[type=file]').first.set_input_files(args.photo)
        expect(page.locator('#upload_box .delete-button')).to_have_count(2, timeout=15000)
        expect(page.locator('#upload_box img').last).to_be_visible()
        page.wait_for_function("[...document.querySelectorAll('#upload_box img')].every(e => e.complete && e.naturalWidth > 0)")
    if args.video:
        page.get_by_text("Video", exact=True).tap()
        page.locator('#upload_box_video input[type=file]').first.set_input_files(args.video)
        expect(page.locator('#frames_gallery button.thumbnail-item')).to_have_count(6, timeout=20000)
    page.set_viewport_size({"width": 844, "height": 390})
    expect(library_button).to_have_class(re.compile(r"\bopen\b"))
    check_layout(page, True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("button", name="Reconstruct Avatar", exact=True).tap()
    expect(page.locator(".splat-viewer-placeholder")).to_contain_text("no_load", timeout=15000)
    page.get_by_role("tab", name="2. Animate", exact=True).tap()
    page.get_by_role("button", name="Render Animation", exact=True).tap()
    expect(page.get_by_text("Simulated animation", exact=False)).to_be_visible(timeout=15000)
    videos = page.locator("#rendered_motion_video video")
    expect(videos).to_be_visible(timeout=15000)
    page.wait_for_function("[...document.querySelectorAll('#app_tabs video')].every(e => e.playsInline)")
    video = videos.last
    video.scroll_into_view_if_needed()
    # An unsupported native media backend can leave play() pending indefinitely.
    # Let the bounded playback assertion below report the failure instead.
    video.evaluate("e => { e.muted = true; e.play().catch(() => {}); }")
    page.wait_for_function("id => { const e = document.getElementById('rendered_motion_video').querySelector('video'); return e && !e.paused && e.currentTime > 0; }", arg=args.rendered_video_id)
    check_layout(page, True)
    if args.screenshots:
        directory = Path(args.screenshots)
        directory.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(directory / f"{engine}-animate.png"), full_page=True)
        page.get_by_role("tab", name="1. Reconstruct", exact=True).tap()
        library_button.tap()
        page.screenshot(path=str(directory / f"{engine}-reconstruct.png"), full_page=True)
    context.close()
    print(f"PASS {engine}: touch selection/removal, rotation, mocked reconstruction/animation, inline video playback", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7868")
    parser.add_argument("--screenshots", default="")
    parser.add_argument("--headed", action="store_true",
                        help="Use desktop browser backends (under xvfb-run on a headless Linux host)")
    parser.add_argument("--engines", nargs="+", choices=["chromium", "webkit"],
                        default=["chromium", "webkit"])
    parser.add_argument("--photo", default="", help="Optional photo fixture, including HEIC")
    parser.add_argument("--video", default="", help="Optional video fixture, including MOV")
    args = parser.parse_args()
    # Model imports can make even no-load startup take a few seconds.
    deadline = time.monotonic() + 30
    while True:
        try:
            with urlopen(args.url + "/config", timeout=2) as response:
                config = json.load(response)
                args.rendered_video_id = next(c["id"] for c in config["components"]
                    if c["props"].get("label") == "Rendered Animation")
                break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)
    with sync_playwright() as playwright:
        for engine in args.engines:
            browser = getattr(playwright, engine).launch(headless=not args.headed)
            try:
                run(browser, engine, args)
            finally:
                browser.close()


if __name__ == "__main__":
    main()
