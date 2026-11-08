#!/usr/bin/env python
"""Check multi-chunk video playback against a running model-loaded app server.

python scripts/test/test_app_streaming.py --url http://127.0.0.1:7860
For Linux WebKit H.264 support, use --headed under xvfb-run with GStreamer codecs.
"""
import argparse
import json
import time
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright


def run(browser, engine, args):
    context = browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
    page = context.new_page()
    chunks = []
    append_errors = []
    errors = []
    page.on('console', lambda message: append_errors.append(message.text)
            if message.type == 'error' and 'bufferAppendError' in message.text else None)
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('response', lambda response: chunks.append(response.url) if '/stream/' in response.url and response.status == 200 and 'playlist' not in response.url else None)
    page.goto(args.url)
    photo = next(Path('assets/example_images/2k2k/00093_0').glob('*.png'))
    page.locator('#upload_box input[type=file]').first.set_input_files(str(photo.resolve()))
    expect(page.locator('#upload_box .delete-button')).to_have_count(1, timeout=15000)
    page.get_by_role('button', name='Reconstruct Avatar', exact=True).tap()
    expect(page.locator('#viewer_out iframe')).to_be_visible(timeout=180000)
    page.get_by_role('tab', name='2. Animate', exact=True).tap()
    start = time.monotonic()
    page.get_by_role('button', name='Render Animation', exact=True).tap()
    video = page.locator('#rendered_motion_video video')
    expect(video).to_be_visible(timeout=30000)
    video.scroll_into_view_if_needed()
    video.evaluate('e => { e.muted = true; e.play().catch(() => {}); }')
    page.wait_for_function("id => { const e = document.getElementById('rendered_motion_video').querySelector('video'); return e && e.currentTime > 0 && !e.paused; }", arg=args.video_id, timeout=30000)
    source = video.evaluate("e => e.currentSrc || e.src")
    first_play = time.monotonic() - start
    playing_before_complete = page.get_by_text('Animated with', exact=False).count() == 0
    expect(page.get_by_text('Animated with', exact=False)).to_be_visible(timeout=60000)
    page.wait_for_function("[...document.querySelectorAll('video')].some(e => e.readyState >= 2 && e.currentTime > 0)")
    elapsed = time.monotonic() - start
    assert len(chunks) >= 2, chunks
    assert playing_before_complete, 'Playback began only after rendering completed'
    assert video.evaluate('e => e.playsInline')
    assert not errors, errors
    page.wait_for_function("() => { const e = document.querySelector('#rendered_motion_video video'); return e && e.ended && e.currentTime >= 3.8; }", timeout=45000)
    assert video.evaluate("e => e.currentSrc || e.src") == source, "Player restarted during the stream"
    assert not append_errors, append_errors
    print(f'PASS {engine}: {len(chunks)} TS responses; playback at {first_play:.2f}s before completion at {elapsed:.2f}s; inline video', flush=True)
    context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    parser.add_argument('--headed', action='store_true')
    parser.add_argument('--engines', nargs='+', default=['chromium', 'webkit'])
    args = parser.parse_args()
    deadline = time.monotonic() + 120
    while True:
        try:
            with urlopen(args.url + '/config', timeout=2) as response:
                config = json.load(response)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.5)
    component = next(c for c in config['components'] if c['props'].get('label') == 'Rendered Animation')
    args.video_id = component["id"]
    assert component['props']['streaming'] and component['props']['autoplay']
    with sync_playwright() as playwright:
        for engine in args.engines:
            browser = getattr(playwright, engine).launch(headless=not args.headed)
            try:
                run(browser, engine, args)
            finally:
                browser.close()


if __name__ == '__main__':
    main()
