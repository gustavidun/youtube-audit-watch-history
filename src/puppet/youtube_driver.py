from patchright.async_api import Page, Playwright, BrowserContext, async_playwright
from patchright.async_api import Error as PWError, TimeoutError as PWTimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pathlib import Path
import re 
import asyncio
import logging
import time

from models import Video 
from config import LOG_DIR

class VideoUnavailableException(Exception):
    def __str__(self):
        return self.__class__.__name__

class PlaybackException(Exception):
    def __str__(self):
        return self.__class__.__name__

class YouTubeDriver():
    """Async context manager for interacting with YouTube using Playwright"""
    def __init__(self, session_dir : Path, headless : bool, ublock_path : Path, logger : logging.Logger):
        self.session_dir = session_dir
        self.headless = headless
        self.ublock_path = ublock_path
        self.logger = logger

        self._page : Page | None = None
        self._context : BrowserContext | None = None
        self._playwright : Playwright | None = None


    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self.session_dir.mkdir(exist_ok=True) #create dir if not exists

        self._context = await self._playwright.chromium.launch_persistent_context(
            self.session_dir,
            headless=self.headless,
            channel="chrome",
        )
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()

        session = await self._context.new_cdp_session(self._page)
        self.logger.info("Initialising YouTube Driver.")

        await self._page.goto("https://www.youtube.com")
        self.logger.info("Succesfully launched chrome.")

        await self._register_consent_handler()

        return self


    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._context: await self._context.close()
        finally:
            if self._playwright: await self._playwright.stop()


    async def _register_consent_handler(self):
        """Register a persistent locator handler that auto-dismisses the
        Google/YouTube cookie consent banner whenever it appears."""
        assert self._page

        consent_locator = self._page.locator(
            'button[aria-label*="Accept the use of cookies"]'
        ).first

        async def _handle_consent(locator):
            self.logger.info("Consent banner detected — accepting cookies...")
            try:
                await locator.hover()
                await self._page.wait_for_timeout(300)
                await locator.click(delay=120)
                self.logger.info("Cookies accepted.")
            except PWError as e:
                self.logger.debug(f"Consent click failed: {e}")

        await self._page.add_locator_handler(consent_locator, _handle_consent)


    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(PWError))
    async def _goto_with_retry(self, url, wait_until="domcontentloaded", timeout=10000):
        return await self._page.goto(url, wait_until=wait_until, timeout=timeout)


    async def _get_recs(self, n_recs) -> list[Video]:
        thumbs = await self._page.locator("a.ytLockupMetadataViewModelTitle").all()
        urls = [await x.get_attribute("href") for x in thumbs]
        ids = []

        for url in urls:
            if not url:
                self.logger.debug(f"No recommendations on page.")
                continue
            
            match = re.search(r"/watch\?v=([a-zA-Z0-9_-]{11})", url)
            if match:
                ids.append(match.group(1))
            else:
                self.logger.debug(f"Ignored non-standard recommendation URL: {url}")

        recs = [Video(id) for id in ids]
        self.logger.debug(f"Video IDs found on video page: {[rec.id for rec in recs]}")
        return recs[:n_recs]  

    async def _screenshot(self):
        if self.logger.level == logging.DEBUG:
            error_time = int(time.time())
            screenshot_path = LOG_DIR / f"error_screenshot_{self.logger.name}_{error_time}.png"
            await self._page.screenshot(path=str(screenshot_path))
            self.logger.debug(f"Saved error state screenshot to {screenshot_path}")

    async def get_homepage_recs(self, n_recs = 8):
        assert self._page

        await self._goto_with_retry("https://www.youtube.com/")
        await self._page.wait_for_load_state("domcontentloaded")
        recs = await self._get_recs(n_recs)
        topics = await self._page.locator(".ytChipShapeChip").all_inner_texts()
        self.logger.debug(f"Video IDs found on homepage: {recs}")
        return recs, topics


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(PWError))
    async def play_video(self, vid : Video, n_recs = 8):
        assert self._page

        url = f"https://www.youtube.com/watch?v={vid.id}"
        await self._goto_with_retry(url)
        await self._page.wait_for_load_state("domcontentloaded")

        # check for ads
        ad = self._page.locator(".ytp-ad-module")
        ad_triggered = False
        while await ad.is_visible():
            if not ad_triggered:
                self.logger.debug("Waiting for ad to end.")
                ad_triggered = True
            skip_button = self._page.locator("button.ytp-skip-ad-button")
            if await skip_button.is_visible():
                await skip_button.click()
                self.logger.debug("Skipping ad.")
            await asyncio.sleep(1)

        # check for video errors
        error = self._page.locator("yt-player-error-message-renderer")
        if await error.count() > 0:
            await self._screenshot()
            raise VideoUnavailableException()

        # fetch title to see if video is available
        title_elem = self._page.locator("h1.ytd-watch-metadata yt-formatted-string")
        try:
            await self._page.wait_for_function(
                "el => el && el.getAttribute('title') && el.getAttribute('title').length > 0",
                arg=await title_elem.element_handle(),
                timeout=3000
            )
        except PWTimeoutError:
            await self._screenshot()
            raise VideoUnavailableException()

        title = await title_elem.get_attribute("title")
        vid.title = title
        self.logger.info(f"Playing video: {vid.title}")

        recs = await self._get_recs(n_recs)

        #pause
        await self._page.evaluate("document.querySelector('video')?.pause();")

        return vid, recs


    async def wait_wt(self, time : float):
        #monitor playback time
        """
        wt = 0
        wt_buffer = deque(maxlen=30)
        while wt <= time:
            wt = await self._page.evaluate("document.querySelector('video')?.currentTime ?? 0") # get player time
            wt_buffer.append(wt)

            if len(wt_buffer) > 25:  # if stalled for 25, skip
                if wt_buffer[-26] == wt_buffer[-1]:
                    raise PlaybackException

            await asyncio.sleep(1)
        """
        WATCH_JS = """
        async t => new Promise(resolve => {
            const tryPlay = v => {v.play()};

            const check = () => {
                const v = document.querySelector('video');
                if (!v) return false;
                if (v.ended) { resolve(v.currentTime); return true; }
                if (v.currentTime >= t) { resolve(v.currentTime); return true; }
                if (v.paused || v.readyState < 2) tryPlay(v);
                return false;
            };

            // quick immediate check
            if (check()) return;

            const id = setInterval(() => {
                if (check()) clearInterval(id);
            }, 500);
        })
        """
        try:
            wt = await asyncio.wait_for(self._page.evaluate(WATCH_JS, arg=time), timeout=60)
        except Exception as e:
            await self._screenshot()
            raise PlaybackException
        
        self.logger.debug(f"Playback time: {wt}")
        return wt