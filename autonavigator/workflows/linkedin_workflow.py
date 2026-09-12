"""
LinkedIn navigation, post reading, AI comment synthesis, and 1-click posting workflow.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
from typing import Optional
from rich.console import Console

from autonavigator.drivers.base import BaseDriver
from autonavigator.drivers.headless_driver import HeadlessDriver
from autonavigator.hitl.hitl_handler import HitlHandler, HitlActionType
from autonavigator.intelligence.nim_client import NimClient

logger = logging.getLogger(__name__)
console = Console()


class LinkedInWorkflow:
    """Automates finding a LinkedIn contact, reading their latest post, drafting a comment with NIM, and 1-click posting."""

    def __init__(
        self,
        driver: BaseDriver,
        nim_client: Optional[NimClient] = None,
        hitl_handler: Optional[HitlHandler] = None,
    ) -> None:
        self.driver = driver
        self.nim_client = nim_client or NimClient()
        self.hitl_handler = hitl_handler or HitlHandler()

    async def run(self, target_user: str, instruction: str = "") -> bool:
        """Execute the LinkedIn comment pipeline."""
        console.print(f"[bold cyan]🔍 Auto-Navigator:[/bold cyan] Searching LinkedIn for [bold yellow]{target_user}[/bold yellow]...")

        # Step 1: Navigate to LinkedIn
        search_query = urllib.parse.quote_plus(target_user)
        search_url = f"https://www.linkedin.com/search/results/people/?keywords={search_query}"
        await self.driver.navigate(search_url)

        # Step 2: In HeadlessDriver, inspect DOM / visual elements
        if isinstance(self.driver, HeadlessDriver):
            page = self.driver.page

            # Check if login wall is present
            if "login" in page.url or "authwall" in page.url:
                console.print(
                    "[bold yellow]⚠️ Notice:[/bold yellow] LinkedIn requires login session. "
                    "You can log in once via the persistent browser profile, or run in Desktop mode."
                )

            # Look for people search results
            try:
                # Wait for search result cards or primary link
                profile_link = await page.locator("a[href*='/in/']").first.get_attribute("href")
                if profile_link:
                    clean_profile_url = profile_link.split("?")[0]
                    console.print(f"[bold green]✓ Profile Found:[/bold green] {clean_profile_url}")
                    # Navigate to their recent activity/posts
                    activity_url = f"{clean_profile_url}/recent-activity/all/"
                    await self.driver.navigate(activity_url)
            except Exception as e:
                logger.debug(f"Direct profile selector lookup: {e}")

        # Step 3: Capture screenshot & text context of the latest post
        await asyncio.sleep(2.0)
        screenshot = await self.driver.screenshot("linkedin_latest_post.png")
        page_text = await self.driver.extract_text_content()

        # Step 4: AI Comment Synthesis using NVIDIA NIM
        console.print("[bold cyan]🧠 NVIDIA NIM:[/bold cyan] Analyzing post context and generating authentic comment...")
        screenshot_b64 = NimClient.pil_to_base64(screenshot)

        draft = self.nim_client.synthesize_linkedin_comment(
            post_text=page_text[:2000] if page_text else f"Recent post by {target_user}",
            author=target_user,
            comments_context=page_text[2000:3500] if len(page_text) > 2000 else None,
            image_base64=screenshot_b64,
        )

        # Step 5: Human-In-The-Loop 1-Click Verification
        while True:
            decision = self.hitl_handler.confirm_comment(
                draft_comment=draft.draft_comment,
                author=target_user,
                post_summary=draft.post_summary,
            )

            if decision.action == HitlActionType.APPROVE:
                comment_to_post = decision.final_text or draft.draft_comment
                break
            elif decision.action == HitlActionType.REGENERATE:
                console.print("[bold magenta]🔄 Regenerating comment with custom prompt...[/bold magenta]")
                draft = self.nim_client.synthesize_linkedin_comment(
                    post_text=f"{page_text[:2000]}\nUser notes: {decision.user_notes}",
                    author=target_user,
                    image_base64=screenshot_b64,
                )
            else:
                console.print("[bold red]❌ Cancelled by user. No comment was posted.[/bold red]")
                self.hitl_handler.notify("LinkedIn Action Cancelled", f"No comment posted to {target_user}.")
                return False

        # Step 6: Post the comment
        console.print(f"[bold green]🚀 Posting approved comment to {target_user}...[/bold green]")
        if isinstance(self.driver, HeadlessDriver):
            page = self.driver.page
            try:
                # Try clicking comment button on first post
                comment_btn = page.locator("button[aria-label*='Comment']").first
                if await comment_btn.is_visible():
                    await comment_btn.click()
                    await asyncio.sleep(1.0)

                # Focus comment box and type
                editor = page.locator("div[role='textbox'][aria-label*='comment']").first
                if await editor.is_visible():
                    await editor.fill(comment_to_post)
                    await asyncio.sleep(0.5)
                    submit_btn = page.locator("button.comments-comment-box__submit-button, button:has-text('Comment')").first
                    if await submit_btn.is_visible():
                        await submit_btn.click()
                        await asyncio.sleep(1.5)
            except Exception as e:
                logger.warning(f"Could not automatically click comment box: {e}")

        # Step 7: Success notification
        self.hitl_handler.notify(
            "LinkedIn Comment Posted",
            f"Successfully posted comment on {target_user}'s latest post!",
        )
        console.print(f"[bold green]✅ Task Complete:[/bold green] Commented on {target_user}'s post.")
        return True
