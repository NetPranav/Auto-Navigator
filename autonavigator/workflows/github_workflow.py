"""
GitHub repository navigation, discovery, and clipboard copy workflow.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional
from rich.console import Console

from autonavigator.drivers.base import BaseDriver
from autonavigator.drivers.headless_driver import HeadlessDriver
from autonavigator.hitl.hitl_handler import HitlHandler
from autonavigator.intelligence.nim_client import NimClient

logger = logging.getLogger(__name__)
console = Console()


class GitHubWorkflow:
    """Automates finding a repository (e.g. op_celestia from overxpowered) and copying its clone URL / content to clipboard."""

    def __init__(
        self,
        driver: BaseDriver,
        nim_client: Optional[NimClient] = None,
        hitl_handler: Optional[HitlHandler] = None,
    ) -> None:
        self.driver = driver
        self.nim_client = nim_client or NimClient()
        self.hitl_handler = hitl_handler or HitlHandler()

    async def run(self, owner: str = "overxpowered", repo_name: str = "op_celestia") -> bool:
        """Navigate to GitHub repo and copy URL to clipboard."""
        console.print(
            f"[bold cyan]🔍 Auto-Navigator:[/bold cyan] Searching GitHub for [bold yellow]{owner}/{repo_name}[/bold yellow]..."
        )

        target_repo_url = f"https://github.com/{owner}/{repo_name}"
        clone_url = f"https://github.com/{owner}/{repo_name}.git"

        # Step 1: Navigate to repo directly or user profile
        await self.driver.navigate(target_repo_url)

        if isinstance(self.driver, HeadlessDriver):
            page = self.driver.page

            # If direct repo gave 404, check user repositories
            if "Page not found" in await page.title() or "404" in page.url:
                console.print(f"[bold yellow]⚠️ Direct repo not found, searching {owner}'s repositories...[/bold yellow]")
                await self.driver.navigate(f"https://github.com/{owner}?tab=repositories")
                # Look for matching repo link
                repo_link = page.locator(f"a[href*='/{owner}/']").filter(has_text=repo_name).first
                if await repo_link.count() > 0:
                    href = await repo_link.get_attribute("href")
                    if href:
                        target_repo_url = f"https://github.com{href}" if href.startswith("/") else href
                        clone_url = f"{target_repo_url}.git"
                        await self.driver.navigate(target_repo_url)

            # Step 2: Try interacting with the '<> Code' clone button if on page
            try:
                code_btn = page.locator("get-repo, button:has-text('Code'), summary:has-text('Code')").first
                if await code_btn.is_visible():
                    await code_btn.click()
                    await asyncio.sleep(0.5)
                    # Look for clone input or copy button
                    copy_btn = page.locator("button[aria-label*='Copy to clipboard'], clipboard-copy").first
                    if await copy_btn.is_visible():
                        await copy_btn.click()
                        await asyncio.sleep(0.5)
            except Exception as e:
                logger.debug(f"Code dropdown interaction: {e}")

        # Step 3: Copy to clipboard
        await self.driver.set_clipboard(clone_url)
        await self.driver.screenshot("github_repo_view.png")

        # Step 4: Notification and completion
        self.hitl_handler.notify(
            "GitHub Repository Copied!",
            f"{owner}/{repo_name} clone URL copied to clipboard: {clone_url}",
        )
        console.print(f"[bold green]📋 Copied to Clipboard:[/bold green] [bold white]{clone_url}[/bold white]")
        return True
