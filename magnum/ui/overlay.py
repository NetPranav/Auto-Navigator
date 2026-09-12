"""
Native macOS Screen Overlay for Auto-Navigator.
Features:
- Glowing cyber HUD border around display with camera shutter flash animation.
- Top-Left Live Plan Checklist showing steps and progress ([✓], [⚡], [ ]).
- Bottom-Center Interactive Glassmorphic Question & Confirmation Card.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
import threading
from typing import List, Optional

if __name__ == "__main__":
    import objc
    import Cocoa
    import Quartz
    from AppKit import (
        NSApplication,
        NSWindow,
        NSView,
        NSColor,
        NSRect,
        NSPoint,
        NSSize,
        NSBezierPath,
        NSScreen,
        NSFont,
        NSDictionary,
        NSForegroundColorAttributeName,
        NSFontAttributeName,
        NSTimer,
    )

    class OverlayView(NSView):
        def init(self):
            self = objc.super(OverlayView, self).init()
            if self:
                self.status_text = "AUTO-NAVIGATOR ACTIVE"
                # Modern electric cyber purple
                self.border_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.62, 0.20, 1.0, 0.90)
                self.flash_alpha = 0.0
                
                # Plan Checklist state
                self.plan_steps = []
                self.active_step_idx = 1
                self.completed_step_indices = []
                self.checklist_rect = None
                self.is_checklist_hovered = False

                # Active Background Watchers
                self.watchers = []

                # Antigravity Autopilot Monitor state
                self.autopilot_active = False
                self.autopilot_state = "unknown"
                self.autopilot_state_display = ""
                self.autopilot_emoji = "❓"
                self.autopilot_mode = "full_auto"
                self.autopilot_status = "stopped"
                self.autopilot_scan_count = 0
                self.autopilot_actions_taken = 0
                self.autopilot_roadmap_step = 0
                self.autopilot_roadmap_total = 0
                self.autopilot_extra = ""
                self.autopilot_log = []  # Last N actions

                # Bottom-Center Question Card state
                self.show_question = False
                self.question_id = ""
                self.question_prompt = ""
                self.question_options = ["Approve", "Cancel"]
                self.btn_rects = []

                # Start hover polling timer (checks cursor position every 60ms)
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    0.06, self, objc.selector(self.checkHoverState_, signature=b"v@:@"), None, True
                )

            return self

        @objc.signature(b"v@:@")
        def checkHoverState_(self, timer):
            if not self.checklist_rect or not self.plan_steps:
                return

            try:
                from AppKit import NSEvent
                loc = NSEvent.mouseLocation()
                r = self.checklist_rect
                now_hovered = (
                    r.origin.x <= loc.x <= r.origin.x + r.size.width and
                    r.origin.y <= loc.y <= r.origin.y + r.size.height
                )
                if now_hovered != self.is_checklist_hovered:
                    self.is_checklist_hovered = now_hovered
                    self.setNeedsDisplay_(True)
            except Exception:
                pass

        @objc.signature(b"v@:@")
        def setStatusText_(self, text):
            self.status_text = str(text) if text else ""
            self.setNeedsDisplay_(True)

        @objc.signature(b"v@:@")
        def setPlanData_(self, data):
            if isinstance(data, dict):
                self.plan_steps = list(data.get("steps", []))
                self.active_step_idx = int(data.get("active_idx", 1))
                self.completed_step_indices = list(data.get("completed", []))
                self.setNeedsDisplay_(True)

        @objc.signature(b"v@:@")
        def setWatchersData_(self, data):
            if isinstance(data, dict):
                self.watchers = list(data.get("watchers", []))
                self.setNeedsDisplay_(True)

        @objc.signature(b"v@:@")
        def setAutopilotState_(self, data):
            if isinstance(data, dict):
                self.autopilot_active = True
                self.autopilot_state = str(data.get("state", "unknown"))
                self.autopilot_state_display = str(data.get("state_display", ""))
                self.autopilot_emoji = str(data.get("emoji", "❓"))
                self.autopilot_mode = str(data.get("mode", "full_auto"))
                self.autopilot_status = str(data.get("status", "stopped"))
                self.autopilot_scan_count = int(data.get("scan_count", 0))
                self.autopilot_actions_taken = int(data.get("actions_taken", 0))
                self.autopilot_roadmap_step = int(data.get("roadmap_step", 0))
                self.autopilot_roadmap_total = int(data.get("roadmap_total", 0))
                self.autopilot_extra = str(data.get("extra", ""))
                if self.autopilot_status == "stopped":
                    self.autopilot_active = False
                self.setNeedsDisplay_(True)

        @objc.signature(b"v@:@")
        def addAutopilotLog_(self, data):
            if isinstance(data, dict):
                action_str = str(data.get("action", ""))
                if action_str:
                    self.autopilot_log.append(action_str)
                    if len(self.autopilot_log) > 5:
                        self.autopilot_log = self.autopilot_log[-5:]
                    self.setNeedsDisplay_(True)

        @objc.signature(b"v@:@")
        def showQuestionData_(self, data):
            if isinstance(data, dict):
                self.question_prompt = str(data.get("prompt", ""))
                self.question_id = str(data.get("id", ""))
                self.question_options = list(data.get("options", ["Approve", "Cancel"]))
                self.show_question = True
                if self.window():
                    self.window().setIgnoresMouseEvents_(False)
                self.setNeedsDisplay_(True)

        @objc.signature(b"v@:@")
        def hideQuestion_(self, sender):
            self.show_question = False
            if self.window():
                self.window().setIgnoresMouseEvents_(True)
            self.setNeedsDisplay_(True)

        @objc.signature(b"v@:@")
        def triggerFlash_(self, sender):
            self.flash_alpha = 0.75
            self.setNeedsDisplay_(True)
            self._fade_step()

        def _fade_step(self):
            if self.flash_alpha > 0.05:
                self.flash_alpha -= 0.15
                self.setNeedsDisplay_(True)
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    0.03, self, objc.selector(self.fadeStep_, signature=b"v@:@"), None, False
                )
            else:
                self.flash_alpha = 0.0
                self.setNeedsDisplay_(True)

        @objc.signature(b"v@:@")
        def fadeStep_(self, timer):
            self._fade_step()

        def mouseDown_(self, event):
            if not self.show_question:
                return

            loc = self.convertPoint_fromView_(event.locationInWindow(), None)
            for rect, opt in self.btn_rects:
                if (rect.origin.x <= loc.x <= rect.origin.x + rect.size.width and
                    rect.origin.y <= loc.y <= rect.origin.y + rect.size.height):
                    # Output chosen answer to stdout
                    sys.stdout.write(f"ANSWER:{self.question_id}:{opt}\n")
                    sys.stdout.flush()
                    self.hideQuestion_(None)
                    break

        def drawRect_(self, dirtyRect):
            bounds = self.bounds()
            w = bounds.size.width
            h = bounds.size.height

            # 1. Camera Flash Shutter Pulse
            if self.flash_alpha > 0.01:
                flash_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, self.flash_alpha)
                flash_color.setFill()
                NSBezierPath.fillRect_(bounds)

            # 2. Sleek Deep Purple & Violet Glowing Border (Soft Outside-to-Inside Fade)
            glow_layers = 10
            glow_depth = 20.0
            for i in range(glow_layers):
                d = (i / float(glow_layers)) * glow_depth
                # Soft exponential falloff from 0.42 down to 0.0
                alpha = ((glow_depth - d) / glow_depth) ** 2.2 * 0.42
                r = 0.44 - (d / glow_depth) * 0.08
                g = 0.12 - (d / glow_depth) * 0.04
                b = 0.74 - (d / glow_depth) * 0.14
                step_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, alpha)
                step_color.setStroke()
                
                inset_rect = NSRect(
                    NSPoint(d + 1.0, d + 1.0),
                    NSSize(w - (d + 1.0) * 2.0, h - (d + 1.0) * 2.0)
                )
                inset_path = NSBezierPath.bezierPathWithRect_(inset_rect)
                inset_path.setLineWidth_(2.0)
                inset_path.stroke()

            # Subtle, elegant violet primary rim line
            rim_path = NSBezierPath.bezierPathWithRect_(
                NSRect(NSPoint(1.0, 1.0), NSSize(w - 2.0, h - 2.0))
            )
            rim_path.setLineWidth_(1.5)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.52, 0.18, 0.84, 0.60).setStroke()
            rim_path.stroke()

            # 3. Corner Brackets (Soft Violet Accent)
            corner_len = 34.0
            corner_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.60, 0.25, 0.90, 0.65)
            corner_color.setStroke()
            corners = [
                [(4.0, 4.0 + corner_len), (4.0, 4.0), (4.0 + corner_len, 4.0)],
                [(w - 4.0 - corner_len, 4.0), (w - 4.0, 4.0), (w - 4.0, 4.0 + corner_len)],
                [(4.0, h - 4.0 - corner_len), (4.0, h - 4.0), (4.0 + corner_len, h - 4.0)],
                [(w - 4.0 - corner_len, h - 4.0), (w - 4.0, h - 4.0), (w - 4.0, h - 4.0 - corner_len)],
            ]
            for pts in corners:
                cpath = NSBezierPath.bezierPath()
                cpath.setLineWidth_(3.0)
                cpath.moveToPoint_(NSPoint(pts[0][0], pts[0][1]))
                cpath.lineToPoint_(NSPoint(pts[1][0], pts[1][1]))
                cpath.lineToPoint_(NSPoint(pts[2][0], pts[2][1]))
                cpath.stroke()

            # 4. Top Status Badge Pill
            if self.status_text:
                badge_w = 360.0
                badge_h = 30.0
                badge_x = (w - badge_w) / 2.0
                badge_y = h - badge_h - 6.0
                badge_rect = NSRect(NSPoint(badge_x, badge_y), NSSize(badge_w, badge_h))
                
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.06, 0.03, 0.12, 0.88).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(badge_rect, 8.0, 8.0).fill()
                
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.50, 0.20, 0.78, 0.60).setStroke()
                b_outline = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(badge_rect, 8.0, 8.0)
                b_outline.setLineWidth_(1.2)
                b_outline.stroke()

                display_str = f"⚡ {self.status_text}"
                font = NSFont.boldSystemFontOfSize_(12.0)
                attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                    [font, NSColor.whiteColor()],
                    [NSFontAttributeName, NSForegroundColorAttributeName]
                )
                str_obj = Cocoa.NSString.stringWithString_(display_str)
                str_size = str_obj.sizeWithAttributes_(attrs)
                str_obj.drawAtPoint_withAttributes_(
                    NSPoint(badge_x + (badge_w - str_size.width) / 2.0, badge_y + (badge_h - str_size.height) / 2.0),
                    attrs
                )

            # 4b. Active Background Watchers Badge (Top-Right Pill)
            if self.watchers:
                watcher_summary = " • ".join(str(w) for w in self.watchers)
                if len(watcher_summary) > 42:
                    watcher_summary = watcher_summary[:39] + "..."
                display_w_str = f"👁️ WATCHING: {watcher_summary}"

                wb_w = 320.0
                wb_h = 28.0
                wb_x = w - wb_w - 24.0
                wb_y = h - wb_h - 10.0
                wb_rect = NSRect(NSPoint(wb_x, wb_y), NSSize(wb_w, wb_h))

                # Purple/Dark glass pill background
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.04, 0.16, 0.94).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(wb_rect, 7.0, 7.0).fill()

                # Glowing neon purple border
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.72, 0.32, 1.0, 0.90).setStroke()
                wb_outline = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(wb_rect, 7.0, 7.0)
                wb_outline.setLineWidth_(1.5)
                wb_outline.stroke()

                w_font = NSFont.boldSystemFontOfSize_(11.0)
                w_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                    [w_font, NSColor.colorWithCalibratedRed_green_blue_alpha_(0.85, 0.55, 1.0, 1.0)],
                    [NSFontAttributeName, NSForegroundColorAttributeName]
                )
                w_str_obj = Cocoa.NSString.stringWithString_(display_w_str)
                w_str_size = w_str_obj.sizeWithAttributes_(w_attrs)
                w_str_obj.drawAtPoint_withAttributes_(
                    NSPoint(wb_x + (wb_w - w_str_size.width) / 2.0, wb_y + (wb_h - w_str_size.height) / 2.0),
                    w_attrs
                )

            # 4c. Antigravity Autopilot Monitor Panel (Top-Right, above watchers badge)
            if self.autopilot_active:
                ap_w = 340.0
                ap_line_h = 18.0
                log_lines = min(len(self.autopilot_log), 3)
                ap_h = 100.0 + log_lines * ap_line_h
                ap_x = w - ap_w - 24.0
                # Position above the watcher badge if present
                ap_y_offset = 44.0 if self.watchers else 10.0
                ap_y = h - ap_h - ap_y_offset
                ap_rect = NSRect(NSPoint(ap_x, ap_y), NSSize(ap_w, ap_h))

                # Deep dark glass background with electric blue accent
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.04, 0.06, 0.14, 0.94).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(ap_rect, 10.0, 10.0).fill()

                # Choose border color based on status
                status_lower = self.autopilot_status.lower()
                if status_lower in ("acting", "executing_roadmap"):
                    border_r, border_g, border_b = 0.0, 1.0, 0.5  # Electric green
                elif status_lower == "waiting":
                    border_r, border_g, border_b = 1.0, 0.8, 0.0  # Amber
                elif status_lower == "error":
                    border_r, border_g, border_b = 1.0, 0.2, 0.2  # Red
                elif status_lower == "paused":
                    border_r, border_g, border_b = 0.6, 0.6, 0.7  # Grey
                else:
                    border_r, border_g, border_b = 0.2, 0.6, 1.0  # Electric blue

                NSColor.colorWithCalibratedRed_green_blue_alpha_(border_r, border_g, border_b, 0.90).setStroke()
                ap_outline = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(ap_rect, 10.0, 10.0)
                ap_outline.setLineWidth_(1.8)
                ap_outline.stroke()

                # Header: "🤖 ANTIGRAVITY AUTOPILOT"
                ap_header = "🤖 ANTIGRAVITY AUTOPILOT"
                aph_font = NSFont.boldSystemFontOfSize_(10.5)
                aph_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                    [aph_font, NSColor.colorWithCalibratedRed_green_blue_alpha_(border_r, border_g, border_b, 1.0)],
                    [NSFontAttributeName, NSForegroundColorAttributeName]
                )
                Cocoa.NSString.stringWithString_(ap_header).drawAtPoint_withAttributes_(
                    NSPoint(ap_x + 12.0, ap_y + ap_h - 22.0), aph_attrs
                )

                # Mode badge (right-aligned in header)
                mode_str = self.autopilot_mode.upper().replace("_", " ")
                mode_font = NSFont.boldSystemFontOfSize_(9.0)
                mode_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                    [mode_font, NSColor.colorWithCalibratedRed_green_blue_alpha_(0.7, 0.75, 0.85, 0.9)],
                    [NSFontAttributeName, NSForegroundColorAttributeName]
                )
                mode_ns = Cocoa.NSString.stringWithString_(mode_str)
                mode_size = mode_ns.sizeWithAttributes_(mode_attrs)
                mode_ns.drawAtPoint_withAttributes_(
                    NSPoint(ap_x + ap_w - mode_size.width - 14.0, ap_y + ap_h - 21.0), mode_attrs
                )

                # State display line
                state_display = self.autopilot_state_display or f"{self.autopilot_emoji} {self.autopilot_state.upper()}"
                if len(state_display) > 40:
                    state_display = state_display[:37] + "..."
                sd_font = NSFont.boldSystemFontOfSize_(11.0)
                sd_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                    [sd_font, NSColor.whiteColor()],
                    [NSFontAttributeName, NSForegroundColorAttributeName]
                )
                Cocoa.NSString.stringWithString_(state_display).drawAtPoint_withAttributes_(
                    NSPoint(ap_x + 12.0, ap_y + ap_h - 42.0), sd_attrs
                )

                # Stats line: "Scans: N  |  Actions: N"
                stats_str = f"Scans: {self.autopilot_scan_count}  |  Actions: {self.autopilot_actions_taken}"
                if self.autopilot_roadmap_total > 0:
                    stats_str += f"  |  Step {self.autopilot_roadmap_step}/{self.autopilot_roadmap_total}"
                stats_font = NSFont.systemFontOfSize_(9.5)
                stats_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                    [stats_font, NSColor.colorWithCalibratedRed_green_blue_alpha_(0.6, 0.65, 0.75, 0.9)],
                    [NSFontAttributeName, NSForegroundColorAttributeName]
                )
                Cocoa.NSString.stringWithString_(stats_str).drawAtPoint_withAttributes_(
                    NSPoint(ap_x + 12.0, ap_y + ap_h - 58.0), stats_attrs
                )

                # Roadmap progress bar (if active)
                if self.autopilot_roadmap_total > 0:
                    bar_x = ap_x + 12.0
                    bar_y = ap_y + ap_h - 72.0
                    bar_w = ap_w - 24.0
                    bar_h = 6.0
                    progress = self.autopilot_roadmap_step / max(self.autopilot_roadmap_total, 1)

                    # Background
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.15, 0.18, 0.25, 0.8).setFill()
                    NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                        NSRect(NSPoint(bar_x, bar_y), NSSize(bar_w, bar_h)), 3.0, 3.0
                    ).fill()

                    # Fill
                    if progress > 0:
                        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.9, 0.5, 0.9).setFill()
                        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                            NSRect(NSPoint(bar_x, bar_y), NSSize(bar_w * progress, bar_h)), 3.0, 3.0
                        ).fill()

                # Action log (last 3 entries)
                if self.autopilot_log:
                    log_start_y = ap_y + ap_h - 82.0
                    if self.autopilot_roadmap_total > 0:
                        log_start_y -= 10.0
                    log_font = NSFont.systemFontOfSize_(9.0)
                    log_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.55, 0.6, 0.7, 0.85)
                    log_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                        [log_font, log_color],
                        [NSFontAttributeName, NSForegroundColorAttributeName]
                    )
                    for li, log_entry in enumerate(reversed(self.autopilot_log[-3:])):
                        entry_str = log_entry
                        if len(entry_str) > 42:
                            entry_str = entry_str[:39] + "..."
                        Cocoa.NSString.stringWithString_(f"  › {entry_str}").drawAtPoint_withAttributes_(
                            NSPoint(ap_x + 8.0, log_start_y - li * ap_line_h), log_attrs
                        )

            # 5. Top-Left Plan Checklist Widget (30% opacity default, 95% on hover)
            if self.plan_steps:
                card_w = 340.0
                line_h = 24.0
                card_h = 42.0 + len(self.plan_steps) * line_h
                card_x = 24.0
                card_y = h - card_h - 24.0
                card_rect = NSRect(NSPoint(card_x, card_y), NSSize(card_w, card_h))
                self.checklist_rect = card_rect

                # 30% opacity when cursor is not hovering, 95% opacity on hover!
                card_alpha = 0.95 if self.is_checklist_hovered else 0.30

                # Background Glassmorphism (dark purple tint)
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.04, 0.16, card_alpha * 0.92).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(card_rect, 10.0, 10.0).fill()
                
                # Glowing Purple Border
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.68, 0.28, 1.0, card_alpha * 0.85).setStroke()
                c_outline = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(card_rect, 10.0, 10.0)
                c_outline.setLineWidth_(1.5)
                c_outline.stroke()

                # Card Header
                header_str = "📋 AI PLAN CHECKLIST"
                h_font = NSFont.boldSystemFontOfSize_(11.0)
                h_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                    [h_font, NSColor.colorWithCalibratedRed_green_blue_alpha_(0.78, 0.45, 1.0, card_alpha)],
                    [NSFontAttributeName, NSForegroundColorAttributeName]
                )
                Cocoa.NSString.stringWithString_(header_str).drawAtPoint_withAttributes_(
                    NSPoint(card_x + 14.0, card_y + card_h - 28.0), h_attrs
                )

                # Steps
                for i, step_text in enumerate(self.plan_steps, start=1):
                    step_y = card_y + card_h - 32.0 - (i * line_h)
                    
                    if i in self.completed_step_indices:
                        prefix = "✓"
                        color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.4, 0.9, 0.4, card_alpha)
                        font = NSFont.systemFontOfSize_(11.0)
                    elif i == self.active_step_idx:
                        prefix = "⚡"
                        color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.9, 0.2, card_alpha)
                        font = NSFont.boldSystemFontOfSize_(11.5)
                    else:
                        prefix = "○"
                        color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.7, 0.75, 0.85, card_alpha * 0.85)
                        font = NSFont.systemFontOfSize_(11.0)

                    item_str = f" {prefix} {step_text}"
                    if len(item_str) > 38:
                        item_str = item_str[:35] + "..."
                    
                    attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                        [font, color],
                        [NSFontAttributeName, NSForegroundColorAttributeName]
                    )
                    Cocoa.NSString.stringWithString_(item_str).drawAtPoint_withAttributes_(
                        NSPoint(card_x + 10.0, step_y), attrs
                    )

            # 6. Bottom-Center Interactive Question Card
            if self.show_question:
                qcard_w = 620.0
                qcard_h = 175.0
                qcard_x = (w - qcard_w) / 2.0
                qcard_y = 36.0
                qcard_rect = NSRect(NSPoint(qcard_x, qcard_y), NSSize(qcard_w, qcard_h))

                # Background Glassmorphism
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.04, 0.07, 0.13, 0.96).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(qcard_rect, 12.0, 12.0).fill()
                
                # Glowing Emerald/Cyan Border
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 1.0, 0.7, 0.95).setStroke()
                q_outline = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(qcard_rect, 12.0, 12.0)
                q_outline.setLineWidth_(2.0)
                q_outline.stroke()

                # Card Header
                qhead_str = "🤖 AUTO-NAVIGATOR • ACTION CONFIRMATION"
                qh_font = NSFont.boldSystemFontOfSize_(11.0)
                qh_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                    [qh_font, NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 1.0, 0.7, 1.0)],
                    [NSFontAttributeName, NSForegroundColorAttributeName]
                )
                Cocoa.NSString.stringWithString_(qhead_str).drawAtPoint_withAttributes_(
                    NSPoint(qcard_x + 18.0, qcard_y + qcard_h - 26.0), qh_attrs
                )

                # Prompt Content
                body_str = self.question_prompt
                if len(body_str) > 220:
                    body_str = body_str[:217] + "..."
                
                b_font = NSFont.systemFontOfSize_(12.0)
                b_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                    [b_font, NSColor.whiteColor()],
                    [NSFontAttributeName, NSForegroundColorAttributeName]
                )
                Cocoa.NSString.stringWithString_(body_str).drawInRect_withAttributes_(
                    NSRect(NSPoint(qcard_x + 18.0, qcard_y + 50.0), NSSize(qcard_w - 36.0, 90.0)),
                    b_attrs
                )

                # Draw Interactive 1-Click Buttons
                self.btn_rects = []
                btn_w = 140.0
                btn_h = 32.0
                total_btns_w = len(self.question_options) * (btn_w + 14.0)
                start_btn_x = qcard_x + (qcard_w - total_btns_w) / 2.0
                btn_y = qcard_y + 12.0

                for j, opt in enumerate(self.question_options):
                    bx = start_btn_x + j * (btn_w + 14.0)
                    brect = NSRect(NSPoint(bx, btn_y), NSSize(btn_w, btn_h))
                    self.btn_rects.append((brect, opt))

                    # Button Fill
                    if "approve" in opt.lower() or "yes" in opt.lower() or "post" in opt.lower():
                        bg_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.7, 0.4, 0.9)
                    elif "cancel" in opt.lower() or "reject" in opt.lower() or "no" in opt.lower():
                        bg_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.8, 0.2, 0.2, 0.9)
                    else:
                        bg_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.5, 0.8, 0.9)

                    bg_color.setFill()
                    btn_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(brect, 6.0, 6.0)
                    btn_path.fill()

                    # Button Label
                    btn_label = f"✓ {opt}" if "approve" in opt.lower() else opt
                    btn_font = NSFont.boldSystemFontOfSize_(11.5)
                    btn_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                        [btn_font, NSColor.whiteColor()],
                        [NSFontAttributeName, NSForegroundColorAttributeName]
                    )
                    str_b = Cocoa.NSString.stringWithString_(btn_label)
                    ssize = str_b.sizeWithAttributes_(btn_attrs)
                    str_b.drawAtPoint_withAttributes_(
                        NSPoint(bx + (btn_w - ssize.width) / 2.0, btn_y + (btn_h - ssize.height) / 2.0),
                        btn_attrs
                    )


    def run_overlay_process():
        app = NSApplication.sharedApplication()
        screen = NSScreen.mainScreen()
        frame = screen.frame()

        style_mask = Cocoa.NSWindowStyleMaskBorderless
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style_mask, Cocoa.NSBackingStoreBuffered, False
        )
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setLevel_(Quartz.kCGOverlayWindowLevel)
        window.setIgnoresMouseEvents_(True)
        window.setCollectionBehavior_(
            Cocoa.NSWindowCollectionBehaviorCanJoinAllSpaces | Cocoa.NSWindowCollectionBehaviorStationary
        )

        view = OverlayView.alloc().init()
        view.setFrame_(frame)
        window.setContentView_(view)
        window.orderFrontRegardless()

        # Native macOS Status Bar Item (Menu Bar Icon ⚡)
        status_item = None
        watcher_item = None
        try:
            from AppKit import NSStatusBar, NSMenu, NSMenuItem
            status_bar = NSStatusBar.systemStatusBar()
            status_item = status_bar.statusItemWithLength_(-1)
            btn = status_item.button()
            if btn:
                btn.setTitle_("⚡")

            sb_menu = NSMenu.alloc().init()
            t_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("⚡ Magnum AI Assistant", None, "")
            t_item.setEnabled_(False)
            sb_menu.addItem_(t_item)
            sb_menu.addItem_(NSMenuItem.separatorItem())

            watcher_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("👁️ Active Watchers: None", None, "")
            watcher_item.setEnabled_(False)
            sb_menu.addItem_(watcher_item)

            quick_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("⚡ Quick Launcher (⌥ + Space)", None, "")
            quick_item.setEnabled_(False)
            sb_menu.addItem_(quick_item)

            sb_menu.addItem_(NSMenuItem.separatorItem())
            q_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit Overlay", "terminate:", "q")
            sb_menu.addItem_(q_item)

            status_item.setMenu_(sb_menu)
        except Exception:
            pass

        # IPC listener thread reading stdin commands
        def listen_stdin():
            for line in sys.stdin:
                line_str = line.strip()
                if not line_str:
                    continue
                if line_str == "FLASH":
                    try:
                        subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"], stderr=subprocess.DEVNULL)
                    except Exception:
                        pass
                    view.performSelectorOnMainThread_withObject_waitUntilDone_(
                        objc.selector(view.triggerFlash_, signature=b"v@:@"), None, False
                    )
                elif line_str.startswith("STATUS "):
                    status = line_str.split("STATUS ", 1)[1]
                    view.performSelectorOnMainThread_withObject_waitUntilDone_(
                        objc.selector(view.setStatusText_, signature=b"v@:@"), status, False
                    )
                elif line_str.startswith("PLAN "):
                    try:
                        payload = json.loads(line_str.split("PLAN ", 1)[1])
                        view.performSelectorOnMainThread_withObject_waitUntilDone_(
                            objc.selector(view.setPlanData_, signature=b"v@:@"),
                            payload, False
                        )
                    except Exception as e:
                        pass
                elif line_str.startswith("WATCHERS "):
                    try:
                        payload = json.loads(line_str.split("WATCHERS ", 1)[1])
                        view.performSelectorOnMainThread_withObject_waitUntilDone_(
                            objc.selector(view.setWatchersData_, signature=b"v@:@"),
                            payload, False
                        )
                        if status_item and watcher_item:
                            watchers_list = payload.get("watchers", [])
                            btn = status_item.button()
                            if watchers_list:
                                if btn:
                                    btn.setTitle_(f"⚡ ({len(watchers_list)} 👁️)")
                                watcher_item.setTitle_(f"👁️ Active Watchers: {len(watchers_list)}")
                            else:
                                if btn:
                                    btn.setTitle_("⚡")
                                watcher_item.setTitle_("👁️ Active Watchers: None")
                    except Exception as e:
                        pass
                elif line_str.startswith("ASK_QUESTION "):
                    try:
                        payload = json.loads(line_str.split("ASK_QUESTION ", 1)[1])
                        view.performSelectorOnMainThread_withObject_waitUntilDone_(
                            objc.selector(view.showQuestionData_, signature=b"v@:@"),
                            payload, False
                        )
                    except Exception as e:
                        pass
                elif line_str == "HIDE_QUESTION":
                    view.performSelectorOnMainThread_withObject_waitUntilDone_(
                        objc.selector(view.hideQuestion_, signature=b"v@:@"), None, False
                    )
                elif line_str.startswith("AUTOPILOT_STATE "):
                    try:
                        payload = json.loads(line_str.split("AUTOPILOT_STATE ", 1)[1])
                        view.performSelectorOnMainThread_withObject_waitUntilDone_(
                            objc.selector(view.setAutopilotState_, signature=b"v@:@"),
                            payload, False
                        )
                    except Exception:
                        pass
                elif line_str.startswith("AUTOPILOT_LOG "):
                    try:
                        payload = json.loads(line_str.split("AUTOPILOT_LOG ", 1)[1])
                        view.performSelectorOnMainThread_withObject_waitUntilDone_(
                            objc.selector(view.addAutopilotLog_, signature=b"v@:@"),
                            payload, False
                        )
                    except Exception:
                        pass
                elif line_str == "QUIT":
                    app.terminate_(None)
                    break

        t = threading.Thread(target=listen_stdin, daemon=True)
        t.start()

        app.run()

    run_overlay_process()
