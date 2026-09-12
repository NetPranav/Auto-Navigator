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
                self.border_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.9, 1.0, 0.85)  # Cyan
                self.flash_alpha = 0.0
                
                # Plan Checklist state
                self.plan_steps = []
                self.active_step_idx = 1
                self.completed_step_indices = []

                # Bottom-Center Question Card state
                self.show_question = False
                self.question_id = ""
                self.question_prompt = ""
                self.question_options = ["Approve", "Cancel"]
                self.btn_rects = []

            return self

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

            # 2. Glowing Neon Border
            border_width = 5.0
            border_path = NSBezierPath.bezierPathWithRect_(
                NSRect(NSPoint(border_width / 2.0, border_width / 2.0),
                       NSSize(w - border_width, h - border_width))
            )
            border_path.setLineWidth_(border_width)
            self.border_color.setStroke()
            border_path.stroke()

            # 3. Corner Brackets (Emerald Green)
            corner_len = 36.0
            corner_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 1.0, 0.4, 0.95)
            corner_color.setStroke()
            corners = [
                [(border_width, border_width + corner_len), (border_width, border_width), (border_width + corner_len, border_width)],
                [(w - border_width - corner_len, border_width), (w - border_width, border_width), (w - border_width, border_width + corner_len)],
                [(border_width, h - border_width - corner_len), (border_width, h - border_width), (border_width + corner_len, h - border_width)],
                [(w - border_width - corner_len, h - border_width), (w - border_width, h - border_width), (w - border_width, h - border_width - corner_len)],
            ]
            for pts in corners:
                cpath = NSBezierPath.bezierPath()
                cpath.setLineWidth_(6.0)
                cpath.moveToPoint_(NSPoint(pts[0][0], pts[0][1]))
                cpath.lineToPoint_(NSPoint(pts[1][0], pts[1][1]))
                cpath.lineToPoint_(NSPoint(pts[2][0], pts[2][1]))
                cpath.stroke()

            # 4. Top Status Badge Pill
            if self.status_text:
                badge_w = 360.0
                badge_h = 32.0
                badge_x = (w - badge_w) / 2.0
                badge_y = h - badge_h - border_width
                badge_rect = NSRect(NSPoint(badge_x, badge_y), NSSize(badge_w, badge_h))
                
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.04, 0.08, 0.14, 0.92).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(badge_rect, 8.0, 8.0).fill()
                
                self.border_color.setStroke()
                b_outline = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(badge_rect, 8.0, 8.0)
                b_outline.setLineWidth_(1.5)
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

            # 5. Top-Left Plan Checklist Widget
            if self.plan_steps:
                card_w = 340.0
                line_h = 24.0
                card_h = 42.0 + len(self.plan_steps) * line_h
                card_x = 24.0
                card_y = h - card_h - 24.0
                card_rect = NSRect(NSPoint(card_x, card_y), NSSize(card_w, card_h))

                # Background & border
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.05, 0.09, 0.16, 0.92).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(card_rect, 10.0, 10.0).fill()
                
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.8, 1.0, 0.7).setStroke()
                c_outline = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(card_rect, 10.0, 10.0)
                c_outline.setLineWidth_(1.5)
                c_outline.stroke()

                # Card Header
                header_str = "📋 AI PLAN CHECKLIST"
                h_font = NSFont.boldSystemFontOfSize_(11.0)
                h_attrs = NSDictionary.dictionaryWithObjects_forKeys_(
                    [h_font, NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.9, 1.0, 1.0)],
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
                        color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.4, 0.9, 0.4, 0.9)
                        font = NSFont.systemFontOfSize_(11.0)
                    elif i == self.active_step_idx:
                        prefix = "⚡"
                        color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.9, 0.2, 1.0)
                        font = NSFont.boldSystemFontOfSize_(11.5)
                    else:
                        prefix = "○"
                        color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.7, 0.75, 0.85, 0.7)
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
                elif line_str == "QUIT":
                    app.terminate_(None)
                    break

        t = threading.Thread(target=listen_stdin, daemon=True)
        t.start()

        app.run()

    run_overlay_process()
