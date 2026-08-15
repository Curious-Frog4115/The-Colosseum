---
name: Web App Builder
description: Build, publish and iterate on single-file HTML/CSS/JS web apps served instantly as live pages.
---
When the user asks for a web app, tool, dashboard, game, page or prototype:

1. Build it as ONE self-contained HTML file (CSS + JS inline) and publish it with
   create_file -> the returned live_url is the app. Give the user that exact path.
2. Confirm it works before delivering: run a quick static sanity check mentally
   (script parses, element ids match, no undefined function calls).
3. Prefer plain modern CSS (flexbox/grid, CSS variables) and vanilla JS. No build
   step, no external frameworks unless the user explicitly asks.
4. Match the visual theme of the site you're in (dark, warm amber accents) unless
   the user wants something else.
5. If the user reports a bug, use edit_file to fix the exact spot (read_file first
   to find the unique 'old' snippet), then publish an updated file.
6. After every change, re-state the live_url so the user can reload.