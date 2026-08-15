---
name: novm
description: Drive the user's real NoVM XFCE workstation (full control + vision). Use for tasks on a real Linux desktop: install/run apps, browse, create files, run commands.
---

# NoVM Workstation Operations

The user has a REAL remote Linux (XFCE) desktop. You control it end-to-end.
Sessions are permanent — REUSE the existing session id, never create new ones
unless none exist or the user asks.

## Workflow

1. `vm_list` — see what sessions exist.
2. If none: `vm_create` (max 2 sessions) then `vm_start`, wait ~15s, `vm_status` until running.
3. `vm_connect` — give the user the viewer link so they can watch.
4. `vm_see` — look at the screen (vision model describes it).
5. Drive it with `vm_click` / `vm_key` / `vm_exec`; verify every action with `vm_see`.

## Desktop map (1280x720, verified live)

- Bottom-left dock: app launcher button at `(64, 704)` — opens Application Finder (xfce4-appfinder) with a search box.
- Installed apps appear as dock icons (after `vm_install_app`).
- Right-click desktop = workspace switcher menu.
- `vm_exec {command}` = open terminal via appfinder (type "xfce4-terminal", Enter), run the command, read output back with vision. This is the reliable full-control path.
- Terminal prompt: `runner@NoVM:.../runner/workspace$` — commands run as user `runner`, home is `/home/runner`.
- Screen captures take ~10-20s (connect + paint + grab). The desktop watermark reads "NOVM WORKSTATION AUTHORIZED SESSION".

## Rules

- NEVER claim success without seeing it on screen (`vm_see`).
- If a session errors: `vm_recover` first, then `vm_restart`.
- Session count is capped at 2 — stop or delete before creating more.
- Apps available: `chromium` (browser), `gedit` (editor), `mousepad` (editor).
- Files: `vm_upload` / `vm_download` / `vm_files` transfer text files to/from the VM home.
- Viewer links expire after ~15 min — refresh with `vm_connect`.
- On Vercel-hosted deploys, VNC tools (screenshot/see/key/click/exec) are unavailable; HTTP-only tools (list/create/start/connect/...) still work — tell the user to open the viewer link and give you feedback instead.