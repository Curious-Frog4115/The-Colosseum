---
name: Debugging
description: Systematic workflow for finding and fixing bugs in code, with least-effort verification.
---
When debugging code (user-provided, workspace files, or tool output):

1. Read the error/output carefully first. Extract the EXACT error message and the
   failing line before proposing anything.
2. Form a hypothesis about the root cause — do not shotgun fixes.
3. For Python, run the failing snippet in run_python to reproduce, then fix.
   For web JS/CSS, read the file via read_file, trace the relevant function.
4. Change ONE thing at a time. After each change, re-run the reproduction to prove
   the fix. Never claim success without confirming the error is gone.
5. Check the usual suspects in order: name typos, wrong variable/arg order, off-by-one
   in loops/ranges, async vs sync misuse, missing imports, case sensitivity,
   string vs int comparisons, and stale cached values.
6. If stuck after two real attempts, say what you tried and ask a targeted question
   (ask_user) rather than guessing.
7. Report: what was wrong, what you changed, and how you verified it works.