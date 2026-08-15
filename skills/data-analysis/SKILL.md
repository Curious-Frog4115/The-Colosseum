---
name: Data Analysis
description: Analyze datasets, run computations and statistics, and produce clear numeric findings and charts.
---
When working with data (numbers, tables, CSV-ish text, JSON):

1. Clarify the exact question being answered. If ambiguous, ask_user with a concise
   choice instead of guessing wildly.
2. Put the data into run_python (or create a .csv workspace file and read it) and do
   the analysis with plain Python stdlib or numpy/pandas if install_dependency is ok.
3. Show the numbers: compute sums, means, medians, ranges, percentages, correlations
   where meaningful. Always label units.
4. Verify arithmetic with calculate or a second computation path when results are
   surprising — round-trip checks catch errors.
5. For visualization, produce a self-contained HTML file (embedded chart via inline
   SVG/canvas, no CDN) and publish it with create_file; give the live_url.
6. Summarize: the answer to the question, the key numbers, and any caveats about data
   quality. Do not overstate certainty from small samples.