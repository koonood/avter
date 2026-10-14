AFTER v8 — FREE REPORT + OPTIONAL AI ANALYSIS

Flow:
1. User uploads chat archive.
2. Server parses the FULL archive and computes the free report locally — no AI call.
3. report.html opens immediately.
4. Free report includes message volume over time, two participant curves, participation split, monthly breakdown, activity, response gaps, message length, observable conflict/repair/future/affection markers and recurring rituals.
5. The extended AI analysis is shown as an optional €2.99 feature.
6. During testing, the button works WITHOUT payment and starts the existing deep AI analysis.
7. Later, payment gating should be inserted in /ai-analyze/{job_id}; the free report should remain unchanged.

Important: AI is NOT started during /upload anymore.

Run:
  pip install -r requirements.txt
  uvicorn main:app --reload
