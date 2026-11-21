# Step 8: fresh-clone reproduction check

Date (UTC): 2026-11-21 11:52. Commit: `3a5b5652dc89d524e4f87acef134c7b779446a8a`. Machine: macOS-27.0.1-arm64-arm-64bit, Python for the check environment: 3.11 (uv). The repository was cloned into a new temporary directory, the environment was created from `requirements.lock` (hashes required), then `make quick` and the test suite were run there. The working copy and its results were not used.

| step | exit code | duration (min) | last output |
|---|---|---|---|
| clone | 0 | 0.0 |  |
| checkout | 0 | 0.0 |  |
| environment | 0 | 0.0 |  |
| install (lock file) | 0 | 0.0 |  |
| install (package) | 0 | 0.0 |  |
| make quick | 0 | 6.8 | 2026-11-21 17:22:14 INFO     sdp.paper.tables: Wrote 15 tables to results_quick/paper/tables / [step 8] done (399 s elapsed) |
| tests | 0 | 0.3 | -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html / 214 passed, 2 skipped, 18 warnings in 14.17s |

Outcome: **passed**. Total: 7.1 min.

Quick-mode reports written in the clone: 37 (`reports_quick/`); the clone's `results/` and `artifacts/` were not created.
The two skipped tests are `test_refit_reproduces_step6[lr]` and `[rf]`, which compare the permuted-label refit with the step 6 sanity artefacts (`artifacts/fidelity/sanity`, git-ignored and not in a fresh clone); they pass in the working copy (216 passed).
