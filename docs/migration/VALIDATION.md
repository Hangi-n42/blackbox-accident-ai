# Migration validation

2026-09-16 Windows checks:

- Core import, CPU 144-feature calculation, PyAV synthetic encode/decode and portable path smoke: passed.
- Archived native-video report/review SHA bindings and contact metric replay: passed, V6 4/9 versus candidate 1/9.
- Both review UI Node tests: passed. VM/controlled decode tests, not browser visual QA.
- Migration and frozen release Python syntax: 24 files passed.
- External resume-evidence verification: 22,357 files, no missing files or different SHA256.
- macOS14 ARM64 CPython3.12 wheel resolution: 42 exact wheels resolved. Native Mac execution is a separate GitHub Actions check.
- Tracked-worktree credential pattern scan: 1,670 files, 85,513,868 bytes; 1,661 text files scanned, zero findings, zero case/Unicode collisions at that point. New metadata/doc additions follow this snapshot.
- Original remote history: only Initial commit with LICENSE and .gitignore. Remote and local main matched before migration commit; no tracked deletions.

Limitations: heuristic secret scanning cannot prove absence of all possible secrets. No full competition inference, Mac MPS model evaluation, browser draft migration, or independent accuracy experiment was performed. The external optional archive tier records sizes, not SHA integrity. The GitHub Actions workflow checks a clean ARM64 install without private credentials or external models/data.

A permission warning from elevated Git concerned research/stage1/stage1_contract_f2bsqec4. Its two MP4 copies are intentionally excluded from Git and are present in the hashed external manifest; no source was lost. A migration metadata generator's Windows cp949 decode error was corrected by specifying UTF-8; subsequent generation passed.

First native macOS run 34997893015 failed installation because host-Windows cross-resolution omitted the macOS-conditional hf-xet dependency. Added hf-xet1.6.0 and its official PyPI ARM64 wheel SHA; lock now contains43 packages. This is a migration dependency defect, not a preexisting model failure.

Native macOS run34998120644 passed ARM64 dependency installation, V6 source bootstrap, CPU/import/codec smoke and archived metric replay. Review test failed because cases.js had been excluded with video assets. Included its existing approximately2MB metadata unchanged (no images) so original UI tests work after clone; the original external capture also contains this now-Git-managed file.
