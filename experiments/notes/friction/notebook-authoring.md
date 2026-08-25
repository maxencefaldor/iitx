# Notebook authoring friction

- **Format before executing.** Ruff lints and formats `.ipynb` sources in this repo.
  The working order is: build the notebook → `uv run ruff format` + `ruff check` →
  execute → commit. Formatting after execution is source-only and does not invalidate
  outputs, but catching an E501 *after* a 30-minute run is a waste; catching it after
  a commit means CI fails (it did — e00 was amended).
- **The soft surrogate's temperature is not "small" until ~0.01.** e00: the softmax
  over 22 cuts leaves the bias plateaued at ≈ 0.06 ibits for τ ≥ 0.03. Any future
  experiment quoting a soft value at τ = 0.05 as "approximately exact" is wrong.
- **Oracle subprocess paths.** The PyPhi check runs `uv run python …` with
  `cwd=tests/oracle/generate` (the pinned env, and PyPhi reads `pyphi_config.yml`
  from the CWD, which the generate dir deliberately lacks). Pass *absolute* paths for
  the script and payloads — relative paths need three `..` levels from that cwd and
  broke on the first try.
- **`MPLBACKEND=Agg` suppresses inline figures under nbclient** (earlier session
  finding, re-recorded here because it will bite again): execute notebooks without
  overriding the backend or the committed run has no images.
