# Source-only GitHub repository

The initial repository is private: `Haozong-Zeng/MouseGeoForge`.

New model development lives in `mousegeoforge/` with tests in `tests/`, instead
of growing a separate copy of VGGT. See `derived_repository_structure_research.md`.

The root `.gitignore` uses a top-level allowlist. Datasets, backup, Blender
projects, output and downloaded dependencies stay local. Inside source trees,
model/data binaries, environments, credentials, archives and Zone.Identifier
sidecars are excluded. Ignoring a file does not delete it.

`vggt-omega/` is a source snapshot including the local modifications and original
license. The existing nested `.git` is intentionally not moved or deleted.
Because Git normally treats a nested repository as a gitlink, the initial import
uses explicit audited blob entries. Do not replace it with `git add vggt-omega`
as a gitlink. On a fresh clone it is simply an ordinary directory.

```bash
# Read-only audit of the files eligible for the source snapshot:
python tools/repository_source_audit.py
# Read-only audit of what is actually staged (run before commit/push):
python tools/repository_source_audit.py --cached
# Explicitly stage all audited source files, including nested source additions:
python tools/repository_source_audit.py --stage
git diff --cached --stat
```

`--stage` intentionally stages the current contents of every eligible source
file; review the diff first if you already have partial staging. It does not
remove deleted tracked files; handle intentional deletions separately. Its secret
checks are heuristic, not a guarantee that a human review can be skipped.

Do not force-push or upload large assets as a workaround for these exclusions.
Put portable dataset schemas and retrieval instructions in docs; regenerate the
local dataset catalogs from the locally available assets. Temporal experiment
JSON files marked `design_only` are specifications, not executable trainer configs.
