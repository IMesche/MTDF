# MANIFEST.sha256 notes

## Purpose

`MANIFEST.sha256` is the authoritative integrity record for the
`gravity/output/` artefact set. It is a machine-clean SHA-256 checksum file,
intended to be consumed by:

```bash
sha256sum -c MANIFEST.sha256
```

The expected output is reproduced in `VERIFY_MANIFEST.txt`, which is regenerated
at the same time as the manifest and should be treated as a frozen witness of
the verification pass at release time.

## Regeneration history

### 2026-04-24 regeneration

The gravity-sector artefact manifest was regenerated on 2026-04-24 to match the
canonical submission package. The update follows a metadata-only JSON
provenance addition dated 2026-04-03, in which every `gravity/output/*.json`
file gained a leading `_meta` block of the form:

```json
"_meta": {
  "author": "Ingo Mesche",
  "affiliation": "Independent Researcher, Malta",
  "framework": "MTDF V74"
}
```

The PNG figure artefacts were unchanged. The numerical payload of every JSON
file was unchanged (the `_meta` block is pure provenance metadata). The
regenerated manifest is the authoritative integrity record for this submission
package.

After the regeneration, all 97 entries verify cleanly (see
`VERIFY_MANIFEST.txt`).

The prior manifest (`MANIFEST.sha256`, generated 2026-02-18) has been archived
outside the submission tree for audit purposes; it does not ship with the
release package.

## What the manifest covers

97 output artefacts across `gravity/output/`:

- 57 PNG figures (rotation curves, RAR, time-delay plots, wide-binary
  distributions, elliptical dispersions, etc.)
- 40 JSON result files and per-step `manifest.json` descriptors
- No source files, no data files, no derived intermediates outside
  `gravity/output/`

The manifest does **not** cover `papers/`, `validation/`, `gpu_validation/`,
or `class_mtdf/`, which are published under their own change-control regimes
(git history, bibliographic DOI, and CLASS upstream release notes
respectively).

## Verification procedure

From the repository root:

```bash
cd gravity
sha256sum -c MANIFEST.sha256
```

Expected: all 97 lines report `OK`. Any `FAILED` line indicates either a
modified artefact or an out-of-sync manifest. In the latter case, regenerate
by re-running the gravity-sector pipeline with a frozen parameter set and
re-hashing; in the former, investigate before submission.

## Freeze policy

After a manifest is regenerated and verified, the covered artefacts must not
be modified. Any subsequent edit invalidates the submission integrity record
and requires a new manifest regeneration, a new verification pass, and a note
in this file.
