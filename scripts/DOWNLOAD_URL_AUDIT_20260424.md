# Download URL audit — 2026-04-24

Scope: verification that every `scripts/download_*.sh` fetches from a reachable
public canonical source, with provenance, hashes, and redistribution status.

## Summary

| Source bucket | Status | Notes |
|---|---|---|
| Cosmic Chronometers (Moresco) | FIXED | Old `github.com/music-group/CC_covariance` is 404; migrated to the author's GitLab |
| eBOSS DR16 BAOplus | FIXED | Old `cmbant/CosmoMC/data/` path is 404; migrated to `CobayaSampler/bao_data` |
| BOSS DR12 voids (Mao+2017) | OK | CDS/VizieR primary works; Vanderbilt mirror unreachable, kept as fallback |
| DESI Y1 BAO | DOCUMENTED | Script emits manual-placement instructions (behaviour unchanged) |
| Pantheon+ SN Ia | FIXED | Upstream directory rename `4_DISTANCES_AND_COVARIANCES` → `4_DISTANCES_AND_COVAR` |
| Pittordis+2023 wide binaries | FIXED | Zenodo filename rename; `TARGET_FILE` preserved for downstream compatibility |
| KiDS-1000 WL, Foundation DR1, Planck PR4 lensing, ZTF DR2, DESI voids (VAST) | OK | All upstream URLs 200 as of 2026-04-24 |
| CosmoPower emulators | PLACEHOLDER | Awaiting Zenodo record ID for MTDF release deposit |

## Verified downloads (dry-run in `/home/ingo/MTDF_layer3_scratch/download_dry_run/`)

### Cosmic Chronometers (Moresco et al.)

Primary: `https://gitlab.com/mmoresco/CCcovariance` (`master` branch, `data/` folder) — author-maintained public source.
Citation: Moresco (2022), *Living Reviews in Relativity* 25, 6.
Redistribution: GitLab-hosted by the first author; public read access.

| File | Size | SHA-256 |
|---|---:|---|
| `HzTable_MM_BC03.dat` | 791 B | `32ce92caf251cb60a7a837c71f1856bea2b44fa5c1041f85410d11cb8164da98` |
| `HzTable_MM_M11.dat` | 646 B | `01171e09a416c7aed0e2cbd4a978981ffc0dffa28665ea2c6ee66b2e4fee44ba` |
| `data_MM20.dat` | 799 B | `577ac2f346e346fe7cf94daa7b7000c05d04ebc8a029cda31e0d8643b956a485` |

Historical source (now 404): `github.com/music-group/CC_covariance`.

### eBOSS DR16 BAOplus (Alam+2021)

Primary: `https://github.com/CobayaSampler/bao_data` (`master` branch, repository root) — current verified public mirror/source used by this analysis. The upstream repository states it hosts "eBOSS DR16, SDSS DR7 MGS, and SDSS DR12 data as originally distributed with CosmoMC" and exposes the exact filenames below.
Citation: eBOSS Collaboration, Alam et al. (2021), *Phys. Rev. D* 103, 083533 (arXiv:2007.08991).
Redistribution: maintained by the Cobaya project; public read access.

| File | Size | SHA-256 |
|---|---:|---|
| `sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat` | 258 B | `a098ea4df320ac1c18a9404237a75ae26953e16403a20294beb1d9573be33c56` |
| `sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8_covtot.txt` | 2,045 B | `409cabbf4ccf6993053427f5a34d52e6557f2429c17777267459471180e72f96` |
| `sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat` | 63 B | `cddd6cbbca7dadc910a5e8742f1f2144c066cb347b8ba03ae0bd4876fa06d8ed` |
| `sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8_covtot.txt` | 107 B | `88f844447fb546792769cdf09b4df7b7a7f77a948f02ef371f54a6f7dddb3d41` |

Historical source (now 404): `raw.githubusercontent.com/cmbant/CosmoMC/master/data/`.

### BOSS DR12 voids (Mao+2017)

Primary: `https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits?J/ApJ/835/161/table1.dat` (CDS/VizieR).
Citation: Mao et al. (2017), *ApJ* 835, 161.
Redistribution: Standard CDS/VizieR public access.

| File | Size | SHA-256 (first fetch) |
|---|---:|---|
| `table1.dat` | 146,880 B | `dc23c44474dc97788d57918dafce063dfde9d9a22654857a364006386c73eae0` |

**Note on byte-level stability.** CDS/VizieR writes a serve-time timestamp into the FITS header `DATE` keyword's comment line of the form `Written on 2026-04-24:HH:MM:SS (GMT)`. This causes the SHA-256 to differ between fetches by five bytes at offset ~778, even though the scientific content (all catalogue rows, WCS metadata, numeric data) is byte-identical across fetches. A second dry-run fetched the same file with hash `575c7045b72a...`, differing only in the header timestamp. For MANIFEST-style byte verification, this file should be excluded or the header timestamp stripped before hashing. Reviewers who refetch will see a different hash but identical data.

Fallback: `http://lss.phy.vanderbilt.edu/voids/ZOBOV_voids_BOSS_DR12/table1.dat` — was unreachable at verification (2026-04-24 17:00 UTC). Retained in script as secondary.

## Other scripts, status

### download_pantheonplus.sh

Patched: upstream directory rename on `PantheonPlusSH0ES/DataRelease` main branch, `Pantheon+_Data/4_DISTANCES_AND_COVARIANCES/` → `Pantheon+_Data/4_DISTANCES_AND_COVAR/`. All 5 path occurrences in the script updated. Upstream file URLs now return HTTP 200 for:

- `Pantheon+SH0ES.dat`
- `Pantheon+SH0ES_STAT+SYS.cov`

### download_pittordis.sh

Patched: Zenodo record 7629240 renamed the CSV file in place. Script now requests `CleanedWB_EDR3_Prlx300pc_Gmag20_20230111_Size73087_ZenodoSample.csv` and saves it as the downstream-expected name `pittordis2023_wb.csv`. Zenodo URL returns HTTP 200.

**File contents are byte-identical to the previous release.** Verified by direct download 2026-04-24: the renamed file at record 7629240 produces SHA-256 `826cbb2c73768515b883a4ff56588ab148592d5f5e11fb577045faba1d9bb8d9`, matching the historical hash already recorded in `download_pittordis.sh`. The file is 170,948,678 bytes, 73,088 rows (73,087 data + 1 header), with the expected Gaia EDR3 wide-binary column schema.

### download_cosmopower_models.sh

Placeholder with `<ZENODO_RECORD_ID>` + `<SHA256_*>` tokens. Awaiting the Zenodo record ID for the MTDF submission deposit before activation.

### Other healthy scripts (URLs 200 as of 2026-04-24)

- `download_kids.sh` — KiDS-1000 primary catalogue at `kids.strw.leidenuniv.nl`
- `download_foundation_dr1.sh` — `github.com/djones1040/Foundation_DR1`
- `download_planck.sh` — `github.com/carronj/planck_PR4_lensing`
- `download_desi_voids.sh` — `github.com/DESI-UR/VAST` + `data.desi.lbl.gov`
- `download_ztf_dr2.sh` — `github.com/ZwickyTransientFacility/ztfcosmo`
- `download_bao.sh` DESI block — prints manual-placement guidance if auto-download is unavailable (behaviour unchanged)
- `download_bao.sh` GAMA block — prints manual-placement guidance (behaviour unchanged)
- `download_bao.sh` Planck plik-lite block — prints manual-placement guidance (behaviour unchanged)

## Post-hardening verification (2026-04-24)

After the replacement URLs were in place, `download_bao.sh` was hardened with:

- a `safe_fetch` helper that rejects empty (zero-byte) and HTML-error-page
  responses so a half-failed wget does not leave a corrupted placeholder on
  disk;
- per-file existence-and-size gates for the Cosmic Chronometer trio and the
  DR16 BAOplus quartet, so that deleting any single file triggers a refetch
  of only that file rather than the whole group.

Three independent dry-runs in scratch confirmed:

1. Cold run: all eight small files (3 CC + 4 DR16 + BOSS voids) fetched
   with positive byte counts, `safe_fetch` returned success on each.
2. Warm run: all eight files skipped as "already present".
3. Mid-set deletion of `HzTable_MM_M11.dat`: only that file refetched; the
   other two CC files remained marked "already present".

Seven of the eight files produced byte-identical SHA-256 on the second fetch.
The eighth (`boss_voids/table1.dat`) differs only in the FITS `DATE` header
timestamp written by CDS at serve time (see the note in the BOSS DR12 section
above); its scientific content is byte-stable.

## Post-hardening targeted validator pass

With the freshly-downloaded small external data in place (CC H(z) + DR16
BAOplus; Pantheon+, DESI BAO and CMB distance prior not fetched), a second
`run_validate.py` run in the Phase C scratch venv reports:

- Cosmic Chronometers H(z): n = 15, χ² = 7.0, DOF = 15, χ²/ν = 0.47
- DR16 fσ₈ Growth: n = 4, χ² = 2.0, DOF = 3, χ²/ν = 0.67
- Strict combined χ² = 10.66, DOF = 33, χ²/ν = 0.32
- Proof: 17/17 = 100 % at 1 σ (15 scalar pillars + 2 vector pillars with data)

The earlier "DR16 fσ₈ n = 0" condition from the workbook-only Phase C run was
therefore a missing-data artefact, not a code issue; it is resolved once the
external tables are present. Pantheon+, DESI BAO and CMB distance prior still
report missing-data errors in this targeted pass because their fetchers were
intentionally out of scope for this small-data-only verification.

## Policy for cached-fallback data

If an upstream source later disappears between this audit and release, a cached
copy may be placed under `anc/data_cache/<dataset>/` **only** for small,
licence-compatible files. Every cached file must carry a `PROVENANCE.md`
sibling that records:

- original source URL
- access date
- upstream citation
- upstream licence or redistribution note
- SHA-256
- reason for caching

The 19 GB bulk data (KiDS, Planck, DESI) must not be redistributed through the
repository or the arXiv ancillary tarball. Only small tables essential to
scalar-pillar reproducibility are candidates for caching.

## Verification command

From the repository root:

```bash
bash scripts/download_data.sh
sha256sum validation/data/External/hz_cc/*.dat \
          validation/data/External/growth_fsig8/sdss_DR16_BAOplus_* \
          validation/data/External/boss_voids/table1.dat
```

The hashes above were computed in the dry-run scratch tree and are expected to
be stable for these authoritative public sources.
