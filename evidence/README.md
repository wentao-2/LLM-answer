# Saved-output evidence

These are byte-identical copies of the three evidence archives published on
2026-10-12. Their sizes and SHA256 identities remain unchanged. They are normal
repository files in this directory. The links below open the included ZIPs.

| Archive | Bytes | SHA256 |
| --- | ---: | --- |
| [supplementary-extension-20261010.zip](supplementary-extension-20261010.zip) | 61,100,150 | `d57e3687fa45bf079af2f12d928a75ca64d85d81649d2b691a30a4118f981244` |
| [supplementary-review-supplement-20261011.zip](supplementary-review-supplement-20261011.zip) | 43,422,751 | `3a617bcd361ff131ec1a17b8e62894e99460d156adf4e2b222ea0d17f7be4ebd` |
| [supplementary-dependence-followup-20261012.zip](supplementary-dependence-followup-20261012.zip) | 43,542,064 | `7ebfbc89bb30ba8267997359276e88b0ae58f1e37a434a7a8848fc01efed510d` |

Run `python scripts/reproduce.py --suite all` from the repository root to use
these included files. The helper verifies complete byte counts and hashes
before extraction and offline reanalysis. No model inference is performed.
