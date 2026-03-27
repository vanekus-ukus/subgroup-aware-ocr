# Data Access

## Public Data Included

Included:
- `data/toy_research/`: a small synthetic-toy benchmark safe for public release.

Not included:
- private benchmark images;
- imported source crops;
- synthetic square pool from the closed study;
- raw production-derived manifests.

## Expected Data Layout

Use the following layout for your own data:

```text
data/local/my_benchmark/
  real/
    000001_AB12CD.png
    000002_QX456B.png
  synth_square/
    000001_QX456B.png
    000002_ZA321D.png
  manifests/
    shape_manifest.csv
    style_manifest.csv
    split_manifest.csv
    source_class_manifest.csv
```

Template CSV files are provided in `data/manifests/`.

## Required Naming Convention

The OCR label is parsed from the filename stem. Supported examples:
- `000001_AB12CD.png`
- `000123_QX456B_QX456B.png`

The parser normalizes punctuation and duplicate tails.

## Privacy And Safety Rules

Before using your own data with this repo:
- remove absolute paths from manifests;
- avoid publishing raw images unless you have redistribution rights;
- do not commit local data directories;
- keep heavy checkpoints and raw outputs out of version control.
