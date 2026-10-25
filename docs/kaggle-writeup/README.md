# Illustrated Kaggle-style write-up

- [Read the write-up](WRITEUP.md)
- [Open/download the self-contained HTML preview](writeup.html)
- [PNG figures](assets/)
- [Copying into Kaggle](KAGGLE_PUBLISHING.md)

The write-up is a visual companion to the [technical report](../technical-report/REPORT.md).
It uses the same aggregate evidence, with a shorter competition-post structure.
It is not posted on Kaggle automatically.

## Regenerate

Install the technical report's Python requirements and provide a Poppler
`pdftoppm` executable on PATH, or pass its absolute path:

```bash
python build_writeup.py --pdftoppm /path/to/pdftoppm
python build_writeup.py --check
```

The builder reads only the write-up and the technical report's aggregate JSON.
It writes six PNG figures, a standalone HTML preview and an artifact manifest.
It performs no training, inference, network requests or patient-data access.
