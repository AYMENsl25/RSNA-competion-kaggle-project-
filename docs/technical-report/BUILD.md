# Build and verify the report

From this directory, in a Python 3.11+ environment with the listed packages:

```bash
python -m pip install -r requirements.txt
python build_report.py
python build_report.py --check
```

The builder uses ReportLab vector drawings and Platypus. It writes the two SVG
figures, the PDF and the SHA-256 artifact manifest. Invariant PDF metadata is
enabled; identical output additionally depends on the Python, ReportLab and
font/runtime versions. PDF byte identity across arbitrary environments is not
promised. The checked reference build uses ReportLab 4.4.9.

All inputs are the Markdown text and [aggregate results](data/aggregate_results.json).
No network, patient data, labels, model weights or GPU are used. The check mode
validates PDF text/page presence, local documentation links and interval ordering.
Visual review remains necessary to confirm layout.

The [evidence manifest](data/evidence_manifest.json) identifies existing research
records by relative path and SHA-256. Some source artifacts remain local; the
manifest makes no promise that these are all downloadable. It is provenance for
this report, not a complete training-reproduction package.

Review the PDF with a PDF reader. For an optional Poppler rendering check:

```bash
pdftoppm -r 90 -png rsna-knee-technical-report.pdf report-page
```

Generated page images are review intermediates and are not part of the report
publication. The package contains no new training results or new label analysis.
