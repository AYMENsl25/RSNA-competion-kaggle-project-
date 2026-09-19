"""Validate generated experiment notebooks and their embedded runtime imports."""
from pathlib import Path
import json
import tempfile
import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    '01_Experiment_Readiness.ipynb',
    '02_Preprocessing_Experiments.ipynb',
    '03_Training_Experiments.ipynb',
    '04_Full_Cohort_Readiness.ipynb',
    '05_Full_Cohort_Preprocessing.ipynb',
    '06_Long_Attention_Training_and_Evaluation.ipynb',
    '07_Matched_Long_Run_Comparison.ipynb',
    '08_Semantic_Slot_and_Laterality_Audit.ipynb',
    '09_Preprocessing_Ablation.ipynb',
    '10_Matched_Ablation_Training.ipynb',
]


def main():
    for name in EXPECTED:
        path = ROOT/'notebooks'/'experiments'/name
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for number, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'{name}:cell{number}', 'exec')
        # Bootstrap is cell 2; redirect /kaggle/working to an isolated local directory.
        # Keep validation scratch inside the repository: some managed Windows
        # environments expose an unreadable system temporary directory.
        with tempfile.TemporaryDirectory(prefix='rsna_experiment_notebook_', dir=ROOT) as directory:
            namespace = {'__name__': '__validation__'}
            source = "import os\nos.environ['RSNA_EXPERIMENT_RUNTIME'] = " + repr(directory) + "\n" + notebook.cells[2].source
            exec(compile(source, f'{name}:bootstrap', 'exec'), namespace)
            package = Path(namespace['PACKAGE'])
            expected_modules = {'__init__.py', 'common.py', 'preparation.py', 'preprocessing.py', 'semantic_audit.py',
                                'registry.py', 'models.py', 'losses.py', 'training.py', 'comparison.py'}
            if {p.name for p in package.glob('*.py')} != expected_modules:
                raise ValueError(f'{name}: incomplete embedded package')
            exec('from rsna_experiments import preparation, preprocessing, semantic_audit, registry, models, losses, training, comparison', namespace)
        print(name, 'PASS syntax/schema/runtime imports')


if __name__ == '__main__':
    main()
