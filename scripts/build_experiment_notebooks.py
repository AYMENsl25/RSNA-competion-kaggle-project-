"""Generate portable research notebooks; previous notebooks are preserved."""
from pathlib import Path
import hashlib
import json
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'notebooks'/'experiments'
MODULES = ['__init__', 'common', 'preparation', 'preprocessing', 'registry', 'models', 'losses', 'training', 'comparison']


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip()+'\n')


def bootstrap():
    sources = {name+'.py': (ROOT/'src'/'rsna_experiments'/(name+'.py')).read_text(encoding='utf-8') for name in MODULES}
    version = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]
    return code('''
        from pathlib import Path
        import os, sys, json, importlib, logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
        RUNTIME = Path(os.environ.get('RSNA_EXPERIMENT_RUNTIME', '/kaggle/working'))/'rsna_experiments___RUNTIME_DIGEST__'
        PACKAGE = RUNTIME/'rsna_experiments'
        PACKAGE.mkdir(parents=True, exist_ok=True)
        SOURCES = SOURCE_LITERAL
        for name, source in SOURCES.items(): (PACKAGE/name).write_text(source, encoding='utf-8')
        for name in list(sys.modules):
            if name == 'rsna_experiments' or name.startswith('rsna_experiments.'): del sys.modules[name]
        sys.path.insert(0, str(RUNTIME)); importlib.invalidate_caches()
        from rsna_experiments.common import *
        import pandas as pd, numpy as np
        print('Experiment runtime __RUNTIME_DIGEST__')
    '''.replace('SOURCE_LITERAL', repr(sources)).replace('__RUNTIME_DIGEST__', version))


def dependencies(training=False):
    names = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn', 'pydicom': 'pydicom', 'PIL': 'pillow', 'cv2': 'opencv-python-headless'}
    if training:
        names.update({'torch': 'torch', 'torchvision': 'torchvision', 'monai': 'monai==1.5.1', 'timm': 'timm'})
    return code(f'''
        import importlib.util, importlib.metadata, subprocess, sys
        INSTALL_MISSING = True  # enable Kaggle Internet for setup, or attach wheels below
        WHEEL_DIR = None
        PACKAGES = {names!r}
        missing = [pip for module,pip in PACKAGES.items() if importlib.util.find_spec(module) is None]
        if missing:
            if not INSTALL_MISSING: raise RuntimeError(f'Missing dependencies: {{missing}}. Enable install or attach wheels.')
            cmd = [sys.executable,'-m','pip','install']
            if WHEEL_DIR: cmd += ['--no-index','--find-links',WHEEL_DIR]
            subprocess.check_call(cmd+missing)
        print('Dependencies available. Tested MONAI version: 1.5.1; actual:', importlib.metadata.version('monai') if 'monai' in PACKAGES else 'not needed')
    ''')


def notebook(name, title, description, cells, training=False):
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# '+title+'\n\n'+description), dependencies(training), bootstrap()]+cells)
    nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                   'language_info': {'name': 'python', 'version': '3.11'},
                   'rsna_experiments': {'standalone': True, 'generator': 'scripts/build_experiment_notebooks.py'}}
    for i, cell in enumerate(nb.cells):
        cell['id'] = hashlib.sha256((name+str(i)).encode()).hexdigest()[:12]
        if cell.cell_type == 'code': compile(cell.source, f'{name}:{i}', 'exec')
    nbf.validate(nb); OUT.mkdir(parents=True, exist_ok=True); nbf.write(nb, OUT/name)
    print(name, 'PASS schema/Python')


def main():
    notebook('01_Experiment_Readiness.ipynb', '01 — Fixed cohort and label readiness',
        'CPU. Attach the saved output from fresh/01_Preparation_Labels_and_Audit (contains fresh_manifest.csv). '
        'No old v1 pseudo labels or slot plan needed. This notebook preserves your fresh labels and folds, audits coverage, '
        'and prepares a bounded 512-training/240-validation cohort. English regex supervision may miss multilingual reports. '
        'Review the audit sample before larger runs. Save this output for notebook 02.', [
        code('''
            MANIFEST_PATH = None  # exact copied file path if there are multiple attached versions
            CORRECTIONS_PATH = None  # optional reviewed training-only correction CSV; see README
            CONFIG = {'manifest_path': str(locate('fresh_manifest.csv', override=MANIFEST_PATH)),
                      'out_dir': '/kaggle/working/rsna_experiment_preparation',
                      'fold': 0, 'max_train': 512, 'max_val': 240, 'seed': 42,
                      'corrections_path': CORRECTIONS_PATH}
            from rsna_experiments.preparation import prepare
            PREP = prepare(CONFIG)
            display(pd.read_csv(PREP/'experiment_manifest.csv').groupby('fold').size())
            readiness = json.loads((PREP/'preparation_meta.json').read_text())
            display(pd.DataFrame(readiness['target_fold_counts']))
            print('Warnings:', readiness['warnings'])
            print('Cohort:', readiness['cohort_id'], 'Supervision:', readiness['supervision_id'])
            print('Saved audit files:', [p.name for p in PREP.iterdir()])
        '''),
        nbf.v4.new_markdown_cell('Keep this cohort and its validation labels unchanged throughout one comparison round. '
            'To expand, use max_train=0 and max_val=0 for all eligible studies and a NEW output directory; rerun the control. '
            'The grouping fallback only detects normalized identical reports, not all related patients. '
            'Undefined AUCs remain undefined. Do not interpret this teacher-agreement cohort as an independent clinical or leaderboard test.')])

    notebook('02_Preprocessing_Experiments.ipynb', '02 — Controlled MRI preprocessing',
        'CPU. Attach the competition and saved experiment notebook 01 output. Select ONE recipe and save its output separately. '
        'Start with a timing pilot, inspect the montage and error CSVs, then build the full fixed cohort. '
        'This cache is multi-plane and is different from the earlier single-series 240-study cache.', [
        code('''
            MANIFEST_PATH = None
            RECIPE = 'p0'  # p0 full FOV; p1 crop130; p2 crop150; p3 zscore; p4 CLAHE; p5 central band
            PILOT = True  # 12-study decoding/timing gate; change False for a trainable full cohort cache
            CONFIG = {'comp_dir': str(competition_path()),
                      'manifest_path': str(locate('experiment_manifest.csv', override=MANIFEST_PATH)),
                      'out_dir': f'/kaggle/working/rsna_experiment_{RECIPE}_' + ('pilot' if PILOT else 'cache'),
                      'recipe': RECIPE, 'slices': 16, 'size': 192, 'max_studies': 12 if PILOT else 0}
            from rsna_experiments.preprocessing import build_cache
            CACHE = build_cache(CONFIG)
            receipt = json.loads((CACHE/'cache_meta.json').read_text())
            print('Counts:', receipt['counts'], 'seconds:', receipt['seconds'], 'trainable:', receipt['complete'] and not PILOT)
            print('Cache ID:', receipt['cache_id'])
        '''),
        code('''
            import matplotlib.pyplot as plt
            index = pd.read_csv(CACHE/'cache_index.csv', dtype={UID:str})
            index = index[index['file'].notna() & index['file'].ne('')]
            if not index.empty:
                with np.load(CACHE/index.iloc[0]['file']) as data:
                    images, mask = data['images'], data['mask']
                fig, axes = plt.subplots(3,4,figsize=(12,9))
                for plane in range(3):
                    positions = np.flatnonzero(mask[plane])
                    for k, ax in enumerate(axes[plane]):
                        ax.axis('off')
                        if len(positions):
                            pos = positions[min(len(positions)-1, round(k*(len(positions)-1)/3))]
                            ax.imshow(images[plane,pos],cmap='gray'); ax.set_title(['SAG','COR','AX'][plane])
                plt.tight_layout(); plt.savefig(CACHE/'montage.png'); plt.show()
            print('Inspect montage and errors. For training attach the full cache, not the pilot.')
        ''')])

    notebook('03_Training_Experiments.ipynb', '03 — Models, losses and training strategies',
        'GPU. Attach ONE full experiment notebook 02 cache. Enable Internet for package setup and generic ResNet18 weights, '
        'or supply offline wheels/weights. Select ONE experiment; each has a separate output. '
        'Comparisons use unchanged validation BCE, per-target AUC/AP and a training-prior baseline. '
        'MONAI supplies focal loss, intensity augmentation and the small 3D DenseNet comparator.', [
        code('''
            from rsna_experiments.registry import EXPERIMENTS, experiment
            display(pd.DataFrame([{'experiment':name, 'changes':str(values)} for name,values in EXPERIMENTS.items()]))
            RUN_TRAINING = True  # False for comparison-only session; set RUN_DIRS in the final cell
            EXPERIMENT = 'r00_resnet_control'
            CACHE_META_PATH = None
            WEIGHTS_PATH = None  # offline: copied resnet18_imagenet.pth or dinov2_vits14_timm.pth path
            ALLOW_WEIGHT_DOWNLOADS = True  # ResNet18 only; generic ImageNet weights, no competition model
            RESUME_PATH = None  # your matching last.pt; copies the best model and restores optimizer/RNG
            CONFIG = experiment(EXPERIMENT,
                cache_dir=str(locate('cache_meta.json', override=CACHE_META_PATH).parent) if RUN_TRAINING else '',
                out_dir=f'/kaggle/working/rsna_{EXPERIMENT}', weights_path=WEIGHTS_PATH,
                allow_downloads=ALLOW_WEIGHT_DOWNLOADS, resume_path=RESUME_PATH)
            print(json.dumps(CONFIG,indent=2))
        '''),
        nbf.v4.new_markdown_cell('Optional DINO weight setup: only needed for m01_dinov2_frozen. '
            'This is a generic pretrained timm encoder, not the previous competition checkpoint. '
            'Enable the cell in an Internet session, save its output and attach it for offline runs.'),
        code('''
            PREPARE_DINO_WEIGHTS = False
            if PREPARE_DINO_WEIGHTS:
                import timm, torch
                weights_dir = Path('/kaggle/working/generic_weights'); weights_dir.mkdir(exist_ok=True)
                encoder = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, img_size=224)
                weight_file = weights_dir/'dinov2_vits14_timm.pth'
                torch.save(encoder.state_dict(), weight_file); del encoder
                write_json(weights_dir/'weights_manifest.json', {'file':weight_file.name,'sha256':sha256(weight_file),'environment':environment()})
                if CONFIG['model'] == 'dinov2': CONFIG['weights_path'] = str(weight_file)
        '''),
        code('''
            from rsna_experiments.training import run_training
            if RUN_TRAINING:
                RUN = run_training(CONFIG)
                display(pd.read_csv(RUN/'history.csv'))
                display(pd.read_csv(RUN/'target_metrics.csv'))
                print('Save this output:', RUN)
        '''),
        nbf.v4.new_markdown_cell('Compare runs after attaching their saved outputs. The comparison checks study IDs, groups and labels; '
            'different validation cohorts cannot be ranked together. Paired bootstrap intervals describe this exploratory validation set. '
            'The equal-weight ensemble is exploratory and must be confirmed on another fold.'),
        code('''
            RUN_DIRS = []  # exact folders containing run_summary.json, control first; include str(RUN) if desired
            if len(RUN_DIRS) >= 2:
                from rsna_experiments.comparison import compare_runs
                COMPARISON = compare_runs(RUN_DIRS, '/kaggle/working/rsna_experiment_comparison', bootstrap=200)
                print('Saved comparison:', COMPARISON)
            else:
                print('Comparison skipped: set RUN_DIRS to at least two matched experiment output folders.')
        ''')], training=True)


if __name__ == '__main__':
    main()
