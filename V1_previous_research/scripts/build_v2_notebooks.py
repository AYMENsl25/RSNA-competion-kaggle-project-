"""Build standalone Kaggle notebooks deterministically from reviewed source modules.

Run from the repository: python scripts/build_v2_notebooks.py
Only generated notebooks/v2/*.ipynb are replaced. Original notebooks remain intact.
"""
from pathlib import Path
import hashlib
import json
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks' / 'v2'
MODULES = ['__init__', 'common', 'preprocessing', 'training', 'evaluation', 'submission']


def code(s):
    s = textwrap.dedent(s).strip()
    s = s.replace("['/kaggle/input']", "['/kaggle/input/datasets/aymenslimani', '/kaggle/input']")
    return nbf.v4.new_code_cell(s + '\n')
def md(s): return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())


def bootstrap():
    sources = {name+'.py': (ROOT/'src'/'rsna_v2'/(name+'.py')).read_text(encoding='utf-8') for name in MODULES}
    digest = hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:16]
    # Embedded source files permit normal Python imports; no repository attachment or internet required.
    return code('''
        from pathlib import Path
        import sys, os, json, importlib
        RUNTIME_ROOT = Path(os.environ.get('RSNA_V2_RUNTIME_ROOT', '/kaggle/working')) / 'rsna_v2_runtime_DIGEST'
        PACKAGE_DIR = RUNTIME_ROOT / 'rsna_v2'
        PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
        SOURCES = SOURCES_LITERAL
        for filename, source in SOURCES.items():
            (PACKAGE_DIR / filename).write_text(source, encoding='utf-8')
        for name in list(sys.modules):
            if name == 'rsna_v2' or name.startswith('rsna_v2.'):
                del sys.modules[name]
        sys.path.insert(0, str(RUNTIME_ROOT))
        importlib.invalidate_caches()
        from rsna_v2.common import *
        print('Embedded source version: DIGEST')
    '''.replace('SOURCES_LITERAL',repr(sources)).replace('DIGEST',digest))


def deps(training=False):
    packages=['numpy','pandas','pydicom','cv2','sklearn','torch','torchvision','timm','tqdm'] if training else ['numpy','pandas','pydicom','cv2','sklearn']
    return code(f'''
        import importlib.util, subprocess, sys
        # Turn on only in an Internet-enabled setup session. Offline training/submission uses attached wheels.
        INSTALL_MISSING = False
        WHEEL_DIR = None  # optional /kaggle/input/<your-wheel-dataset>; never a guessed attachment slug
        REQUIRED = {packages!r}
        PIP_NAMES = {{'cv2':'opencv-python-headless', 'sklearn':'scikit-learn'}}
        missing = [p for p in REQUIRED if importlib.util.find_spec(p) is None]
        if missing and INSTALL_MISSING:
            command = [sys.executable, '-m', 'pip', 'install']
            if WHEEL_DIR:
                command += ['--no-index', '--find-links', WHEEL_DIR]
            subprocess.check_call(command + [PIP_NAMES.get(p,p) for p in missing])
        elif missing:
            raise RuntimeError(f'Missing dependencies: {{missing}}. Install in setup or attach wheels; see README.')
        print('Required imports available. Compressed DICOM decoder availability is checked by the pilot.')
    ''')


def notebook(name,title,description,cells,training=False):
    nb=nbf.v4.new_notebook(cells=[md(f'# {title}\n\n{description}'), deps(training), bootstrap()]+cells)
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                 'language_info':{'name':'python','version':'3.11'},
                 'rsna_v2':{'standalone':True,'generated_by':'scripts/build_v2_notebooks.py'}}
    for i,cell in enumerate(nb.cells): cell['id']=hashlib.sha256((name+str(i)).encode()).hexdigest()[:12]
    nbf.validate(nb)
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=='code': compile(cell.source,f'{name}:cell{i}','exec')
    OUT.mkdir(parents=True,exist_ok=True)
    nbf.write(nb,OUT/name)
    print(f'{name}: {len(nb.cells)} cells, valid JSON/schema/Python')


def main():
    notebook('00_Inputs_Splits_and_Weights.ipynb','00 — Frozen inputs, splits, and pretrained weights',
        'Run first on CPU. Attach competition data, Phase 4A-3 pseudo-label outputs, Phase 4B-2A slot plan, and the v1.2 fold CSV. Save Version → Save & Run All, then attach this output to later notebooks. Enable Internet only if downloading pretrained weights.',[
        md('The original competition mount and original output filenames are retained. Exact previous attachment paths are documented in README. Duplicate files must have identical hashes or be resolved explicitly. Gold labels are never exported into the training manifest.'),
        code('''
            COMP_DIR = resolve_comp_dir()
            DATASET_ROOT = Path('/kaggle/input/datasets/aymenslimani')
            INPUT_ROOTS = [str(DATASET_ROOT), '/kaggle/input']
            OUTPUT_ROOT = Path('/kaggle/working/rsna_v2_inputs')
            # Check every required attachment together, before writing any outputs.
            REQUIRED_INPUTS = {
                'slot_plan_path': 'rsna_phase4b2a_six_slot_plan_v1.csv',
                'pseudo_path': 'rsna_pseudo_labels_v1_1_wide.csv',
                'pseudo_meta_path': 'PSEUDO_LABELS_v1_1.json',
                'folds_path': 'rsna_phase4b1b_pseudo_fold_assignments_v1_2.csv',
            }
            INPUT_OVERRIDES = {key: None for key in REQUIRED_INPUTS}
            # If a filename differs, paste its copied Kaggle path into INPUT_OVERRIDES.
            resolved_inputs, input_errors = {}, []
            for key, filename in REQUIRED_INPUTS.items():
                try:
                    resolved_inputs[key] = str(resolve_unique(filename, INPUT_ROOTS, INPUT_OVERRIDES[key]))
                except (FileNotFoundError, ValueError) as exc:
                    input_errors.append(str(exc))
            if input_errors:
                available_csvs = sorted(str(p) for p in Path('/kaggle/input').rglob('*fold*.csv'))
                raise RuntimeError('Input check failed. Attach Phase 4A-3, PSEUDO_LABELS_v1_1 dataset, '
                                   'and result phase 4 b. Do not regenerate folds silently.\\n'
                                   + '\\n'.join(input_errors)
                                   + '\\nAvailable fold CSVs: ' + repr(available_csvs))
            print('Required inputs:')
            for key, path in resolved_inputs.items():
                print(key, '->', path)
            RAW_SLOT_PLAN = resolved_inputs['slot_plan_path']
            role_cols = ['StudyInstanceUID', 'training_role', 'cv_group_id']
            raw_slot = pd.read_csv(RAW_SLOT_PLAN, usecols=role_cols, dtype=str, low_memory=False)
            clean_slot = raw_slot[role_cols].dropna().drop_duplicates().copy()
            if clean_slot['StudyInstanceUID'].duplicated().any():
                bad = clean_slot.loc[clean_slot['StudyInstanceUID'].duplicated(keep=False), role_cols]
                raise ValueError(f'Inconsistent study role/group rows after cleaning: {bad.head(20).to_dict("records")}')
            CLEAN_SLOT_PLAN = OUTPUT_ROOT / 'clean_inputs' / 'rsna_phase4b2a_six_slot_plan_v1_clean.csv'
            CLEAN_SLOT_PLAN.parent.mkdir(parents=True, exist_ok=True)
            clean_slot.to_csv(CLEAN_SLOT_PLAN, index=False)
            print('Raw slot plan:', RAW_SLOT_PLAN, raw_slot.shape)
            print('Clean slot plan:', CLEAN_SLOT_PLAN, clean_slot.shape)
            print(clean_slot['training_role'].value_counts().to_string())
            INPUT_CONFIG = {
                'input_roots': INPUT_ROOTS,
                'out_dir': str(OUTPUT_ROOT / 'supervision'),
                **resolved_inputs,
                'slot_plan_path': str(CLEAN_SLOT_PLAN),
                'regenerate_folds': False,  # reuse the previous v1.2 fold CSV
                'n_folds': 5,
            }
            print('Competition:', COMP_DIR)
            SUPERVISION_DIR = build_supervision(INPUT_CONFIG)
        '''),
        md('Optional weight preparation: these are generic pretrained backbones, not the old NLI model or a public competition ensemble. Offline runs require attached weights. Enable PREPARE_WEIGHTS in this setup notebook with Internet enabled; this cell records the downloaded files and hashes.'),
        code('''
            PREPARE_WEIGHTS = False
            WEIGHT_FAMILIES = ['resnet18', 'dinov2']
            if PREPARE_WEIGHTS:
                import torch
                weights_dir = OUTPUT_ROOT / 'pretrained'
                weights_dir.mkdir(parents=True, exist_ok=True)
                records = {}
                for family in WEIGHT_FAMILIES:
                    if family == 'resnet18':
                        from torchvision.models import resnet18, ResNet18_Weights
                        model = resnet18(weights=ResNet18_Weights.DEFAULT)
                        filename = 'resnet18_imagenet.pth'
                    elif family == 'dinov2':
                        import timm
                        model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, img_size=224)
                        filename = 'dinov2_vits14_timm.pth'
                    else:
                        raise ValueError(family)
                    path = weights_dir / filename
                    torch.save(model.state_dict(), path)
                    records[family] = {'file':filename, 'sha256':hash_file(path)}
                    del model
                write_json(weights_dir / 'pretrained_manifest.json', {'weights':records,'environment':environment()})
                print(records)
            else:
                print('Weight downloads disabled. Set PREPARE_WEIGHTS=True here, or attach matching local weights to training.')
        '''),
        code('''
            write_json(OUTPUT_ROOT / 'environment.json', environment())
            print('SAVE THIS OUTPUT and add it to later notebooks:', OUTPUT_ROOT)
        ''')],training=True)

    for number,pilot in [('01',True),('02',False)]:
        description = ('Run this small preprocessing pilot first. Inspect montages, coverage, decode errors, and runtime before a full cache. Pilot caches cannot be used for production training.' if pilot else 'Run after approving the pilot. Choose ONE preprocessing recipe; save each recipe as a separate Kaggle output. This notebook builds the full cache, not model weights.')
        notebook(f'{number}_Preprocessing_{"Pilot" if pilot else "Full"}.ipynb',f'{number} — MRI preprocessing {"pilot" if pilot else "full cache"}', description,[
            md('Default control uses legacy six slots, a 130-mm crop, 224 pixels and a broad slice band. It is a new v2 control, not an exact replay of v1.2. Compare one parameter at a time. Full-size crop, wider/narrower physical crop, and normalization are selectable below. Geometry canonicalization requires pilot inspection.'),
            code(f'''
                PREPROCESS_RECIPE = 'p0_control'
                RECIPES = {{
                    'p0_control': {{}},
                    'p1_full_fov': {{'crop_mm': None}},
                    'p2_crop_150mm': {{'crop_mm': 150.0}},
                    'p3_clahe': {{'normalization':'clahe'}},
                    'p4_zscore': {{'normalization':'zscore'}},
                    'p5_central_band': {{'slice_band':[0.12,0.88]}},
                }}
                CONFIG = {{
                    'comp_dir':str(resolve_comp_dir()),
                    'input_roots':['/kaggle/input/datasets/aymenslimani', '/kaggle/input'], 'slot_plan_path':None,
                    'out_dir':f'/kaggle/working/rsna_v2_{{PREPROCESS_RECIPE}}_{"pilot" if pilot else "cache"}',
                    'n_slices':24, 'image_size':224, 'crop_mm':130.0,
                    'normalization':'percentile', 'slice_band':[0.05,0.95],
                    'laterality':'off', 'smoke_studies':{8 if pilot else 0},
                }}
                CONFIG.update(RECIPES[PREPROCESS_RECIPE])
                from rsna_v2.preprocessing import run_preprocessing
                CACHE_DIR = run_preprocessing(CONFIG)
            '''),
            code('''
                import numpy as np, pandas as pd, matplotlib.pyplot as plt
                index = pd.read_csv(CACHE_DIR / 'cache_index.csv', dtype={'StudyInstanceUID':str})
                display(index.groupby('slot').agg(n_series=('SeriesInstanceUID','count'), mean_valid=('n_valid','mean')))
                # No report text or gold label selection: show the first deterministic studies.
                rows = index.sort_values(['StudyInstanceUID','slot']).head(6)
                fig,axes=plt.subplots(max(len(rows),1),4,figsize=(10,2.5*max(len(rows),1)),squeeze=False)
                for r,(_,row) in enumerate(rows.iterrows()):
                    with np.load(CACHE_DIR / row['path'],allow_pickle=False) as data:
                        good=np.flatnonzero(data['valid'])
                        if len(good)==0:
                            for ax in axes[r]: ax.axis('off')
                            continue
                        for c,k in enumerate(np.linspace(0,len(good)-1,4).round().astype(int)):
                            axes[r,c].imshow(data['images'][good[k]],cmap='gray',vmin=0,vmax=255)
                            axes[r,c].set_title(f"{row['slot']} slice {good[k]}")
                            axes[r,c].axis('off')
                plt.tight_layout(); plt.savefig(CACHE_DIR/'pilot_montage.png',dpi=120); plt.show()
                print('Save Version output:',CACHE_DIR)
            ''')])

    for number,family in [('03','resnet18'),('04','dinov2')]:
        notebook(f'{number}_Train_{"ResNet18" if family=="resnet18" else "Frozen_DINOv2"}.ipynb',f'{number} — {family} grouped selection training',
            'GPU notebook. Attach the full preprocessing output and notebook 00 output (supervision and weights). Run one experiment and one fold at a time. Checkpoints and validation predictions are saved for comparison; no gold metrics are computed.',[
            code(f'''
                # If several different caches are attached, set CACHE_META_PATH explicitly.
                CACHE_META_PATH = None
                SUPERVISION_META_PATH = None
                PRETRAINED_PATH = None
                CACHE_DIR = resolve_unique('cache_meta.json',['/kaggle/input'],CACHE_META_PATH).parent
                SUPERVISION_DIR = resolve_unique('supervision_meta.json',['/kaggle/input'],SUPERVISION_META_PATH).parent
                PRETRAINED = resolve_unique('{"resnet18_imagenet.pth" if family=="resnet18" else "dinov2_vits14_timm.pth"}',
                                            ['/kaggle/input'],PRETRAINED_PATH)
                EXPERIMENT = '{"r0_resnet_meanmax" if family=="resnet18" else "d0_dino_frozen_meanmax"}'
                FOLD = 0
                CONFIG = {{
                    'cache_dir':str(CACHE_DIR), 'supervision_dir':str(SUPERVISION_DIR),
                    'out_dir':f'/kaggle/working/rsna_v2_{{EXPERIMENT}}_fold{{FOLD}}',
                    'model_family':'{family}', 'pooling':'meanmax',
                    'input_mode':'repeat', 'train_anchors':8, 'eval_anchors':16,
                    'fold':FOLD, 'n_folds':5, 'mode':'selection',
                    'epochs':12, 'patience':3, 'batch_size':2, 'num_workers':2,
                    'learning_rate':0.0001, 'backbone_lr':0.00001,
                    'frozen_encoder':{family=='dinov2'}, 'seed':2026,
                    'pretrained_path':str(PRETRAINED), 'allow_download':False,
                    'require_cuda':True, 'resume_path':None,
                }}
                # Ablations: input_mode='triplet'; train_anchors=16; pooling='attention'.
                # Change only one factor, and assign a new EXPERIMENT/output directory.
                from rsna_v2.training import run_training
                RUN_DIR = run_training(CONFIG)
            '''),
            code('''
                import pandas as pd, matplotlib.pyplot as plt
                history = pd.read_csv(RUN_DIR/'history.csv')
                display(history)
                numeric = history.select_dtypes('number')
                if 'epoch' in numeric:
                    numeric.set_index('epoch').plot(subplots=True,figsize=(9,8),title='Selection diagnostics')
                    plt.tight_layout(); plt.show()
                print('Save Version and attach this run to comparison:',RUN_DIR)
            '''),
            md('A best epoch at the last allowed epoch is right-censored: report it and extend the prespecified budget or confirm convergence before refit. Compare folds with the same cohorts. A lower pseudo loss measures teacher agreement, not proven gold-AUC improvement.')],training=True)

    notebook('05_Compare_Experiments.ipynb','05 — Compare matching validation experiments',
        'CPU notebook. Attach completed training runs and notebook 00 output. Set explicit RUN_DIRS. Comparisons require matching held-out studies and unchanged supervision.',[
        code('''
            RUN_DIRS = []  # e.g. paths to two attached rsna_v2_*_fold0 directories
            SUPERVISION_META_PATH = None
            SUPERVISION_DIR = resolve_unique('supervision_meta.json',['/kaggle/input'],SUPERVISION_META_PATH).parent
            from rsna_v2.evaluation import compare_runs
            if not RUN_DIRS:
                print('Set RUN_DIRS to the explicit directories of completed matching-fold runs.')
            else:
                comparison=compare_runs(RUN_DIRS,SUPERVISION_DIR,'/kaggle/working/rsna_v2_comparison')
                display(comparison)
        '''),
        md('Use the same validation labels and weights for all supervision variants. Confirm promising changes with a second seed and the remaining folds before spending on a diverse ensemble. Full OOF rank-stacking and uncertainty-based relabeling are subsequent experiments; this notebook provides matching-fold comparison and an equal probability-mean diagnostic.')])

    notebook('06_Refit_Frozen_Recipe.ipynb','06 — Refit the selected recipe',
        'GPU notebook. Run only after choosing the recipe and epoch budget from pseudo-validation. Attach the selected training checkpoint, full cache, and supervision. This refit uses all pseudo-training studies and excludes gold groups.',[
        code('''
            RUN_REFIT = False
            SELECTION_CHECKPOINT = None  # explicit best.pt from the selected experiment
            CACHE_META_PATH = None
            SUPERVISION_META_PATH = None
            PRETRAINED_PATH = None  # original generic backbone weights, not selection model weights
            REFIT_EPOCHS = None  # prespecified after reviewing selection histories
            if RUN_REFIT:
                if not SELECTION_CHECKPOINT or not PRETRAINED_PATH or not REFIT_EPOCHS:
                    raise ValueError('Set selected checkpoint, original pretrained weights, and refit epoch budget.')
                from rsna_v2.training import load_trained_model, run_training
                model, selected_config = load_trained_model(SELECTION_CHECKPOINT,'cpu')
                del model
                CONFIG = dict(selected_config)
                CONFIG.update({
                    'cache_dir':str(resolve_unique('cache_meta.json',['/kaggle/input'],CACHE_META_PATH).parent),
                    'supervision_dir':str(resolve_unique('supervision_meta.json',['/kaggle/input'],SUPERVISION_META_PATH).parent),
                    'mode':'refit','epochs':int(REFIT_EPOCHS),'pretrained_path':PRETRAINED_PATH,
                    'out_dir':'/kaggle/working/rsna_v2_final_refit','resume_path':None,
                    'allow_download':False,'require_cuda':True,
                })
                RUN_DIR=run_training(CONFIG)
                print(RUN_DIR)
            else:
                print('Refit disabled until recipe and epochs are frozen.')
        ''')],training=True)

    notebook('07_Historical_Gold_Audit.ipynb','07 — Historical gold audit after recipe freeze',
        'The 58 gold cases have been repeatedly examined in prior work. This is a descriptive historical audit, not a fresh independent test. Default disabled. Never use this notebook for an architecture sweep.',[
        code('''
            RUN_GOLD_AUDIT = False
            RECIPE = {
                'frozen':False,
                'rationale':'',  # selected using grouped pseudo-validation, before this audit
                'checkpoints':[],  # [{'path':'/kaggle/input/.../best.pt','sha256':'...'}]
            }
            CACHE_META_PATH = None
            SUPERVISION_META_PATH = None
            if RUN_GOLD_AUDIT:
                recipe_path=Path('/kaggle/working/frozen_recipe.json')
                write_json(recipe_path,RECIPE)
                from rsna_v2.evaluation import run_gold_audit
                metrics,summary=run_gold_audit({
                    'run_gold_audit':True,'recipe_path':str(recipe_path),
                    'cache_dir':str(resolve_unique('cache_meta.json',['/kaggle/input'],CACHE_META_PATH).parent),
                    'supervision_dir':str(resolve_unique('supervision_meta.json',['/kaggle/input'],SUPERVISION_META_PATH).parent),
                    'out_dir':'/kaggle/working/rsna_v2_gold_audit','bootstrap':1000,
                })
                display(metrics); print(summary)
            else:
                print('Historical gold audit disabled.')
        ''')],training=True)

    notebook('08_Submission_Inference.ipynb','08 — Offline submission inference',
        'Attach final v2 checkpoints and the exact cache_meta.json used for preprocessing. Enable a GPU; Internet off. The notebook uses the same preprocessing and model code as training and processes test studies with bounded temporary storage.',[
        code('''
            RUN_SUBMISSION = False
            CONFIG = {
                'run_submission':RUN_SUBMISSION,
                'comp_dir':'/kaggle/input/competitions/rsna-knee-abnormality-detection',
                'out_dir':'/kaggle/working',
                'checkpoints':[],  # [{'path':'/kaggle/input/.../best.pt','sha256':'...'}]
                'cache_meta_path':None,
                'cache_meta_sha256':None,
                'device':'cuda',
            }
            if RUN_SUBMISSION:
                from rsna_v2.submission import run_submission
                result=run_submission(CONFIG)
                print(result)
            else:
                print('Set exact checkpoint paths/hashes and preprocessing metadata, then enable RUN_SUBMISSION.')
        ''')],training=True)


if __name__=='__main__': main()
