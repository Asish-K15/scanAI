# Dataset Manager — VetVision / PetVision AI

Pipeline: **Download → Extract → Taxonomy Map → Validate → Dedup → Stats → clean/ (training-ready)**

## Setup

```bash
cd dataset_manager
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Kaggle API:** get your token from kaggle.com → Account → Create New API Token, place the
downloaded `kaggle.json` at `~/.kaggle/kaggle.json` (Linux/Mac) or
`C:\Users\<you>\.kaggle\kaggle.json` (Windows).

**Roboflow API:** get your API key from your Roboflow workspace settings, then:
```bash
export ROBOFLOW_API_KEY=your_key_here      # Mac/Linux
set ROBOFLOW_API_KEY=your_key_here         # Windows
```

## Folder structure

```
dataset_manager/
├── downloads/       # raw downloads (zips, roboflow exports) — stage 1 output
├── raw/              # extracted, source-tagged, but still using each dataset's own labels — stage 2 output
├── clean/            # final training-ready dataset, unified taxonomy — stage 3+ output
│   ├── skin/ringworm/
│   ├── skin/mange/
│   ├── parasites/tick_infestation/
│   ├── eye/pink_eye/
│   ├── ...
│   └── _UNMAPPED/    # anything that didn't match taxonomy.json — review these
├── metadata/          # reports: duplicates.csv, image_quality_report.csv, dataset_inventory.csv
├── taxonomy.json      # the single source of truth for class names — EDIT THIS, not folder names
└── licenses_template.csv  # fill in as you add datasets, don't skip this
```

## How to run

1. **Edit `download.py`** — add your vetted dataset list (Kaggle slugs, Roboflow workspace/project/version) to the `KAGGLE_DATASETS` and `ROBOFLOW_DATASETS` lists at the top.
2. **Fill in `licenses_template.csv`** as you go — rename it to `licenses.csv` once you start, log every dataset's license before you use it.
3. Run:
   ```bash
   python download.py
   python main.py
   ```
4. Check the console output and `metadata/dataset_inventory.csv` for classes below the 200-image minimum-viable threshold — that's your self-collection priority list.
5. Check `clean/_UNMAPPED/` — anything here means `taxonomy.json` needs a new alias added. Edit the JSON, then re-run:
   ```bash
   python taxonomy_mapper.py
   ```
6. Review `metadata/duplicates.csv` before actually deleting anything:
   ```bash
   python dedup.py --remove
   ```

## Adding a new dataset later

You don't need to re-run everything from scratch. Add the new source to
`download.py`, run `download.py` again (only pulls the new one), then run
`main.py` again — it re-processes everything, which is intentional since a
new dataset can introduce new duplicates against ones you already have.

## What NOT to put in clean/

`taxonomy.json` has an `excluded_not_photo_diagnosable` list (distemper,
parvovirus, feline leukemia, otitis media, mastitis, FMD, dental disease).
The taxonomy mapper automatically skips these — don't manually add them
back in, even if a source dataset includes them. These require blood
tests, internal imaging, or lab confirmation that a phone photo can't
provide, and including them undermines the project's credibility framing.

## Next stage after this

Once `clean/` looks solid (check `dataset_inventory.csv`), move to the
Colab baseline training notebook — don't wait for every class to hit
"excellent" tier first. Train early on what you have, then come back and
top up the weakest classes based on where the trained model actually
struggles.
