"""Download the IBM AMLSim example dataset from Kaggle into data/raw/amlsim/.

    pip install kagglehub
    python scripts/get_amlsim.py

Public dataset: no Kaggle login needed for kagglehub in most setups. If it asks, create a token at
kaggle.com/settings and put kaggle.json in ~/.kaggle/.
"""

import shutil
from pathlib import Path

import kagglehub

src = Path(kagglehub.dataset_download("anshankul/ibm-amlsim-example-dataset"))
dst = Path(__file__).resolve().parents[1] / "data" / "raw" / "amlsim"
dst.mkdir(parents=True, exist_ok=True)
for name in ("transactions.csv", "alerts.csv", "accounts.csv"):
    shutil.copy(src / name, dst / name)
    print(f"{name}: {(dst / name).stat().st_size / 1e6:.1f} MB")
print(f"ready: {dst}\nnext: make run-amlsim")
