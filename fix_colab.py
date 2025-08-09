import json, subprocess, re
from pathlib import Path

OWNER = "ElMartinez31"
REPO = "Data_Science"
BRANCH = "main"
BADGE_IMG = "https://colab.research.google.com/assets/colab-badge.svg"

# Récupère tous les notebooks suivis par git
nb_paths = subprocess.check_output(
    ["git", "ls-files", "*.ipynb"], text=True
).splitlines()

badge_md_pattern = re.compile(
    r'\[!\[.*?colab-badge\.svg.*?\)\]\((.*?)\)', re.IGNORECASE
)

changed_files = []

for nb_path in nb_paths:
    path_obj = Path(nb_path)
    rel_path = path_obj.as_posix()

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"⚠️ Impossible de lire {nb_path} : {e}")
        continue

    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = "".join
