#!/bin/bash

echo "🔧 Création de l'environnement virtuel..."
python3 -m venv venv

echo "⚡ Activation de l'environnement..."
source venv/bin/activate

echo "📦 Installation des dépendances depuis requirements.txt..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "⚠️  Fichier requirements.txt non trouvé. Aucune dépendance installée."
fi

echo "🧠 Installation du kernel Jupyter lié au venv..."
pip install ipykernel

python -m ipykernel install --user --name=store_sales_venv --display-name "Python (Store Sales)"

echo "✅ Environnement prêt et kernel 'Python (Store Sales)' enregistré dans Jupyter."
