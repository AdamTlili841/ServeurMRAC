# ServeurMRAC

API d’inférence **MRAC-FND** (texte + image) pour l’extension Chrome MRACT-FND.  
Hébergée par le propriétaire de l’extension ; les utilisateurs envoient leurs requêtes vers cette API en HTTPS.

## Contenu

| Fichier | Rôle |
|---------|------|
| `main.py` | API FastAPI (`/health`, `/predict`) |
| `multimodal_model.py` | Architecture ViT + BERT + co-attention |
| `best_model.pt` | Checkpoint entraîné (**à ajouter**) |
| `requirements.txt` | Dépendances Python |
| `Dockerfile` | Déploiement conteneur (Render, Railway, VPS) |

## Ajouter le modèle

1. Copiez votre fichier `best_model.pt` à la **racine** de ce dépôt.
2. Poussez sur GitHub (Git LFS recommandé si le fichier > 100 Mo) :

```bash
git lfs install
git add best_model.pt
git add .
git commit -m "Ajout du checkpoint best_model.pt"
git push
```

## Lancer en local

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Test : `GET http://localhost:8000/health`

## Déployer (ex. Render)

1. Connectez le dépôt [ServeurMRAC](https://github.com/AdamTlili841/ServeurMRAC) sur [Render](https://render.com).
2. Choisissez **Docker** (utilise `Dockerfile` + `render.yaml`).
3. Une fois déployé, copiez l’URL HTTPS (ex. `https://serveur-mrac.onrender.com`).
4. Dans l’extension Chrome, éditez `config.js` :

```js
export const API_BASE_URL = "https://VOTRE-APP.onrender.com";
```

## API

### `GET /health`

```json
{ "status": "ok", "checkpoint": "best_model.pt" }
```

### `POST /predict`

`multipart/form-data` :

- `text` (obligatoire)
- `image` (optionnel)

Réponse :

```json
{
  "verdict": "fake",
  "confidence_fake": 0.87,
  "probs": [0.13, 0.87],
  "demo": false,
  "explanation": "Prédiction du modèle MRAC-FND (serveur central)."
}
```
