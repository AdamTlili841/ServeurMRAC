# ServeurMRAC (archive GitHub)

L’API est maintenant déployée sur **Hugging Face Spaces** :

**https://huggingface.co/spaces/AdamTLILI/MRAC**

Fichiers prêts à pousser : dossier `d:\site web\MRAC` (voir `DEPLOY.md`).

L’extension Chrome utilise `https://adamtlili-mrac.hf.space`.

## Contenu (référence)

| Fichier | Rôle |
|---------|------|
| `main.py` | API FastAPI |
| `multimodal_model.py` | ViT + BERT + co-attention |
| `best_model.pt` | Checkpoint (Git LFS) |
| `Dockerfile` | Image Docker (Render obsolète — utiliser le dossier `MRAC` pour HF) |

## Local

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
