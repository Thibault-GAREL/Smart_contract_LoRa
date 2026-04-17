# Guide : Entraînement LoRA sur RunPod & Sauvegarde (GitHub + Hugging Face)

Ce guide détaille les étapes validées pour cloner le projet depuis GitHub, configurer l'environnement GPU sur RunPod, lancer l'entraînement LoRA, et sauvegarder les résultats.

---

## Étape 1 : Créer un Pod sur RunPod

- Va sur [runpod.io](https://runpod.io) → **Deploy**
- Choisis un GPU : **RTX 3090 ou A40** (meilleur rapport prix/VRAM pour rester dans le budget de 3€)
- **Evite les GPU Blackwell** (RTX PRO 4500, RTX 5000 series) — incompatibles avec bitsandbytes
- Template : **RunPod PyTorch** (CUDA déjà configuré)
- Volume persistant : `/workspace` (5-10 Go suffisent)

---

## Étape 2 : Connexion et mise en place de l'espace de travail

Connecte-toi à l'instance via ton terminal local :

```bash
ssh root@<IP_DU_POD> -p <PORT>
```

Clone le projet dans le volume persistant :

```bash
cd /workspace
git clone https://github.com/Thibault-GAREL/Smart_contract_LoRa.git
cd Smart_contract_LoRa
```

---

## Étape 3 : Création de l'environnement Python

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Note : l'installation prend plusieurs minutes (téléchargement de torch + dépendances). Attends le message `Successfully installed ...` avant de continuer.

---

## Étape 4 : Authentification Hugging Face

```bash
python -c "from huggingface_hub import login; login(token='TON_TOKEN_WRITE')"
```

Génère un token Write dans Settings > Access Tokens sur huggingface.co.

---

## Étape 5 : Lancer l'entraînement

Lance les phases dans l'ordre — ne pas sauter directement à la Phase 3 sans valider les phases précédentes.

Note : `screen` et `tmux` ne sont pas disponibles sur RunPod. Utilise `nohup` pour que l'entraînement survive à une déconnexion.

### Phase 1 — Baseline zero-shot (test rapide, coût négligeable)

```bash
python evaluate.py --baseline --model_dir ./baseline_test --dataset "dataset/dataset_9l_w_v2 (1).csv" --max_samples 100 --no_quantize
```

### Phase 2 — Quick test (valide que le pipeline fonctionne)

```bash
nohup python train.py --max_samples 200 --epochs 1 --no_quantize > training_phase2.log 2>&1 &
tail -f training_phase2.log
```

### Phase 3 — Entraînement complet

```bash
nohup python train.py --epochs 3 --no_quantize > training.log 2>&1 &
tail -f training.log
```

- Détacher le suivi des logs sans arrêter l'entraînement : `Ctrl+C`
- Revenir voir les logs : `tail -f training.log`

---

## Étape 6 : Evaluer le modèle fine-tuné

```bash
python evaluate.py --model_dir models/tinyllama-lora_run-01_date-YYYY-MM-DD --dataset "dataset/dataset_9l_w_v2 (1).csv" --no_quantize
```

---

## Étape 7 : Sauvegarder les résultats

### Push le modèle sur Hugging Face

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id='Thibault-GAREL/smart-contract-lora-v2', repo_type='model', exist_ok=True)
api.upload_folder(
    folder_path='./models/tinyllama-lora_run-02_date-2026-03-29',
    repo_id='Thibault-GAREL/smart-contract-lora-v2',
    repo_type='model',
    commit_message='LoRA v2 - 5 epochs full training'
)
"
```

### Push le code sur GitHub

```bash
git add .
git commit -m "Fin de l'entraînement v1, modèle pushé sur Hugging Face"
git push origin main
```

---

## Étape 8 : Stopper le Pod

1. Quitte le terminal : `exit`
2. **Important** : va sur le dashboard RunPod → section *Pods* → bouton **Stop** pour arrêter la facturation GPU.

---

## Resynchroniser le PC local après un run

```bash
git fetch origin
git reset --hard origin/main
```

---

## Notes importantes

- Toujours utiliser `--no_quantize` : bitsandbytes est incompatible avec les GPU Blackwell et certaines configurations RunPod
- `models/`, `venv/`, `dataset/` et `*.log` sont dans `.gitignore` — ils ne sont pas pushés sur GitHub
- Les poids du modèle vont sur **Hugging Face**, pas sur GitHub
