# Guide : Entraînement LoRA sur RunPod & Sauvegarde (GitHub + Hugging Face)

Ce guide détaille les étapes pour cloner le projet depuis GitHub, configurer l'environnement GPU sur RunPod, lancer l'entraînement LoRA en arrière-plan, et sauvegarder les résultats.

---

## Étape 1 : Créer un Pod sur RunPod

- Va sur [runpod.io](https://runpod.io) → **Deploy**
- Choisis un GPU : **RTX 3090 ou A40** (meilleur rapport prix/VRAM pour rester dans le budget de 3€)
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
pip install -r requirements.txt && echo "Ca marche !"
```

---

## Étape 4 : Authentification (GitHub & Hugging Face)

### GitHub (pour push le code en fin de run)

```bash
git config --global user.name "Thibault-GAREL"
git config --global user.email "ton-email@exemple.com"
git config --global credential.helper store
```

Note : lors du premier `git push`, il demandera un Personal Access Token GitHub en guise de mot de passe.

### Hugging Face (pour sauvegarder les poids du modèle)

```bash
huggingface-cli login
```

Note : génère un token Write dans Settings > Access Tokens sur huggingface.co, puis colle-le.

---

## Étape 5 : Lancer l'entraînement en arrière-plan

Utilise `nohup` pour que l'entraînement survive à une déconnexion (`screen` et `tmux` ne sont pas disponibles sur RunPod par défaut).

Lance les phases dans l'ordre — ne pas sauter directement à la Phase 3 sans valider les phases précédentes.

### Phase 1 — Baseline zero-shot (test rapide, coût négligeable)

```bash
python evaluate.py --baseline --dataset "dataset/dataset_9l_w_v2 (1).csv" --max_samples 100
```

### Phase 2 — Quick test (valide que le pipeline fonctionne)

```bash
nohup python train.py --max_samples 200 --epochs 1 > training_phase2.log 2>&1 &
```

### Phase 3 — Entraînement complet

```bash
nohup python train.py --epochs 3 > training.log 2>&1 &
```

Suivre les logs en direct (Ctrl+C pour arrêter l'affichage sans arrêter l'entraînement) :

```bash
tail -f training.log
```

---

## Étape 6 : Sauvegarder les résultats

### Push le modèle sur Hugging Face

```bash
huggingface-cli upload Thibault-GAREL/smart-contract-lora ./models/<nom_du_run> --commit-message "LoRA v1 - full training"
```

### Push le code sur GitHub

```bash
git add .
git commit -m "Fin de l'entraînement v1, modèle pushé sur Hugging Face"
git push origin main
```

---

## Étape 7 : Stopper le Pod

1. Quitte le terminal : `exit`
2. **Important** : va sur le dashboard RunPod → section *Pods* → bouton **Stop** pour arrêter la facturation GPU.

---

## Astuce : Push automatique depuis le code Python

Tu peux ajouter ces lignes à la fin de `train.py` pour automatiser l'upload :

```python
model.push_to_hub("Thibault-GAREL/smart-contract-lora", token="TON_TOKEN_WRITE")
tokenizer.push_to_hub("Thibault-GAREL/smart-contract-lora", token="TON_TOKEN_WRITE")
```
