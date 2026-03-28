# CLAUDE.md — ILab Smart Contract LoRA

## Objectif du projet

Fine-tuner **TinyLlama-1.1B** avec **LoRA** pour classifier des vulnérabilités dans des smart contracts Solidity.
Tâche : classification multi-classe (9 classes) — le modèle reçoit du code Solidity et doit prédire le type de vulnérabilité.

## Dataset

- Fichier : `dataset/dataset_9l_w_v2 (1).csv`
- Colonnes : `filename`, `code`, `label`, `label_encoded`
- **4068 échantillons**, 9 labels
- Distribution (déséquilibrée) :
  - Safe : 1000 (24.6%)
  - Unchecked external call (UC) : 600 (14.7%)
  - Reentrancy (RE) : 600 (14.7%)
  - Integer overflow (OF) : 590 (14.5%)
  - Block number dependency (BN) : 406 (10.0%)
  - Ether strict equality (SE) : 366 (9.0%)
  - Timestamp dependency (TP) : 312 (7.7%)
  - Dangerous delegatecall (DE) : 97 (2.4%)
  - Ether frozen (EF) : 97 (2.4%)
- Les labels bruts contiennent des préfixes de chemin (`./Dataset/reentrancy (RE)/`) — **les nettoyer** avant entraînement.
- Longueur du code : médiane ~7260 chars, max ~121k chars. Beaucoup dépassent le context window de TinyLlama (2048 tokens).

## Mapping des labels

```python
LABELS = {
    0: "Block number dependency",
    1: "Dangerous delegatecall",
    2: "Ether frozen",
    3: "Ether strict equality",
    4: "Integer overflow",
    5: "Reentrancy",
    6: "Timestamp dependency",
    7: "Unchecked external call",
    8: "Safe"
}
```

## Modèle

- **Base** : `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Méthode** : LoRA via PEFT
- **Context window** : 2048 tokens — les codes longs doivent être tronqués ou résumés
- Commencer petit, itérer, augmenter progressivement

## Contraintes techniques

### Prompt
- **En anglais** (TinyLlama est entraîné principalement en anglais)
- Le modèle doit répondre avec un seul chiffre (0-8)
- Format de prompt à optimiser pour la classification

### Troncature des codes longs
Le context window de TinyLlama (2048 tokens) est insuffisant pour beaucoup de contrats.
Stratégies à explorer :
1. **Troncature simple** : garder les N premiers tokens (le début du contrat contient souvent les patterns de vulnérabilité)
2. **Tête + queue** : garder le début et la fin du contrat
3. **Extraction ciblée** : extraire uniquement les fonctions pertinentes (transfer, call, delegatecall, etc.)

### Budget RunPod
- **Maximum 3 euros** pour l'ensemble des runs
- Être efficace : petits runs d'abord, valider l'approche, puis scaler
- GPU recommandé : le moins cher qui tient en mémoire (ex: RTX 3090 ou A40)

### Déséquilibre des classes
- DE et EF n'ont que 97 échantillons vs 1000 pour Safe
- Envisager : weighted loss, oversampling des classes minoritaires, ou stratified split

## Infrastructure

- **Entraînement** : RunPod (GPU cloud)
- **Dev/test local** : possible sur le PC de l'utilisateur (GTX 1660 Ti, 6 Go VRAM) pour des tests rapides
- Le code doit être pushable sur GitHub et exécutable sur RunPod (voir `Runpod-tuto.md`)

## Commandes

```bash
# --- Sur RunPod ---
cd /workspace
git clone <REPO_URL>
cd ILab-Smart_contract_LoRa
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Phase 1 : Baseline zero-shot
python evaluate.py --model_dir ./baseline_test --baseline --dataset "dataset/dataset_9l_w_v2 (1).csv" --max_samples 100

# Phase 2 : Quick test (200 samples, 1 epoch)
python train.py --max_samples 200 --epochs 1

# Phase 3 : Full training
python train.py --epochs 3

# Evaluation du modèle entraîné
python evaluate.py --model_dir models/tinyllama-lora_run-01_date-YYYY-MM-DD
```

## Fichiers du projet

- `train.py` — Script d'entraînement LoRA (QLoRA 4-bit, SFTTrainer)
- `evaluate.py` — Evaluation complète (accuracy, F1 macro/weighted, confusion matrix, classification report)
- `dataset/` — Contient le CSV du dataset
- `requirements.txt` — Dépendances Python
- `Runpod-tuto.md` — Guide de déploiement RunPod
- `models/` — Modèles sauvegardés (généré automatiquement, convention `{name}_run-{NN}_date-{YYYY-MM-DD}/`)

## Hyperparamètres par défaut

| Paramètre | Valeur | Justification |
|---|---|---|
| LoRA rank (r) | 16 | Bon compromis capacité/efficacité pour un petit modèle |
| LoRA alpha | 32 | Ratio alpha/r = 2 (standard) |
| LoRA dropout | 0.05 | Légère régularisation |
| Target modules | q, k, v, o_proj | Couvre toutes les projections d'attention |
| Learning rate | 2e-4 | Standard pour LoRA |
| LR scheduler | Cosine | Décroissance douce |
| Warmup | 10% des steps | Stabilise le début de l'entraînement |
| Batch size | 4 x 4 accum = 16 effective | Adapté au budget mémoire GPU |
| Quantization | 4-bit NF4 (QLoRA) | Réduit la VRAM ~4x |
| Max code tokens | 1500 | Laisse ~548 tokens pour le prompt template |
| Optimizer | paged_adamw_8bit | Économise la VRAM |

## Choix techniques

### Prompt en anglais avec chat template TinyLlama
Le prompt utilise le format natif `<|system|>...<|user|>...<|assistant|>` de TinyLlama-Chat.
Le modèle doit répondre avec un seul chiffre (0-8). Le parsing tolère du texte autour du chiffre.

### Troncature
Stratégie : garder les **premiers 1500 tokens** du code Solidity. Justification :
- Les imports, héritages, et signatures de fonctions sont au début
- Les patterns de vulnérabilité (reentrancy guard absent, unchecked call, etc.) apparaissent souvent dans les premières fonctions

### Métriques
- **F1 macro** : métrique principale (traite toutes les classes de manière égale malgré le déséquilibre)
- **F1 weighted** : pondéré par la fréquence des classes
- **Accuracy** : pour comparaison
- **Confusion matrix** : diagnostic visuel des erreurs
- **Classification report** : precision/recall/F1 par classe

## Approche itérative

1. **Phase 1** : Baseline — `evaluate.py --baseline` pour mesurer le zero-shot de TinyLlama
2. **Phase 2** : Quick test — `train.py --max_samples 200 --epochs 1` pour valider le pipeline
3. **Phase 3** : Training complet — `train.py --epochs 3` sur tout le dataset
4. **Phase 4** : Optimisation — ajuster hyperparamètres si nécessaire (lr, rank, epochs, troncature)
