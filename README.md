# Simulation V2X — Alerte piéton ITS-G5 / DSRC 5.9 GHz

Simulation d'un système d'alerte piéton-véhicule à une intersection urbaine.
Les véhicules et piétons sont simulés via **SUMO**, les alertes V2X sont transmises
sur le canal radio **5.9 GHz (ITS-G5 / DSRC)**, et le choix du chemin de transmission
est optimisé par des algorithmes **Multi-Armed Bandit (MAB)**.

---

## Architecture générale

```
Intersection (0,0)
       │
       │  piétons (sud → nord, 0.8–1.5 m/s)
       ↓
  ─────┼─────  route véhicules (ouest → est, 50 km/h)
       │

Chemin 1 : Véhicule ──── radio 5.9 GHz (direct) ────► Piéton
Chemin 2 : Véhicule ──► RSU (0,0) ──► radio 5.9 GHz ──► Piéton
```

### Composants

| Composant | Fichier | Rôle |
|---|---|---|
| Réseau routier | `simulation/network/nodes.nod.xml` + `edges.edg.xml` | Topologie : route E-O + passage piéton N-S |
| Scénario de trafic | `simulation/network/routes.rou.xml` | ~60 véhicules + ~40 piétons sur 300 s |
| Modèle de canal radio | `simulation/environment/channel_model.py` | Perte de trajet log-distance + ombrage log-normal + pénalité NLOS |
| Détection collision | `simulation/run_simulation.py` | Prédiction par **TTC** (Time-To-Collision) sur trajectoires rectilignes |
| Sélection du chemin | `simulation/run_simulation.py` | **MAB ε-greedy** (2 bras : `direct` / `via_rsu`) |
| Variante UCB | `simulation/run_simulation_ucb.py` | **MAB UCB1** — exploration déterministe sans ε |
| Sweep hyperparamètre | `simulation/sweep_eps0.py` | Lance N simulations pour différentes valeurs de ε₀ |

### Modèle de canal (5.9 GHz)

- **LOS** (`d < 50 m`) : `n = 2.5`, `σ = 4 dB`
- **NLOS** (`d ≥ 50 m`) : `n = 2.6`, `σ = 5 dB`, pénalité `+3 dB`
- **RSU** (toujours LOS, en hauteur) : `n = 2.2`, `σ = 4.5 dB`
- Sensibilité RX : `−85 dBm` | Puissance TX : `20 dBm`
- Probabilité de livraison : courbe sigmoïde centrée sur la sensibilité

### Algorithmes MAB

| Algorithme | Fichier | Paramètre clé |
|---|---|---|
| ε-greedy | `run_simulation.py` | `EPS0` (défaut 0.10) — taux d'exploration fixe |
| UCB1 | `run_simulation_ucb.py` | `UCB_C` (défaut 1.0) — coefficient exploration |

Les deux algorithmes choisissent à chaque collision entre `direct` et `via_rsu`,
reçoivent une récompense binaire (succès ≥ 50 % Pdlv), et mettent à jour leur
estimation Q(bras). Un **oracle idéal** est calculé en parallèle pour mesurer le regret.

---

## Prérequis

- **Python 3.10+**
- **SUMO** installé, avec `$SUMO_HOME` pointant vers le répertoire d'installation
  (ex. `C:\Program Files (x86)\Eclipse\Sumo`) et `$SUMO_HOME/bin` dans le `PATH`

---

## Installation

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

---

## Commandes

### 1. Générer le réseau (une seule fois)

```powershell
.venv\Scripts\python.exe simulation/setup_network.py
```

Produit `simulation/network/map.net.xml` à partir des fichiers nœuds/arêtes.

---

### 2. Simulation MAB ε-greedy

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe simulation/run_simulation.py
```

Paramétrage via variable d'environnement :

```powershell
$env:MAB_EPS0="0.20"   # taux d'exploration ε₀ (défaut 0.10)
.venv\Scripts\python.exe simulation/run_simulation.py
```

**Sorties** dans `resultats/` :
- `resultats_simulation.txt` — rapport texte complet
- `courbes_v2x.png` — RSSI, TTC, alertes cumulées, latences
- `comparaison_mab_ideal.png` — MAB vs oracle idéal, regret, exploration

---

### 3. Simulation MAB UCB1

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe simulation/run_simulation_ucb.py
```

Paramétrage :

```powershell
$env:MAB_UCB_C="1.414"  # coefficient exploration UCB (défaut 1.0)
.venv\Scripts\python.exe simulation/run_simulation_ucb.py
```

**Sorties** dans `resultats/` :
- `resultatsUCB.txt` — rapport texte
- `courbes_v2x_ucb.png` — courbes V2X
- `comparaison_ucb_ideal.png` — Q-valeurs, bonus UCB, regret

---

### 4. Sweep ε₀ (sensibilité au paramètre d'exploration)

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe simulation/sweep_eps0.py
```

Valeurs personnalisées :

```powershell
.venv\Scripts\python.exe simulation/sweep_eps0.py 0 0.05 0.1 0.2 0.5 1.0
```

**Sorties** dans `resultats/` :
- `sweep_eps0_taux_reussite.png`
- `sweep_eps0_taux_reussite.csv`
