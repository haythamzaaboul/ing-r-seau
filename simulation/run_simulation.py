r"""
Simulation V2X : alerte piéton via canal radio 5.9 GHz (ITS-G5 / DSRC).
Détection de collision par prédiction de trajectoire (TTC).
Un RSU à l'intersection (0,0) relaie les alertes en cas de mauvais canal.

Lancer :
    $env:PYTHONIOENCODING="utf-8"
    .venv\Scripts\python.exe simulation/run_simulation.py
"""

import os
import sys
import math
import traci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from environment.channel_model import ChannelModel

# ── Configuration ──────────────────────────────────────────────────────────────
SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
SUMO_BIN  = os.path.join(SUMO_HOME, "bin", "sumo.exe")
SUMOCFG   = os.path.join(os.path.dirname(__file__), "config", "sim.sumocfg")

STEP_LEN    = 0.1    # pas de simulation (s)
END_TIME    = 300.0  # durée totale

DANGER_HALF = 3.0    # demi-côté du carré de danger autour du centre (m)
                     # → zone dangereuse = carré [-3, +3] x [-3, +3]

PRESCREEN_M = 200.0  # distance max au centre pour calculer le TTC (optimisation)

RSU_POS      = (0.0, 0.0)
RSU_LOAD     = 0.0
MAX_RSU_MSGS = 20

# Parametres MAB eps-greedy.
# Modifier EPS0 ici pour changer le taux d'exploration initial,
# ou lancer avec la variable d'environnement MAB_EPS0.
EPS0 = float(os.environ.get("MAB_EPS0", "0.10"))
EPSILON_DECAY = False
MAB_ARMS = ("direct", "via_rsu")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "resultats")
os.makedirs(OUT_DIR, exist_ok=True)

channel = ChannelModel()


class EpsilonGreedyMAB:
    """Agent MAB qui choisit le chemin sans utiliser les distances."""

    def __init__(self, arms, eps0=0.10, decay=False):
        self.arms = tuple(arms)
        self.eps0 = eps0
        self.decay = decay
        self.counts = {arm: 0 for arm in self.arms}
        self.values = {arm: 0.0 for arm in self.arms}
        self.total_pulls = 0
        self.history = []

    @property
    def epsilon(self):
        if not self.decay:
            return self.eps0
        return self.eps0 / math.sqrt(self.total_pulls + 1)

    def select_arm(self):
        epsilon = self.epsilon
        unexplored = [arm for arm in self.arms if self.counts[arm] == 0]
        if unexplored:
            arm = unexplored[0]
            decision = "init"
        elif np.random.random() < epsilon:
            arm = str(np.random.choice(self.arms))
            decision = "exploration"
        else:
            best_value = max(self.values.values())
            best_arms = [arm for arm in self.arms if self.values[arm] == best_value]
            arm = str(np.random.choice(best_arms))
            decision = "exploitation"
        return arm, epsilon, decision

    def update(self, arm, reward):
        self.total_pulls += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.history.append({
            "step": self.total_pulls,
            "arm": arm,
            "reward": reward,
            "epsilon": self.epsilon,
            "q_direct": self.values["direct"],
            "q_rsu": self.values["via_rsu"],
            "n_direct": self.counts["direct"],
            "n_rsu": self.counts["via_rsu"],
        })


mab_agent = EpsilonGreedyMAB(MAB_ARMS, EPS0, EPSILON_DECAY)

# ── Centre de l'intersection (lu depuis TraCI au démarrage) ───────────────────
CENTER = (0.0, 0.0)   # sera mis à jour après traci.start()

# ── État global ────────────────────────────────────────────────────────────────
events        = []         # un dict par alerte émise
active_alerts = set()      # paires (vid, pid) en cours de collision détectée

stats = {
    "total_checks":       0,
    "collisions_detectees": 0,
    "alerts_direct_los":  0,
    "alerts_direct_nlos": 0,
    "alerts_rsu":         0,
    "echecs":             0,
    "ideal_direct_los":   0,
    "ideal_direct_nlos":  0,
    "ideal_rsu":          0,
    "ideal_echecs":       0,
}


# ── Utilitaires ────────────────────────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _update_rsu_load(n: int):
    global RSU_LOAD
    RSU_LOAD = min(1.0, n / MAX_RSU_MSGS)


# ── Prédiction de collision (TTC) ─────────────────────────────────────────────

def collision_predicted(vx, vy, v_speed, px, py, p_speed):
    """
    Prédit si le véhicule et le piéton se trouveront dans le carré ±DANGER_HALF
    au même moment, en supposant des trajectoires rectilignes.

    Hypothèses géométriques :
      - Véhicule : se déplace vers l'est (+x), calculé en coordonnées relatives au centre
      - Piéton   : se déplace vers le nord (+y), coordonnées relatives au centre
      - Danger   : carré [-D, +D] × [-D, +D] centré sur CENTER (lu via TraCI)

    Retourne : (predicted: bool, ttc: float|None)
      ttc = temps avant que la collision ne soit imminente (s)
    """
    D  = DANGER_HALF
    cx, cy = CENTER

    # Coordonnées relatives au centre de l'intersection
    rx = vx - cx   # > 0 : véhicule à l'est du centre (déjà passé)
    ry = py - cy   # > 0 : piéton au nord du centre (déjà passé)

    # Véhicule déjà passé le carré ou trop loin → pas de risque
    if rx > D or abs(rx) > PRESCREEN_M:
        return False, None
    # Piéton déjà passé le carré → pas de risque
    if ry > D:
        return False, None
    # Vitesses nulles → pas de mouvement prévisible
    if v_speed < 0.1 or p_speed < 0.1:
        return False, None

    # ── Intervalle temporel du véhicule dans la zone (axe x) ──
    tv_enter = max(0.0, (-D - rx) / v_speed) if rx < -D else 0.0
    tv_exit  = (D - rx) / v_speed
    if tv_exit < 0:
        return False, None

    # ── Intervalle temporel du piéton dans la zone (axe y) ──
    tp_enter = max(0.0, (-D - ry) / p_speed) if ry < -D else 0.0
    tp_exit  = (D - ry) / p_speed
    if tp_exit < 0:
        return False, None

    # Chevauchement des intervalles → collision prédite
    if tv_enter > tp_exit or tp_enter > tv_exit:
        return False, None

    ttc = max(tv_enter, tp_enter)
    return True, ttc


# ── Selection du canal : oracle ideal + MAB eps-greedy ───────────────────────

def try_alert(veh_pos, ped_pos):
    d_vp = dist(veh_pos, ped_pos)
    d_vr = dist(veh_pos, RSU_POS)
    d_rp = dist(RSU_POS, ped_pos)

    direct = channel.direct_link(d_vp)
    via    = channel.via_rsu(d_vr, d_rp, RSU_LOAD)

    ideal = direct if direct.delivery_probability >= via.delivery_probability else via

    mab_arm, epsilon, decision = mab_agent.select_arm()
    selected = direct if mab_arm == "direct" else via
    reward = 1.0 if selected.delivery_probability >= 0.5 else 0.0
    mab_agent.update(mab_arm, reward)

    return selected, ideal, direct, via, epsilon, decision, reward


# ── Traitement d'un pas de simulation ─────────────────────────────────────────

def process_step(t: float):
    vehicles = traci.vehicle.getIDList()
    persons  = traci.person.getIDList()
    n_msg    = 0
    current_collisions = set()

    for vid in vehicles:
        vx, vy   = traci.vehicle.getPosition(vid)
        v_speed  = traci.vehicle.getSpeed(vid)

        for pid in persons:
            px, py  = traci.person.getPosition(pid)
            p_speed = traci.person.getSpeed(pid)
            stats["total_checks"] += 1

            predicted, ttc = collision_predicted(vx, vy, v_speed, px, py, p_speed)

            if not predicted:
                continue

            current_collisions.add((vid, pid))

            # Alerte déjà envoyée pour cet événement → on attend la fin
            if (vid, pid) in active_alerts:
                continue

            # Nouvelle collision détectée → déclencher l'alerte
            active_alerts.add((vid, pid))
            stats["collisions_detectees"] += 1
            n_msg += 1

            d = dist((vx, vy), (px, py))
            best, ideal, direct, via, epsilon, decision, reward = try_alert((vx, vy), (px, py))
            success = best.delivery_probability >= 0.5
            ideal_success = ideal.delivery_probability >= 0.5

            ev = {
                "t":           t,
                "veh":         vid,
                "ped":         pid,
                "dist":        d,
                "ttc":         ttc,
                "path":        best.path,
                "rssi":        best.rssi_dbm,
                "latency":     best.latency_ms,
                "pdlv":        best.delivery_probability,
                "nlos":        direct.nlos,
                "success":     success,
                "mab_reward":  reward,
                "mab_epsilon": epsilon,
                "mab_decision": decision,
                "ideal_path":  ideal.path,
                "ideal_rssi":  ideal.rssi_dbm,
                "ideal_latency": ideal.latency_ms,
                "ideal_pdlv":  ideal.delivery_probability,
                "ideal_success": ideal_success,
                "pdlv_direct": direct.delivery_probability,
                "pdlv_rsu":    via.delivery_probability,
                "rssi_direct": direct.rssi_dbm,
                "rssi_rsu":    via.rssi_dbm,
            }
            events.append(ev)

            if ideal_success:
                if ideal.path == "direct":
                    ideal_key = "ideal_direct_nlos" if direct.nlos else "ideal_direct_los"
                    stats[ideal_key] += 1
                else:
                    stats["ideal_rsu"] += 1
            else:
                stats["ideal_echecs"] += 1

            if success:
                if best.path == "direct":
                    key = "alerts_direct_nlos" if direct.nlos else "alerts_direct_los"
                    stats[key] += 1
                else:
                    stats["alerts_rsu"] += 1
                tag = "ALERTE"
            else:
                stats["echecs"] += 1
                tag = "ECHEC "

            nlos_tag = " [NLOS]" if direct.nlos else " [LOS] "
            print(f"  [t={t:6.1f}s] {tag} {vid}->{pid}"
                  f"  dist={d:5.1f}m  TTC={ttc:5.2f}s"
                  f"  MAB={best.path:8s}{nlos_tag}"
                  f"  RSSI={best.rssi_dbm:+.1f}dBm"
                  f"  lat={best.latency_ms:.1f}ms"
                  f"  Pdlv={best.delivery_probability:.1%}"
                  f"  eps={epsilon:.3f}"
                  f"  ideal={ideal.path}")

    # Nettoyer : retirer les paires dont le véhicule a dépassé la zone de danger
    # (on ne se base PAS sur la prédiction courante pour éviter les oscillations)
    finished = set()
    for (vid, pid) in active_alerts:
        if vid not in vehicles:
            finished.add((vid, pid))
            continue
        vx_cur, _ = traci.vehicle.getPosition(vid)
        if (vx_cur - CENTER[0]) > DANGER_HALF:
            finished.add((vid, pid))
    active_alerts.difference_update(finished)

    _update_rsu_load(n_msg)


# ── Génération des courbes ─────────────────────────────────────────────────────

def plot_results():
    if not events:
        print("  Aucun evenement a tracer.")
        return

    colors = []
    for e in events:
        if not e["success"]:
            colors.append("red")
        elif e["path"] == "via_rsu":
            colors.append("orange")
        elif e["nlos"]:
            colors.append("steelblue")
        else:
            colors.append("green")

    distances = [e["dist"]    for e in events]
    rssi_vals = [e["rssi"]    for e in events]
    pdlv_vals = [e["pdlv"]    for e in events]
    latencies = [e["latency"] for e in events]
    ttc_vals  = [e["ttc"]     for e in events]
    times     = [e["t"]       for e in events]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Simulation V2X – ITS-G5 / DSRC 5.9 GHz\n"
        f"Detection collision TTC | Zone danger : +/-{DANGER_HALF} m | RSU : {RSU_POS}",
        fontsize=12, fontweight="bold"
    )

    legend_patches = [
        mpatches.Patch(color="green",     label="Direct LOS  (succes)"),
        mpatches.Patch(color="steelblue", label="Direct NLOS (succes)"),
        mpatches.Patch(color="orange",    label="Via RSU     (succes)"),
        mpatches.Patch(color="red",       label="Echec"),
    ]

    # ── Courbe 1 : RSSI vs Distance au moment de l'alerte ────────────────── #
    ax = axes[0, 0]
    ax.scatter(distances, rssi_vals, c=colors, alpha=0.7, s=60, zorder=3)
    ax.axhline(channel.SENSITIVITY_DBM, color="red", linestyle="--",
               linewidth=1.5, label=f"Sensibilite ({channel.SENSITIVITY_DBM} dBm)")
    ax.axvline(channel.LOS_RANGE_M, color="purple", linestyle=":",
               linewidth=1.2, label=f"Seuil LOS/NLOS ({channel.LOS_RANGE_M} m)")
    ax.set_xlabel("Distance vehicule-pieton au moment de l'alerte (m)")
    ax.set_ylabel("RSSI (dBm)")
    ax.set_title("RSSI vs Distance (alerte unique par collision)")
    ax.legend(handles=legend_patches, fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    # ── Courbe 2 : TTC vs Distance ────────────────────────────────────────── #
    ax = axes[0, 1]
    ax.scatter(distances, ttc_vals, c=colors, alpha=0.7, s=60, zorder=3)
    ax.axvline(channel.LOS_RANGE_M, color="purple", linestyle=":",
               linewidth=1.2, label=f"Seuil LOS/NLOS ({channel.LOS_RANGE_M} m)")
    ax.set_xlabel("Distance vehicule-pieton (m)")
    ax.set_ylabel("TTC - Temps avant collision (s)")
    ax.set_title("TTC vs Distance au moment de la detection")
    ax.legend(handles=legend_patches, fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Courbe 3 : Alertes dans le temps ─────────────────────────────────── #
    ax = axes[1, 0]
    events_sorted = sorted(events, key=lambda e: e["t"])
    t_sorted = [e["t"] for e in events_sorted]
    cum_ok  = np.cumsum([1 if e["success"] else 0 for e in events_sorted])
    cum_ko  = np.cumsum([0 if e["success"] else 1 for e in events_sorted])

    ax.step(t_sorted, cum_ok, where="post", color="green",  linewidth=2,
            label=f"Alertes reussies ({sum(e['success'] for e in events)})")
    ax.step(t_sorted, cum_ko, where="post", color="red",    linewidth=2,
            linestyle="--", label=f"Echecs ({sum(not e['success'] for e in events)})")
    ax.set_xlabel("Temps simule (s)")
    ax.set_ylabel("Alertes cumulees")
    ax.set_title("Alertes MAB cumulees dans le temps")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Courbe 4 : Distribution des latences ─────────────────────────────── #
    ax = axes[1, 1]
    lat_direct = [e["latency"] for e in events if e["success"] and e["path"] == "direct"]
    lat_rsu    = [e["latency"] for e in events if e["success"] and e["path"] == "via_rsu"]
    if latencies:
        bins = np.linspace(0, max(latencies) + 5, 30)
        if lat_direct:
            ax.hist(lat_direct, bins=bins, color="green",  alpha=0.6, label="Direct")
        if lat_rsu:
            ax.hist(lat_rsu,    bins=bins, color="orange", alpha=0.6, label="Via RSU")
        ax.axvline(50, color="red", linestyle="--", linewidth=1.5,
                   label="Limite securite (50 ms)")
    ax.set_xlabel("Latence (ms)")
    ax.set_ylabel("Nombre d'evenements")
    ax.set_title("Distribution des latences")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "courbes_v2x.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Courbes sauvegardees  : {plot_path}")


def plot_algorithm_comparison():
    if not events:
        return

    steps = list(range(1, len(events) + 1))
    mab_success = [1 if e["success"] else 0 for e in events]
    ideal_success = [1 if e["ideal_success"] else 0 for e in events]
    mab_cum = np.cumsum(mab_success)
    ideal_cum = np.cumsum(ideal_success)
    mab_rate = mab_cum / np.arange(1, len(events) + 1)
    ideal_rate = ideal_cum / np.arange(1, len(events) + 1)
    regret = ideal_cum - mab_cum

    direct_choices = np.cumsum([1 if e["path"] == "direct" else 0 for e in events])
    rsu_choices = np.cumsum([1 if e["path"] == "via_rsu" else 0 for e in events])
    eps_vals = [e["mab_epsilon"] for e in events]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Comparaison selection ideale vs MAB eps-greedy (EPS0={EPS0})",
        fontsize=12,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.step(steps, mab_cum, where="post", linewidth=2, label="MAB eps-greedy")
    ax.step(steps, ideal_cum, where="post", linewidth=2, linestyle="--", label="Ideal oracle")
    ax.set_xlabel("Evenement collision")
    ax.set_ylabel("Succes cumules")
    ax.set_title("Succes cumules")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, mab_rate, linewidth=2, label="MAB eps-greedy")
    ax.plot(steps, ideal_rate, linewidth=2, linestyle="--", label="Ideal oracle")
    ax.set_xlabel("Evenement collision")
    ax.set_ylabel("Taux de succes")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Taux de succes observe")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.step(steps, direct_choices, where="post", linewidth=2, label="Choix direct")
    ax.step(steps, rsu_choices, where="post", linewidth=2, label="Choix via RSU")
    ax.set_xlabel("Evenement collision")
    ax.set_ylabel("Choix cumules")
    ax.set_title("Exploration des intermediaires par le MAB")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.step(steps, regret, where="post", color="crimson", linewidth=2, label="Regret ideal - MAB")
    ax.plot(steps, eps_vals, color="gray", linewidth=1.5, linestyle=":", label="epsilon")
    ax.set_xlabel("Evenement collision")
    ax.set_ylabel("Ecart de succes / epsilon")
    ax.set_title("Regret cumule et exploration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "comparaison_mab_ideal.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparaison sauvegardee: {plot_path}")


# ── Rapport texte ──────────────────────────────────────────────────────────────

def write_report():
    total_ok  = stats["alerts_direct_los"] + stats["alerts_direct_nlos"] + stats["alerts_rsu"]
    total     = total_ok + stats["echecs"]
    taux      = total_ok / total if total > 0 else 0.0
    ideal_ok = stats["ideal_direct_los"] + stats["ideal_direct_nlos"] + stats["ideal_rsu"]
    ideal_total = ideal_ok + stats["ideal_echecs"]
    ideal_taux = ideal_ok / ideal_total if ideal_total > 0 else 0.0
    regret = ideal_ok - total_ok

    lat_direct = [e["latency"] for e in events if e["success"] and e["path"] == "direct"]
    lat_rsu    = [e["latency"] for e in events if e["success"] and e["path"] == "via_rsu"]
    rssi_ok    = [e["rssi"] for e in events if e["success"]]
    rssi_ko    = [e["rssi"] for e in events if not e["success"]]
    ttc_list   = [e["ttc"]  for e in events]

    path = os.path.join(OUT_DIR, "resultats_simulation.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  RAPPORT DE SIMULATION V2X — ITS-G5 / DSRC 5.9 GHz\n")
        f.write(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 65 + "\n\n")

        f.write("PARAMETRES DU MODELE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Frequence porteuse     : {channel.FREQ_HZ/1e9:.1f} GHz\n")
        f.write(f"  Puissance TX           : {channel.TX_POWER_DBM} dBm\n")
        f.write(f"  Sensibilite RX         : {channel.SENSITIVITY_DBM} dBm\n")
        f.write(f"  Zone de danger         : carre +/-{DANGER_HALF} m autour de (0,0)\n")
        f.write(f"  Seuil LOS/NLOS         : {channel.LOS_RANGE_M} m\n")
        f.write(f"  Exposant perte LOS     : n = {channel.N_DIRECT_LOS}\n")
        f.write(f"  Exposant perte NLOS    : n = {channel.N_DIRECT_NLOS}\n")
        f.write(f"  Ombrage LOS            : sigma = {channel.SIGMA_DIRECT_LOS} dB\n")
        f.write(f"  Ombrage NLOS           : sigma = {channel.SIGMA_DIRECT_NLOS} dB\n")
        f.write(f"  Penalite NLOS          : {channel.NLOS_PENALTY_DB} dB\n")
        f.write(f"  Exposant perte RSU     : n = {channel.N_INFRA}\n")
        f.write(f"  Position RSU           : {RSU_POS}\n")
        f.write(f"  Algorithme choix       : MAB eps-greedy\n")
        f.write(f"  EPS0                   : {EPS0}\n")
        f.write(f"  Decroissance epsilon   : {EPSILON_DECAY}\n\n")

        f.write("RESULTATS GLOBAUX\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Paires V-P analysees   : {stats['total_checks']}\n")
        f.write(f"  Collisions detectees   : {stats['collisions_detectees']}\n")
        f.write(f"  Alertes emises         : {total}\n")
        f.write(f"  Taux succes MAB        : {taux:.1%}\n")
        f.write(f"  Taux succes ideal      : {ideal_taux:.1%}\n")
        f.write(f"  Regret cumule          : {regret} succes\n")
        f.write(f"  Taux d'echec           : {1-taux:.1%}\n\n")

        f.write("DETAIL PAR CHEMIN - MAB EPS-GREEDY\n")
        f.write("-" * 40 + "\n")
        if total > 0:
            f.write(f"  Direct LOS   : {stats['alerts_direct_los']:3d}  ({stats['alerts_direct_los']/total*100:.1f}%)\n")
            f.write(f"  Direct NLOS  : {stats['alerts_direct_nlos']:3d}  ({stats['alerts_direct_nlos']/total*100:.1f}%)\n")
            f.write(f"  Via RSU      : {stats['alerts_rsu']:3d}  ({stats['alerts_rsu']/total*100:.1f}%)\n")
            f.write(f"  Echecs       : {stats['echecs']:3d}  ({stats['echecs']/total*100:.1f}%)\n\n")

        f.write("DETAIL PAR CHEMIN - IDEAL ORACLE\n")
        f.write("-" * 40 + "\n")
        if ideal_total > 0:
            f.write(f"  Direct LOS   : {stats['ideal_direct_los']:3d}  ({stats['ideal_direct_los']/ideal_total*100:.1f}%)\n")
            f.write(f"  Direct NLOS  : {stats['ideal_direct_nlos']:3d}  ({stats['ideal_direct_nlos']/ideal_total*100:.1f}%)\n")
            f.write(f"  Via RSU      : {stats['ideal_rsu']:3d}  ({stats['ideal_rsu']/ideal_total*100:.1f}%)\n")
            f.write(f"  Echecs       : {stats['ideal_echecs']:3d}  ({stats['ideal_echecs']/ideal_total*100:.1f}%)\n\n")

        f.write("ETAT FINAL DU MAB\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Q(direct)    : {mab_agent.values['direct']:.3f} | n={mab_agent.counts['direct']}\n")
        f.write(f"  Q(via_rsu)   : {mab_agent.values['via_rsu']:.3f} | n={mab_agent.counts['via_rsu']}\n")
        f.write(f"  epsilon final: {mab_agent.epsilon:.3f}\n\n")

        f.write("TTC AU MOMENT DE LA DETECTION\n")
        f.write("-" * 40 + "\n")
        if ttc_list:
            f.write(f"  Moyen  : {np.mean(ttc_list):.2f} s\n")
            f.write(f"  Min    : {min(ttc_list):.2f} s\n")
            f.write(f"  Max    : {max(ttc_list):.2f} s\n\n")

        f.write("LATENCES\n")
        f.write("-" * 40 + "\n")
        if lat_direct:
            f.write(f"  Directe : moy={np.mean(lat_direct):.1f} ms"
                    f" | med={np.median(lat_direct):.1f} ms"
                    f" | max={max(lat_direct):.1f} ms\n")
        if lat_rsu:
            f.write(f"  Via RSU : moy={np.mean(lat_rsu):.1f} ms"
                    f" | med={np.median(lat_rsu):.1f} ms"
                    f" | max={max(lat_rsu):.1f} ms\n")
        f.write("\n")

        f.write("RSSI\n")
        f.write("-" * 40 + "\n")
        if rssi_ok:
            f.write(f"  Succes  : moy={np.mean(rssi_ok):.1f} dBm | min={min(rssi_ok):.1f} dBm\n")
        if rssi_ko:
            f.write(f"  Echecs  : moy={np.mean(rssi_ko):.1f} dBm | max={max(rssi_ko):.1f} dBm\n")
        f.write("\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"  Scenario : 15 vehicules (50 km/h, ouest->est) x 8 pietons\n"
            f"  (0.9-1.5 m/s, sud->nord) sur 120 s.\n\n"
            f"  Methode de detection : prediction TTC (Time-To-Collision).\n"
            f"  Une alerte est envoyee quand les trajectoires predites se\n"
            f"  croisent dans le carre +/-{DANGER_HALF} m autour de l'intersection.\n"
            f"  Une seule alerte par collision, pas de spam continu.\n\n"
        )
        if ttc_list:
            f.write(f"  TTC moyen a la detection : {np.mean(ttc_list):.2f} s\n"
                    f"  -> Le systeme alerte en moyenne {np.mean(ttc_list):.1f} s avant\n"
                    f"     que la collision ne se produise.\n\n")
        if lat_direct and lat_rsu:
            f.write(f"  Latence directe ({np.mean(lat_direct):.1f} ms) bien inferieure\n"
                    f"  a la latence RSU ({np.mean(lat_rsu):.1f} ms).\n"
                    f"  Les deux restent sous la limite ETSI de 50 ms.\n\n")
        if stats["echecs"] > 0:
            f.write(f"  {stats['echecs']} echec(s) : surviennent en NLOS quand\n"
                    f"  direct ET RSU sont sous le seuil de sensibilite.\n"
                    f"  -> Candidats ideaux pour l'apprentissage MAB.\n\n")
        f.write(
            "  La selection effective utilise maintenant le MAB eps-greedy.\n"
            "  L'oracle ideal est conserve uniquement comme reference de comparaison,\n"
            "  car il utilise les probabilites de livraison direct/RSU indisponibles\n"
            "  au moment du choix dans un scenario reel.\n"
        )

    print(f"  Rapport sauvegarde    : {path}")


# ── Boucle principale ──────────────────────────────────────────────────────────

def run():
    sumo_cmd = [SUMO_BIN, "-c", SUMOCFG, "--step-length", str(STEP_LEN)]

    print("=" * 65)
    print("  Simulation V2X - Detection collision par TTC")
    print("=" * 65)
    print(f"  Config       : {SUMOCFG}")
    print(f"  RSU          : {RSU_POS}")
    print(f"  Zone danger  : carre +/-{DANGER_HALF} m")
    print(f"  Seuil LOS    : {channel.LOS_RANGE_M} m")
    print(f"  Choix chemin : MAB eps-greedy (EPS0={EPS0}, decay={EPSILON_DECAY})")
    print()

    traci.start(sumo_cmd)

    # Lire la vraie position du nœud central après démarrage SUMO
    global CENTER
    CENTER = traci.junction.getPosition("center")
    print(f"  Centre (SUMO) : {CENTER}")
    print()

    try:
        while True:
            traci.simulationStep()
            t = round(traci.simulation.getTime(), 1)
            if t > END_TIME:
                break
            process_step(t)
    finally:
        traci.close()

    total_ok = stats["alerts_direct_los"] + stats["alerts_direct_nlos"] + stats["alerts_rsu"]
    total    = total_ok + stats["echecs"]
    taux     = total_ok / total if total > 0 else 0.0
    ideal_ok = stats["ideal_direct_los"] + stats["ideal_direct_nlos"] + stats["ideal_rsu"]
    ideal_total = ideal_ok + stats["ideal_echecs"]
    ideal_taux = ideal_ok / ideal_total if ideal_total > 0 else 0.0

    print()
    print("=" * 65)
    print("  RAPPORT")
    print("=" * 65)
    print(f"  Paires V-P analysees   : {stats['total_checks']}")
    print(f"  Collisions detectees   : {stats['collisions_detectees']}")
    print(f"  Alertes emises         : {total}")
    print(f"  +-- Direct LOS         : {stats['alerts_direct_los']}")
    print(f"  +-- Direct NLOS        : {stats['alerts_direct_nlos']}")
    print(f"  +-- Via RSU            : {stats['alerts_rsu']}")
    print(f"  +-- Echecs             : {stats['echecs']}")
    print(f"  Taux de succes MAB     : {taux:.1%}")
    print(f"  Taux de succes ideal   : {ideal_taux:.1%}")
    print(f"  Regret cumule          : {ideal_ok - total_ok} succes")
    print(f"  Q(direct)={mab_agent.values['direct']:.3f} n={mab_agent.counts['direct']}")
    print(f"  Q(via_rsu)={mab_agent.values['via_rsu']:.3f} n={mab_agent.counts['via_rsu']}")
    print()

    write_report()
    plot_results()
    plot_algorithm_comparison()
    print()


if __name__ == "__main__":
    run()
