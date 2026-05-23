r"""
Simulation V2X : alerte piéton via canal radio 5.9 GHz (ITS-G5 / DSRC).
Détection de collision par prédiction de trajectoire (TTC).
Sélection du chemin par algorithme MAB UCB1 (Upper Confidence Bound).

Lancer :
    $env:PYTHONIOENCODING="utf-8"
    .venv\Scripts\python.exe simulation/run_simulation_ucb.py

Paramètre clé UCB : UCB_C (variable d'env MAB_UCB_C, défaut 1.0)
  Petit (0.5)   -> peu d'exploration, exploitation rapide
  Moyen (1.0)   -> équilibre exploration/exploitation
  Grand (1.414) -> exploration agressive (théoriquement optimal pour rewards ∈ [0,1])
  Très grand (2.0+) -> sur-exploration
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
PRESCREEN_M = 200.0  # distance max au centre pour calculer le TTC

RSU_POS      = (0.0, 0.0)
RSU_LOAD     = 0.0
MAX_RSU_MSGS = 20

# Paramètre UCB : coefficient d'exploration.
# Plus UCB_C est grand, plus l'agent explore avant d'exploiter.
# Modifier UCB_C ici ou via la variable d'environnement MAB_UCB_C.
UCB_C    = float(os.environ.get("MAB_UCB_C", "1.0"))
MAB_ARMS = ("direct", "via_rsu")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "resultats")
os.makedirs(OUT_DIR, exist_ok=True)

channel = ChannelModel()


class UCBAgent:
    """
    Agent MAB UCB1.

    Règle de sélection : argmax_a [ Q(a) + c * sqrt( ln(t) / N(a) ) ]
      Q(a)  = récompense moyenne estimée du bras a
      t     = nombre total de tirages
      N(a)  = nombre de fois que le bras a a été tiré
      c     = coefficient d'exploration (UCB_C)

    Propriété clé : le bonus sqrt(ln(t)/N(a)) diminue automatiquement
    à mesure qu'un bras est exploré → pas de randomness après init.
    """

    def __init__(self, arms, c=1.0):
        self.arms   = tuple(arms)
        self.c      = c
        self.counts = {arm: 0   for arm in self.arms}
        self.values = {arm: 0.0 for arm in self.arms}
        self.total_pulls = 0
        self.history = []

    def ucb_score(self, arm):
        """Calcule le score UCB1 d'un bras."""
        if self.counts[arm] == 0:
            return float("inf")
        exploration_bonus = self.c * math.sqrt(math.log(self.total_pulls) / self.counts[arm])
        return self.values[arm] + exploration_bonus

    def select_arm(self):
        # Phase d'initialisation : tirer chaque bras une fois
        unexplored = [arm for arm in self.arms if self.counts[arm] == 0]
        if unexplored:
            arm      = unexplored[0]
            decision = "init"
            bonus    = float("inf")
        else:
            scores = {arm: self.ucb_score(arm) for arm in self.arms}
            arm    = max(scores, key=scores.get)
            bonus  = self.c * math.sqrt(math.log(self.total_pulls) / self.counts[arm])
            # Exploitation si bonus < Q(arm), sinon exploration
            decision = "exploitation" if self.values[arm] >= bonus else "exploration"

        return arm, bonus, decision

    def update(self, arm, reward):
        self.total_pulls    += 1
        self.counts[arm]    += 1
        n                    = self.counts[arm]
        self.values[arm]    += (reward - self.values[arm]) / n

        # Calcul des bonus courants pour l'historique
        bonus_direct = (
            self.c * math.sqrt(math.log(self.total_pulls) / self.counts["direct"])
            if self.counts["direct"] > 0 else float("inf")
        )
        bonus_rsu = (
            self.c * math.sqrt(math.log(self.total_pulls) / self.counts["via_rsu"])
            if self.counts["via_rsu"] > 0 else float("inf")
        )

        self.history.append({
            "step":          self.total_pulls,
            "arm":           arm,
            "reward":        reward,
            "ucb_bonus":     bonus_direct if arm == "direct" else bonus_rsu,
            "q_direct":      self.values["direct"],
            "q_rsu":         self.values["via_rsu"],
            "bonus_direct":  bonus_direct,
            "bonus_rsu":     bonus_rsu,
            "ucb_direct":    self.values["direct"]  + (bonus_direct if bonus_direct != float("inf") else 0),
            "ucb_rsu":       self.values["via_rsu"] + (bonus_rsu    if bonus_rsu    != float("inf") else 0),
            "n_direct":      self.counts["direct"],
            "n_rsu":         self.counts["via_rsu"],
        })


mab_agent = UCBAgent(MAB_ARMS, UCB_C)

# ── Centre de l'intersection ───────────────────────────────────────────────────
CENTER = (0.0, 0.0)

# ── État global ────────────────────────────────────────────────────────────────
events        = []
active_alerts = set()

stats = {
    "total_checks":         0,
    "collisions_detectees": 0,
    "alerts_direct_los":    0,
    "alerts_direct_nlos":   0,
    "alerts_rsu":           0,
    "echecs":               0,
    "ideal_direct_los":     0,
    "ideal_direct_nlos":    0,
    "ideal_rsu":            0,
    "ideal_echecs":         0,
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
    """
    D  = DANGER_HALF
    cx, cy = CENTER

    rx = vx - cx
    ry = py - cy

    if rx > D or abs(rx) > PRESCREEN_M:
        return False, None
    if ry > D:
        return False, None
    if v_speed < 0.1 or p_speed < 0.1:
        return False, None

    tv_enter = max(0.0, (-D - rx) / v_speed) if rx < -D else 0.0
    tv_exit  = (D - rx) / v_speed
    if tv_exit < 0:
        return False, None

    tp_enter = max(0.0, (-D - ry) / p_speed) if ry < -D else 0.0
    tp_exit  = (D - ry) / p_speed
    if tp_exit < 0:
        return False, None

    if tv_enter > tp_exit or tp_enter > tv_exit:
        return False, None

    ttc = max(tv_enter, tp_enter)
    return True, ttc


# ── Sélection du canal : oracle idéal + MAB UCB ───────────────────────────────

def try_alert(veh_pos, ped_pos):
    d_vp = dist(veh_pos, ped_pos)
    d_vr = dist(veh_pos, RSU_POS)
    d_rp = dist(RSU_POS, ped_pos)

    direct = channel.direct_link(d_vp)
    via    = channel.via_rsu(d_vr, d_rp, RSU_LOAD)

    ideal = direct if direct.delivery_probability >= via.delivery_probability else via

    mab_arm, ucb_bonus, decision = mab_agent.select_arm()
    selected = direct if mab_arm == "direct" else via
    reward   = 1.0 if selected.delivery_probability >= 0.5 else 0.0
    mab_agent.update(mab_arm, reward)

    return selected, ideal, direct, via, ucb_bonus, decision, reward


# ── Traitement d'un pas de simulation ─────────────────────────────────────────

def process_step(t: float):
    vehicles = traci.vehicle.getIDList()
    persons  = traci.person.getIDList()
    n_msg    = 0
    current_collisions = set()

    for vid in vehicles:
        vx, vy  = traci.vehicle.getPosition(vid)
        v_speed = traci.vehicle.getSpeed(vid)

        for pid in persons:
            px, py  = traci.person.getPosition(pid)
            p_speed = traci.person.getSpeed(pid)
            stats["total_checks"] += 1

            predicted, ttc = collision_predicted(vx, vy, v_speed, px, py, p_speed)
            if not predicted:
                continue

            current_collisions.add((vid, pid))

            if (vid, pid) in active_alerts:
                continue

            active_alerts.add((vid, pid))
            stats["collisions_detectees"] += 1
            n_msg += 1

            d = dist((vx, vy), (px, py))
            best, ideal, direct, via, ucb_bonus, decision, reward = try_alert((vx, vy), (px, py))
            success       = best.delivery_probability >= 0.5
            ideal_success = ideal.delivery_probability >= 0.5

            ev = {
                "t":            t,
                "veh":          vid,
                "ped":          pid,
                "dist":         d,
                "ttc":          ttc,
                "path":         best.path,
                "rssi":         best.rssi_dbm,
                "latency":      best.latency_ms,
                "pdlv":         best.delivery_probability,
                "nlos":         direct.nlos,
                "success":      success,
                "mab_reward":   reward,
                "ucb_bonus":    ucb_bonus,
                "mab_decision": decision,
                "ideal_path":   ideal.path,
                "ideal_rssi":   ideal.rssi_dbm,
                "ideal_latency":ideal.latency_ms,
                "ideal_pdlv":   ideal.delivery_probability,
                "ideal_success":ideal_success,
                "pdlv_direct":  direct.delivery_probability,
                "pdlv_rsu":     via.delivery_probability,
                "rssi_direct":  direct.rssi_dbm,
                "rssi_rsu":     via.rssi_dbm,
                # Snapshot UCB interne au moment de la décision
                "q_direct":     mab_agent.values["direct"],
                "q_rsu":        mab_agent.values["via_rsu"],
                "bonus_direct": mab_agent.history[-1]["bonus_direct"] if mab_agent.history else 0,
                "bonus_rsu":    mab_agent.history[-1]["bonus_rsu"]    if mab_agent.history else 0,
                "n_direct":     mab_agent.counts["direct"],
                "n_rsu":        mab_agent.counts["via_rsu"],
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
                  f"  UCB={best.path:8s}{nlos_tag}"
                  f"  RSSI={best.rssi_dbm:+.1f}dBm"
                  f"  lat={best.latency_ms:.1f}ms"
                  f"  Pdlv={best.delivery_probability:.1%}"
                  f"  bonus={ucb_bonus:.3f}"
                  f"  ideal={ideal.path}")

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


# ── Graphes V2X (RSSI, TTC, alertes, latences) ────────────────────────────────

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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Simulation V2X – ITS-G5 / DSRC 5.9 GHz  |  MAB UCB (c={UCB_C})\n"
        f"Detection collision TTC | Zone danger : +/-{DANGER_HALF} m | RSU : {RSU_POS}",
        fontsize=12, fontweight="bold"
    )

    legend_patches = [
        mpatches.Patch(color="green",     label="Direct LOS  (succes)"),
        mpatches.Patch(color="steelblue", label="Direct NLOS (succes)"),
        mpatches.Patch(color="orange",    label="Via RSU     (succes)"),
        mpatches.Patch(color="red",       label="Echec"),
    ]

    ax = axes[0, 0]
    ax.scatter(distances, rssi_vals, c=colors, alpha=0.7, s=60, zorder=3)
    ax.axhline(channel.SENSITIVITY_DBM, color="red", linestyle="--",
               linewidth=1.5, label=f"Sensibilite ({channel.SENSITIVITY_DBM} dBm)")
    ax.axvline(channel.LOS_RANGE_M, color="purple", linestyle=":",
               linewidth=1.2, label=f"Seuil LOS/NLOS ({channel.LOS_RANGE_M} m)")
    ax.set_xlabel("Distance vehicule-pieton (m)")
    ax.set_ylabel("RSSI (dBm)")
    ax.set_title("RSSI vs Distance (alerte unique par collision)")
    ax.legend(handles=legend_patches, fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(distances, ttc_vals, c=colors, alpha=0.7, s=60, zorder=3)
    ax.axvline(channel.LOS_RANGE_M, color="purple", linestyle=":",
               linewidth=1.2, label=f"Seuil LOS/NLOS ({channel.LOS_RANGE_M} m)")
    ax.set_xlabel("Distance vehicule-pieton (m)")
    ax.set_ylabel("TTC - Temps avant collision (s)")
    ax.set_title("TTC vs Distance au moment de la detection")
    ax.legend(handles=legend_patches, fontsize=7)
    ax.grid(True, alpha=0.3)

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
    ax.set_title("Alertes MAB UCB cumulees dans le temps")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

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
    path = os.path.join(OUT_DIR, "courbes_v2x_ucb.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Courbes V2X sauvegardees : {path}")


# ── Graphes spécifiques UCB ────────────────────────────────────────────────────

def plot_ucb_analysis():
    """
    4 graphes propres à l'algorithme UCB :
      1. Q-valeurs + bornes UCB au fil des évènements
      2. Bonus d'exploration (racine(ln t / N(a))) par bras — montre la décroissance
      3. Succès cumulés MAB UCB vs oracle idéal
      4. Regret cumulé + répartition des choix de bras
    """
    if not events:
        return

    steps = list(range(1, len(events) + 1))
    n     = len(events)

    q_direct  = [e["q_direct"]     for e in events]
    q_rsu     = [e["q_rsu"]        for e in events]
    b_direct  = [e["bonus_direct"] for e in events]
    b_rsu     = [e["bonus_rsu"]    for e in events]

    # Clip infinis pour l'affichage (premiers tirages)
    b_direct_plot = [min(b, 5.0) for b in b_direct]
    b_rsu_plot    = [min(b, 5.0) for b in b_rsu]

    ucb_direct = [q + b for q, b in zip(q_direct, b_direct_plot)]
    ucb_rsu    = [q + b for q, b in zip(q_rsu,    b_rsu_plot)]

    mab_success   = [1 if e["success"] else 0 for e in events]
    ideal_success = [1 if e["ideal_success"] else 0 for e in events]
    mab_cum       = np.cumsum(mab_success)
    ideal_cum     = np.cumsum(ideal_success)
    regret        = ideal_cum - mab_cum
    mab_rate      = mab_cum   / np.arange(1, n + 1)
    ideal_rate    = ideal_cum / np.arange(1, n + 1)

    direct_choices = np.cumsum([1 if e["path"] == "direct" else 0 for e in events])
    rsu_choices    = np.cumsum([1 if e["path"] == "via_rsu" else 0 for e in events])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Analyse MAB UCB1 (c={UCB_C})  —  Convergence et regret",
        fontsize=12, fontweight="bold"
    )

    # ── Panneau 1 : Q-valeurs + bornes UCB ─────────────────────────────────── #
    ax = axes[0, 0]
    ax.plot(steps, q_direct,  color="green",  linewidth=2,   label="Q(direct)")
    ax.plot(steps, q_rsu,     color="orange", linewidth=2,   label="Q(via_rsu)")
    ax.plot(steps, ucb_direct, color="green",  linewidth=1,
            linestyle="--", alpha=0.6, label="UCB(direct) = Q + bonus")
    ax.plot(steps, ucb_rsu,    color="orange", linewidth=1,
            linestyle="--", alpha=0.6, label="UCB(via_rsu) = Q + bonus")
    ax.fill_between(steps, q_direct, ucb_direct, color="green",  alpha=0.12)
    ax.fill_between(steps, q_rsu,    ucb_rsu,    color="orange", alpha=0.12)
    ax.set_xlabel("Evenement collision")
    ax.set_ylabel("Score UCB")
    ax.set_title("Q-valeurs et bornes UCB (zone = bonus d'exploration)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panneau 2 : Bonus d'exploration par bras ────────────────────────────── #
    ax = axes[0, 1]
    ax.plot(steps, b_direct_plot, color="green",  linewidth=2, label="Bonus direct  = c·√(ln t / N_direct)")
    ax.plot(steps, b_rsu_plot,    color="orange", linewidth=2, label="Bonus via_rsu = c·√(ln t / N_rsu)")
    ax.set_xlabel("Evenement collision")
    ax.set_ylabel("Bonus d'exploration (clip a 5)")
    ax.set_title("Decroissance du bonus UCB par bras\n(converge vers 0 → exploitation pure)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panneau 3 : Taux de succès MAB UCB vs oracle idéal ─────────────────── #
    ax = axes[1, 0]
    ax.plot(steps, mab_rate,   linewidth=2, color="royalblue", label=f"MAB UCB (c={UCB_C})")
    ax.plot(steps, ideal_rate, linewidth=2, color="black",
            linestyle="--", label="Oracle ideal")
    ax.set_xlabel("Evenement collision")
    ax.set_ylabel("Taux de succes cumule")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Taux de succes : UCB vs Oracle ideal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panneau 4 : Regret cumulé + répartition des choix ──────────────────── #
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.step(steps, regret, where="post", color="crimson", linewidth=2,
            label="Regret cumule (ideal - UCB)")
    ax2.step(steps, direct_choices, where="post", color="green",  linewidth=1.5,
             linestyle=":", label="Choix direct")
    ax2.step(steps, rsu_choices,    where="post", color="orange", linewidth=1.5,
             linestyle=":", label="Choix via_rsu")
    ax.set_xlabel("Evenement collision")
    ax.set_ylabel("Regret cumule", color="crimson")
    ax2.set_ylabel("Choix cumules par bras")
    ax.set_title("Regret cumule et exploration des bras")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "comparaison_ucb_ideal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Analyse UCB sauvegardee  : {path}")


# ── Rapport texte ──────────────────────────────────────────────────────────────

def write_report():
    total_ok  = stats["alerts_direct_los"] + stats["alerts_direct_nlos"] + stats["alerts_rsu"]
    total     = total_ok + stats["echecs"]
    taux      = total_ok / total if total > 0 else 0.0
    ideal_ok  = stats["ideal_direct_los"] + stats["ideal_direct_nlos"] + stats["ideal_rsu"]
    ideal_total = ideal_ok + stats["ideal_echecs"]
    ideal_taux  = ideal_ok / ideal_total if ideal_total > 0 else 0.0
    regret      = ideal_ok - total_ok

    lat_direct = [e["latency"] for e in events if e["success"] and e["path"] == "direct"]
    lat_rsu    = [e["latency"] for e in events if e["success"] and e["path"] == "via_rsu"]
    rssi_ok    = [e["rssi"]    for e in events if e["success"]]
    rssi_ko    = [e["rssi"]    for e in events if not e["success"]]
    ttc_list   = [e["ttc"]     for e in events]

    # Analyse convergence UCB
    n_direct_final = mab_agent.counts["direct"]
    n_rsu_final    = mab_agent.counts["via_rsu"]
    total_pulls    = mab_agent.total_pulls
    bonus_direct_final = (
        UCB_C * math.sqrt(math.log(total_pulls) / n_direct_final)
        if n_direct_final > 0 and total_pulls > 1 else float("inf")
    )
    bonus_rsu_final = (
        UCB_C * math.sqrt(math.log(total_pulls) / n_rsu_final)
        if n_rsu_final > 0 and total_pulls > 1 else float("inf")
    )

    path = os.path.join(OUT_DIR, "resultatsUCB.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  RAPPORT DE SIMULATION V2X — MAB UCB1\n")
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
        f.write(f"  Algorithme choix       : MAB UCB1\n")
        f.write(f"  UCB_C                  : {UCB_C}\n")
        f.write(f"  Formule selection      : argmax Q(a) + {UCB_C}*sqrt(ln(t)/N(a))\n\n")

        f.write("RESULTATS GLOBAUX\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Paires V-P analysees   : {stats['total_checks']}\n")
        f.write(f"  Collisions detectees   : {stats['collisions_detectees']}\n")
        f.write(f"  Alertes emises         : {total}\n")
        f.write(f"  Taux succes MAB UCB    : {taux:.1%}\n")
        f.write(f"  Taux succes ideal      : {ideal_taux:.1%}\n")
        f.write(f"  Regret cumule          : {regret} succes\n")
        f.write(f"  Taux d'echec           : {1-taux:.1%}\n\n")

        f.write("DETAIL PAR CHEMIN - MAB UCB\n")
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

        f.write("ETAT FINAL DU MAB UCB\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Q(direct)              : {mab_agent.values['direct']:.4f} | n={n_direct_final}\n")
        f.write(f"  Q(via_rsu)             : {mab_agent.values['via_rsu']:.4f} | n={n_rsu_final}\n")
        f.write(f"  Bonus final direct     : {bonus_direct_final:.4f}\n")
        f.write(f"  Bonus final via_rsu    : {bonus_rsu_final:.4f}\n")
        f.write(f"  UCB final direct       : {mab_agent.values['direct'] + bonus_direct_final:.4f}\n")
        f.write(f"  UCB final via_rsu      : {mab_agent.values['via_rsu'] + bonus_rsu_final:.4f}\n")
        f.write(f"  Total tirages          : {total_pulls}\n\n")

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

        f.write("INTERPRETATION UCB\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"  Algorithme : MAB UCB1 avec c={UCB_C}\n"
            f"  Formule    : selectionner argmax Q(a) + c*sqrt(ln(t)/N(a))\n\n"
            f"  Contrairement a eps-greedy, UCB explore de facon DETERMINISTE :\n"
            f"  plus un bras est sous-explore, plus son bonus est grand.\n"
            f"  Le bonus decroit automatiquement -> plus besoin de fixer epsilon.\n\n"
        )
        if n_direct_final > 0 and n_rsu_final > 0:
            f.write(f"  Repartition finale : {n_direct_final} tirages direct"
                    f" vs {n_rsu_final} tirages RSU\n")
            dom = "direct" if mab_agent.values["direct"] >= mab_agent.values["via_rsu"] else "via_rsu"
            f.write(f"  Bras dominant (Q max) : {dom}\n\n")
        if regret == 0:
            f.write("  Regret nul : UCB a fait les memes choix que l'oracle.\n")
        elif regret <= 2:
            f.write(f"  Regret tres faible ({regret}) : UCB quasi-optimal.\n")
        else:
            f.write(
                f"  Regret = {regret} succes manques par rapport a l'oracle.\n"
                f"  -> Augmenter UCB_C pour plus d'exploration si regret eleve.\n"
                f"  -> Diminuer UCB_C si l'agent sur-explore (regret stagne).\n"
            )

    print(f"  Rapport sauvegarde       : {path}")


# ── Boucle principale ──────────────────────────────────────────────────────────

def run():
    sumo_cmd = [SUMO_BIN, "-c", SUMOCFG, "--step-length", str(STEP_LEN)]

    print("=" * 65)
    print("  Simulation V2X - Detection collision par TTC")
    print("  Algorithme : MAB UCB1")
    print("=" * 65)
    print(f"  Config       : {SUMOCFG}")
    print(f"  RSU          : {RSU_POS}")
    print(f"  Zone danger  : carre +/-{DANGER_HALF} m")
    print(f"  Seuil LOS    : {channel.LOS_RANGE_M} m")
    print(f"  Choix chemin : MAB UCB1 (c={UCB_C})")
    print(f"  Score UCB    : Q(a) + {UCB_C}*sqrt(ln(t)/N(a))")
    print()

    traci.start(sumo_cmd)

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
    ideal_taux  = ideal_ok / ideal_total if ideal_total > 0 else 0.0

    print()
    print("=" * 65)
    print("  RAPPORT UCB")
    print("=" * 65)
    print(f"  Paires V-P analysees   : {stats['total_checks']}")
    print(f"  Collisions detectees   : {stats['collisions_detectees']}")
    print(f"  Alertes emises         : {total}")
    print(f"  +-- Direct LOS         : {stats['alerts_direct_los']}")
    print(f"  +-- Direct NLOS        : {stats['alerts_direct_nlos']}")
    print(f"  +-- Via RSU            : {stats['alerts_rsu']}")
    print(f"  +-- Echecs             : {stats['echecs']}")
    print(f"  Taux de succes UCB     : {taux:.1%}")
    print(f"  Taux de succes ideal   : {ideal_taux:.1%}")
    print(f"  Regret cumule          : {ideal_ok - total_ok} succes")
    print(f"  Q(direct) ={mab_agent.values['direct']:.4f} n={mab_agent.counts['direct']}")
    print(f"  Q(via_rsu)={mab_agent.values['via_rsu']:.4f} n={mab_agent.counts['via_rsu']}")
    print()

    write_report()
    plot_results()
    plot_ucb_analysis()
    print()


if __name__ == "__main__":
    run()
