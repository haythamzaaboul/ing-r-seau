"""
Modèle de canal radio V2X à 5.9 GHz (ITS-G5 / DSRC).
Modèle log-distance + ombrage log-normal + pénalité NLOS (IEEE 802.11p).
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class ChannelResult:
    rssi_dbm: float
    latency_ms: float
    delivery_probability: float
    path: str          # 'direct' | 'via_rsu'
    nlos: bool         # True si condition Non-Line-of-Sight


class ChannelModel:
    """
    Paramètres calés sur mesures V2X urbaines (5.9 GHz, ETSI TR 103 257).

    LOS  : n=2.5, σ=5 dB
    NLOS : n=3.0, σ=8 dB + pénalité d'atténuation de 6 dB
           (obstacle partiel : bâtiment de coin, signalisation, bus)
    """

    FREQ_HZ          = 5.9e9
    SPEED_LIGHT      = 3e8
    TX_POWER_DBM     = 20.0
    SENSITIVITY_DBM  = -85.0

    # LOS (vue dégagée, obstacles légers)
    N_DIRECT_LOS     = 2.5
    SIGMA_DIRECT_LOS = 4.0   # légèrement réduit

    # NLOS (obstacle partiel : bâtiment de coin, véhicule large)
    # Recalibré pour RSSI_mean ≈ -80 dBm à 80 m → Pdlv ~88 %
    N_DIRECT_NLOS     = 2.6   # était 3.0 — plus réaliste pour urbain partiel
    SIGMA_DIRECT_NLOS = 5.0   # était 8.0
    NLOS_PENALTY_DB   = 3.0   # était 10.0 — atténuation modérée

    # Infrastructure — légèrement dégradée pour équilibrer (RSU pas toujours parfait)
    N_INFRA     = 2.2         # était 2.0
    SIGMA_INFRA = 4.5         # était 3.5

    D0 = 1.0

    # dist < 50 m  → LOS  (zone piéton dégagée à l'approche du passage)
    # 50 m–200 m  → NLOS (obstacle croissant à mesure qu'on s'éloigne)
    LOS_RANGE_M = 50.0

    # ── Primitives ──────────────────────────────────────────────────────────── #

    def _pl_d0(self) -> float:
        return 20 * np.log10(4 * np.pi * self.D0 * self.FREQ_HZ / self.SPEED_LIGHT)

    def _path_loss(self, d: float, n: float, sigma: float,
                   nlos_penalty: float = 0.0) -> float:
        d = max(d, self.D0)
        pl = self._pl_d0() + 10 * n * np.log10(d / self.D0)
        pl += np.random.normal(0.0, sigma)
        pl += nlos_penalty
        return pl

    def rssi_at(self, d: float, n: float, sigma: float,
                nlos_penalty: float = 0.0) -> float:
        return self.TX_POWER_DBM - self._path_loss(d, n, sigma, nlos_penalty)

    def delivery_prob(self, rssi_dbm: float) -> float:
        """Courbe en S centrée sur la sensibilité (-85 dBm)."""
        return 1.0 / (1.0 + np.exp(-0.4 * (rssi_dbm - self.SENSITIVITY_DBM)))

    # ── Scénarios ───────────────────────────────────────────────────────────── #

    def direct_link(self, dist_v2p: float) -> ChannelResult:
        """
        Communication directe Véhicule->Piéton (V2P / PC5).
        Active le mode NLOS automatiquement si dist > LOS_RANGE_M.
        """
        nlos = dist_v2p > self.LOS_RANGE_M

        if nlos:
            n, sigma, penalty = self.N_DIRECT_NLOS, self.SIGMA_DIRECT_NLOS, self.NLOS_PENALTY_DB
        else:
            n, sigma, penalty = self.N_DIRECT_LOS, self.SIGMA_DIRECT_LOS, 0.0

        rssi    = self.rssi_at(dist_v2p, n, sigma, penalty)
        latency = 5.0 + abs(np.random.normal(0.0, 1.5))

        return ChannelResult(
            rssi_dbm=rssi,
            latency_ms=max(1.0, latency),
            delivery_probability=self.delivery_prob(rssi),
            path="direct",
            nlos=nlos,
        )

    def via_rsu(self, dist_v2rsu: float, dist_rsu2p: float,
                rsu_load: float) -> ChannelResult:
        """
        Communication V->RSU->Piéton (V2I2P).
        RSU en hauteur = toujours en LOS.
        """
        rssi_up = self.rssi_at(dist_v2rsu, self.N_INFRA, self.SIGMA_INFRA)
        rssi_dn = self.rssi_at(dist_rsu2p, self.N_INFRA, self.SIGMA_INFRA)
        rssi    = min(rssi_up, rssi_dn)

        rsu_proc = 10.0 + rsu_load * 60.0
        latency  = rsu_proc + abs(np.random.normal(0.0, 2.0))

        p_up      = self.delivery_prob(rssi_up)
        p_dn      = self.delivery_prob(rssi_dn)
        load_drop = max(0.0, rsu_load - 0.7) * 2.5
        delivery  = p_up * p_dn * (1.0 - load_drop)

        return ChannelResult(
            rssi_dbm=rssi,
            latency_ms=max(1.0, latency),
            delivery_probability=float(np.clip(delivery, 0.0, 1.0)),
            path="via_rsu",
            nlos=False,
        )
