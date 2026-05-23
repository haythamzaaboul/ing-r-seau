"""
Génère map.net.xml à partir des fichiers nodes + edges.
Lancer UNE SEULE FOIS avant la simulation :
    python setup_network.py
"""
import os
import subprocess
import sys

NETWORK_DIR = os.path.join(os.path.dirname(__file__), "network")


def generate():
    nodes = os.path.join(NETWORK_DIR, "nodes.nod.xml")
    edges = os.path.join(NETWORK_DIR, "edges.edg.xml")
    output = os.path.join(NETWORK_DIR, "map.net.xml")

    cmd = [
        "netconvert",
        "--node-files", nodes,
        "--edge-files", edges,
        "--output-file", output,
        "--no-warnings", "true",
        "--crossings.guess", "true",
        "--sidewalks.guess", "false",
    ]

    print("Génération du réseau SUMO...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Réseau généré : {output}")
    except FileNotFoundError:
        print("ERREUR: netconvert introuvable.")
        print("Installez SUMO et ajoutez $SUMO_HOME/bin au PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERREUR netconvert : {e}")
        sys.exit(1)


if __name__ == "__main__":
    generate()
