"""Exécution du notebook depuis un noyau vierge, avec sauvegarde des sorties."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", choices=("lsa", "sentencebert"), default="lsa")
    args = parser.parse_args()
    dossier = Path(__file__).resolve().parent
    os.chdir(dossier)
    os.environ["BANKING77_EMBEDDINGS"] = args.embeddings
    os.environ.setdefault("MPLCONFIGDIR", str(dossier / ".cache" / "matplotlib"))
    os.environ.setdefault("IPYTHONDIR", str(dossier / ".cache" / "ipython"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output

    chemin = dossier / "Projet_Classification_Intentions_Banking77.ipynb"
    notebook = json.loads(chemin.read_text(encoding="utf-8"))
    shell = TerminalInteractiveShell.instance()
    shell.enable_matplotlib("inline")
    shell.display_formatter.active_types = [
        "text/plain", "text/html", "text/markdown", "image/png", "image/svg+xml", "application/json"
    ]
    compteur = 0
    echec = False
    for numero, cellule in enumerate(notebook["cells"]):
        cellule.setdefault("id", f"banking77-{numero:02d}")
        if cellule["cell_type"] == "code":
            cellule["outputs"] = []
            cellule["execution_count"] = None
    for numero, cellule in enumerate(notebook["cells"]):
        if cellule["cell_type"] != "code":
            continue
        compteur += 1
        print(f"Cellule {numero + 1}/{len(notebook['cells'])}…", flush=True)
        with capture_output() as capture:
            resultat = shell.run_cell("".join(cellule["source"]), store_history=False)
        sorties = []
        for nom, contenu in (("stdout", capture.stdout), ("stderr", capture.stderr)):
            if contenu:
                sorties.append({"output_type": "stream", "name": nom, "text": contenu.splitlines(keepends=True)})
        for sortie in capture.outputs:
            if not sortie.update:
                sorties.append({"output_type": "display_data", "data": sortie.data, "metadata": sortie.metadata})
        erreur = resultat.error_before_exec or resultat.error_in_exec
        if erreur:
            sorties.append({"output_type": "error", "ename": type(erreur).__name__, "evalue": str(erreur), "traceback": []})
            echec = True
        cellule["execution_count"] = compteur
        cellule["outputs"] = sorties
        if capture.stdout:
            print(capture.stdout[:1800], flush=True)
        if erreur:
            print(f"Échec : {type(erreur).__name__}: {erreur}", file=sys.stderr)
            break
    notebook["metadata"]["language_info"]["version"] = sys.version.split()[0]
    chemin.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if echec:
        raise SystemExit(1)
    print(f"Notebook exécuté intégralement : {compteur} cellules de code.", flush=True)


if __name__ == "__main__":
    main()
