import os
import glob
import time
import re
from typing import TypedDict

from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from rich.console import Console

# --- CONFIGURATION ---
os.environ["SERPER_API_KEY"] = "2b39648999be6031ae7c672afd9b248359fee448"

# Modèle pour la rédaction
WRITER_LLM = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434", temperature=0.7)
# Modèle spécialisé pour le code et les tâches techniques
CODER_LLM = OllamaLLM(model="qwen2.5-coder:7b", base_url="http://localhost:11434", temperature=0.2)

# --- INITIALISATION ---
console = Console()

# --- DÉFINITION DE L'ÉTAT DU GRAPHE ---
class ChapterState(TypedDict):
    nom_fichier: str
    contenu_brut: str
    plan_chapitre: str
    texte_redige: str
    texte_avec_schemas: str
    texte_final: str

# --- DÉFINITION DES "AGENTS" (NOEUDS DU GRAPHE) ---

def planifier_chapitre(state: ChapterState):
    console.print(f"[yellow]Agent Architecte:[/yellow] Planification pour {state['nom_fichier']}...")
    prompt = f"""Tu es un architecte de contenu expert. Analyse le contenu brut suivant et propose un plan détaillé pour un chapitre de livre. Identifie 1 à 2 endroits clés où un schéma Graphviz serait utile pour la compréhension et mentionne-le.

CONTENU BRUT :
{state['contenu_brut']} """
    plan = WRITER_LLM.invoke(prompt)
    return {"plan_chapitre": plan}

def rediger_chapitre(state: ChapterState):
    console.print(f"[yellow]Agent Rédacteur:[/yellow] Rédaction de {state['nom_fichier']}...")
    prompt = f"""Tu es un rédacteur professionnel. En te basant sur le plan détaillé, rédige un chapitre complet. IMPORTANT : Là où le plan suggère un schéma, insère un placeholder clair dans le texte sous la forme : [SCHEMA: description très claire et concise du schéma à créer].

PLAN DÉTAILLÉ :
{state['plan_chapitre']}

CONTENU BRUT DE RÉFÉRENCE :
{state['contenu_brut']} """
    texte = WRITER_LLM.invoke(prompt)
    return {"texte_redige": texte}

def generer_et_inserer_schemas(state: ChapterState):
    console.print(f"[yellow]Agent Illustrateur:[/yellow] Recherche de schémas pour {state['nom_fichier']}...")
    texte = state['texte_redige']
    placeholders = re.findall(r'\[SCHEMA: (.*?)\]', texte)

    if not placeholders:
        console.print("[green]Aucun schéma à générer.[/green]")
        return {"texte_avec_schemas": texte}

    for i, description in enumerate(placeholders):
        console.print(f"  [cyan]Génération du schéma : {description}[/cyan]")
        nom_image = f"{state['nom_fichier'].replace('.md', '')}_{i+1}"
        
        prompt_schema = f"""Tu es un expert du langage DOT de Graphviz. Écris un code DOT complet et fonctionnel pour la description suivante : '{description}'. Ne fournis que le code, sans explication.

Format attendu :
digraph G {{
  // tes instructions dot
}}
"""
        
        try:
            code_dot = CODER_LLM.invoke(prompt_schema)
            
            # Nettoyage du code DOT généré
            code_dot = re.sub(r'^```(dot)?', '', code_dot.strip(), flags=re.MULTILINE)
            code_dot = re.sub(r'```$', '', code_dot.strip(), flags=re.MULTILINE)

            dot_path = f"images/{nom_image}.dot"
            png_path = f"images/{nom_image}.png"
            with open(dot_path, "w", encoding="utf-8") as f:
                f.write(code_dot)
            
            os.system(f"dot -Tpng {dot_path} -o {png_path}")
            
            lien_markdown = f"\n![{description}]({png_path})\n"
            texte = texte.replace(f"[SCHEMA: {description}]", lien_markdown, 1)
            console.print(f"  [green]✓ Schéma '{png_path}' créé et inséré.[/green]")
        except Exception as e:
            console.print(f"  [red]✗ Erreur création schéma : {e}[/red]")
            texte = texte.replace(f"[SCHEMA: {description}]", "\n[Erreur de génération du schéma]\n", 1)
            
    return {"texte_avec_schemas": texte}

def revoir_chapitre(state: ChapterState):
    console.print(f"[yellow]Agent Relecteur:[/yellow] Relecture de {state['nom_fichier']}...")
    prompt = f"""Tu es un éditeur exigeant. Relis et corrige le chapitre suivant. Améliore la fluidité, la grammaire, le style et assure-toi que les références aux schémas sont naturelles. Ne réécris pas tout, polis le texte existant.

TEXTE À RELIRE :
{state['texte_avec_schemas']} """
    texte_final = WRITER_LLM.invoke(prompt)
    return {"texte_final": texte_final}

# --- CONSTRUCTION DU GRAPHE ---
workflow = StateGraph(ChapterState)
workflow.add_node("architecte", planifier_chapitre)
workflow.add_node("redacteur", rediger_chapitre)
workflow.add_node("illustrateur", generer_et_inserer_schemas)
workflow.add_node("relecteur", revoir_chapitre)

workflow.set_entry_point("architecte")
workflow.add_edge("architecte", "redacteur")
workflow.add_edge("redacteur", "illustrateur")
workflow.add_edge("illustrateur", "relecteur")
workflow.add_edge("relecteur", END)

app = workflow.compile()
console.print("[bold green]Graphe de l'excellence (4 agents) compilé et prêt ![/bold green]")

# --- FONCTION PRINCIPALE ---
def lire_fichiers_md():
    # ... (le reste du script est identique)
    pass # Placeholder, le reste du code est le même

def main():
    console.print("\n[bold cyan]=== CRÉATION DU LIVRE DE L'EXCELLENCE ===[/bold cyan]")
    os.makedirs("images", exist_ok=True)
    fichiers_md = lire_fichiers_md()
    livre_final = []

    for i, (nom_fichier, contenu) in enumerate(fichiers_md.items(), 1):
        console.print(f"\n[bold yellow]>>> Traitement du chapitre {i}/{len(fichiers_md)}: {nom_fichier} <<<[/bold yellow]")
        input_state = {"nom_fichier": nom_fichier, "contenu_brut": contenu}
        final_state = app.invoke(input_state)
        livre_final.append(f"# Chapitre {i}: {nom_fichier.replace('.md', '').replace('_', ' ').title()}\n")
        livre_final.append(final_state.get('texte_final', 'Erreur de relecture.'))
        time.sleep(2)

    console.print("\n[bold yellow]Assemblage final du livre...[/bold yellow]")
    with open('livre_excellence.md', 'w', encoding='utf-8') as f:
        f.write("\n\n".join(livre_final))
    console.print("\n[bold green]LIVRE DE L'EXCELLENCE CRÉÉ ![/bold green]")
    console.print("[green]Sauvegardé dans : [bold]livre_excellence.md[/bold][/green]")

# Le code de lire_fichiers_md() est nécessaire ici
def lire_fichiers_md():
    fichiers_md = {}
    for fichier in glob.glob("**/*.md", recursive=True):
        if fichier != "README.md" and not fichier.startswith(("venv", "images", "livre_")):
            try:
                with open(fichier, 'r', encoding='utf-8') as f:
                    fichiers_md[fichier] = f.read()
            except Exception as e:
                console.print(f"[red]✗ Erreur lecture {fichier} : {e}[/red]")
    return fichiers_md

if __name__ == "__main__":
    main()