
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

# Modèle pour la rédaction et les tâches créatives
WRITER_LLM = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434", temperature=0.7)
# Modèle spécialisé pour le code et les tâches techniques/scientifiques
TECHNICAL_LLM = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434", temperature=0.2)

# --- INITIALISATION ---
console = Console()

# --- DÉFINITION DE L'ÉTAT DU GRAPHE ---
class ChapterState(TypedDict):
    nom_fichier: str
    contenu_brut: str
    plan_chapitre: str
    synthese_validee: str
    texte_redige: str
    texte_avec_schemas: str
    texte_final: str

# --- DÉFINITION DES NOEUDS DU GRAPHE (AGENTS) ---

def architecte(state: ChapterState):
    console.print(f"[yellow]1. Agent Architecte:[/yellow] Planification pour {state['nom_fichier']}...")
    prompt = f"""Tu es un expert équestre. Analyse ce mémo et propose un plan détaillé pour un chapitre de livre de 2000-3000 mots.

MÉMO:
{state['contenu_brut']}

Crée un plan structuré avec des sections et sous-sections claires."""
    plan = WRITER_LLM.invoke(prompt)
    return {"plan_chapitre": plan}

def validateur(state: ChapterState):
    console.print(f"[yellow]2. Agent Validateur:[/yellow] Validation pour {state['nom_fichier']}...")
    prompt = f"""Tu es un scientifique et vétérinaire rigoureux. Analyse le plan du chapitre et le contenu source. 
    Produis une synthèse factuelle et fiable qui servira de base à la rédaction.

PLAN DU CHAPITRE:
{state['plan_chapitre']}

CONTENU SOURCE:
{state['contenu_brut']}

SYNTHÈSE VALIDÉE:"""
    synthese = TECHNICAL_LLM.invoke(prompt)
    return {"synthese_validee": synthese}

def redacteur(state: ChapterState):
    console.print(f"[yellow]3. Agent Rédacteur:[/yellow] Rédaction de {state['nom_fichier']}...")
    prompt = f"""Tu es un rédacteur expert équestre. En te basant sur la synthèse validée, rédige un chapitre complet et détaillé de 2000-3000 mots.

SYNTHÈSE VALIDÉE:
{state['synthese_validee']}

INSTRUCTIONS:
- Crée un chapitre complet et autonome
- Structure avec des sous-sections claires (## et ###)
- Ajoute des exemples concrets et des conseils pratiques
- Utilise un style pédagogique et professionnel
- Insère des placeholders [SCHEMA: description] là où un schéma serait utile

FORMAT:
# Chapitre: [Titre du chapitre]

[Introduction du chapitre]

## [Première sous-section]
[Contenu détaillé...]

## [Deuxième sous-section]
[Contenu détaillé...]

[Continuer avec toutes les sous-sections nécessaires...]

## Résumé du chapitre
[Résumé des points clés]"""
    texte = WRITER_LLM.invoke(prompt)
    return {"texte_redige": texte}

def illustrateur(state: ChapterState):
    console.print(f"[yellow]4. Agent Illustrateur:[/yellow] Création des schémas pour {state['nom_fichier']}...")
    texte = state['texte_redige']
    placeholders = re.findall(r'\[SCHEMA: (.*?)\]', texte)
    if not placeholders:
        return {"texte_avec_schemas": texte}
    
    for i, desc in enumerate(placeholders):
        nom_image = state['nom_fichier'].replace('.md', '').replace('/', '_').replace('\\', '_') + f"_{i+1}"
        prompt_schema = f"""Crée un schéma simple en format DOT pour Graphviz qui illustre: '{desc}'.

Exemple de format:
digraph G {{
    A [label="Élément A"];
    B [label="Élément B"];
    A -> B [label="Relation"];
}}

Ne fournis que le code DOT, sans explications."""
        
        try:
            code_dot = TECHNICAL_LLM.invoke(prompt_schema).strip()
            # Nettoyer le code DOT
            if code_dot.startswith('```'):
                code_dot = code_dot.split('```')[1]
            if code_dot.startswith('dot'):
                code_dot = code_dot.split('\n', 1)[1]
            
            dot_path = f"images/{nom_image}.dot"
            png_path = f"images/{nom_image}.png"
            
            # Créer le dossier images s'il n'existe pas
            os.makedirs("images", exist_ok=True)
            
            # Sauvegarder le fichier DOT
            with open(dot_path, "w", encoding="utf-8") as f:
                f.write(code_dot)
            
            # Essayer de générer le PNG (si Graphviz est installé)
            try:
                result = os.system(f"dot -Tpng {dot_path} -o {png_path}")
                if result == 0:
                    texte = texte.replace(f"[SCHEMA: {desc}]", f"\n![{desc}]({png_path})\n", 1)
                    console.print(f"  [green]✓ Schéma créé: {png_path}[/green]")
                else:
                    # Si Graphviz échoue, garder le fichier DOT et ajouter un lien
                    texte = texte.replace(f"[SCHEMA: {desc}]", f"\n[Schéma DOT: {dot_path}]({dot_path})\n", 1)
                    console.print(f"  [yellow]⚠ Schéma DOT créé: {dot_path}[/yellow]")
            except Exception as e:
                console.print(f"  [red]✗ Erreur Graphviz: {e}[/red]")
                # Garder le fichier DOT même si PNG échoue
                texte = texte.replace(f"[SCHEMA: {desc}]", f"\n[Schéma DOT: {dot_path}]({dot_path})\n", 1)
                
        except Exception as e:
            console.print(f"  [red]✗ Erreur création schéma: {e}[/red]")
            # En cas d'erreur, remplacer par un texte simple
            texte = texte.replace(f"[SCHEMA: {desc}]", f"\n*[Schéma: {desc} - Erreur de génération]*\n", 1)
    
    return {"texte_avec_schemas": texte}

def relecteur(state: ChapterState):
    console.print(f"[yellow]5. Agent Relecteur:[/yellow] Relecture de {state['nom_fichier']}...")
    prompt = f"""Tu es un relecteur professionnel. Relis et polis ce chapitre final. 
    Corrige les fautes, améliore le style et la clarté. Garde le même nombre de mots.

TEXTE:
{state['texte_avec_schemas']}

Corrige et améliore ce texte en gardant la même structure et longueur."""
    texte_final = WRITER_LLM.invoke(prompt)
    return {"texte_final": texte_final}

# --- CONSTRUCTION DU GRAPHE ---
workflow = StateGraph(ChapterState)
workflow.add_node("architecte", architecte)
workflow.add_node("validateur", validateur)
workflow.add_node("redacteur", redacteur)
workflow.add_node("illustrateur", illustrateur)
workflow.add_node("relecteur", relecteur)

workflow.set_entry_point("architecte")
workflow.add_edge("architecte", "validateur")
workflow.add_edge("validateur", "redacteur")
workflow.add_edge("redacteur", "illustrateur")
workflow.add_edge("illustrateur", "relecteur")
workflow.add_edge("relecteur", END)

app = workflow.compile()
console.print("[bold green]Graphe de l'Excellence Absolue (5 agents) prêt ![/bold green]")

# --- FONCTION PRINCIPALE ---
def lire_fichiers_md():
    """Lit tous les fichiers .md du projet et retourne leur contenu"""
    fichiers_md = {}
    
    for fichier in glob.glob("**/*.md", recursive=True):
        if fichier != "README.md" and not fichier.startswith(("venv", "images", "livre_")):
            try:
                with open(fichier, 'r', encoding='utf-8') as f:
                    contenu = f.read()
                    fichiers_md[fichier] = contenu
                    console.print(f"[cyan]✓ Lu : {fichier}[/cyan]")
            except Exception as e:
                console.print(f"[red]✗ Erreur lecture {fichier} : {e}[/red]")
    
    console.print(f"[bold green]Total : {len(fichiers_md)} fichiers .md chargés[/bold green]")
    return fichiers_md

def main():
    console.print("\n[bold cyan]=== CRÉATION DU LIVRE DE L'EXCELLENCE ABSOLUE ===[/bold cyan]")
    
    # Créer le dossier images
    os.makedirs("images", exist_ok=True)
    
    # Lire les fichiers
    fichiers_md = lire_fichiers_md()
    if not fichiers_md:
        console.print("[bold red]Aucun fichier .md trouvé. Arrêt du script.[/bold red]")
        return
    
    livre_final = []
    
    # Traiter chaque chapitre avec le graphe LangGraph
    for i, (nom_fichier, contenu) in enumerate(fichiers_md.items(), 1):
        try:
            console.print(f"\n[bold yellow]>>> Traitement du chapitre {i}/{len(fichiers_md)}: {nom_fichier} <<<[/bold yellow]")
            
            # Exécuter le workflow complet pour ce chapitre
            input_state = {"nom_fichier": nom_fichier, "contenu_brut": contenu}
            final_state = app.invoke(input_state)
            
            livre_final.append(f"# Chapitre {i}: {nom_fichier.replace('.md', '').replace('_', ' ').title()}\n")
            livre_final.append(final_state.get('texte_final', 'Erreur de relecture.'))
            
            console.print(f"[green]✓ Chapitre {i} terminé avec succès[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Erreur chapitre {i}: {e}[/red]")
            # En cas d'erreur, ajouter le contenu original
            livre_final.append(f"# Chapitre {i}: {nom_fichier.replace('.md', '').replace('_', ' ').title()}\n")
            livre_final.append(contenu)
        
        # Pause entre les chapitres
        time.sleep(2)
    
    # Assemblage final
    console.print("\n[bold yellow]Assemblage final du livre...[/bold yellow]")
    try:
        with open('livre_excellence_absolue.md', 'w', encoding='utf-8') as f:
            f.write("\n\n".join(livre_final))
        console.print("\n[bold green]LIVRE DE L'EXCELLENCE ABSOLUE CRÉÉ ![/bold green]")
        console.print("[green]Sauvegardé dans : [bold]livre_excellence_absolue.md[/bold][/green]")
        console.print("[green]Chaque chapitre a été enrichi et structuré avec des schémas[/green]")
    except Exception as e:
        console.print(f"[bold red]Erreur lors de la sauvegarde : {e}[/bold red]")

if __name__ == "__main__":
    main()
