import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_community.llms import Ollama
from rich.console import Console

# --- CONFIGURATION DE LA CLÉ API ---
# La clé a été mise manuellement par l'utilisateur
os.environ["SERPER_API_KEY"] = "2b39648999be6031ae7c672afd9b248359fee448" 

# --- CONFIGURATION OLLAMA ---
# C'est la configuration la plus explicite pour se connecter à Ollama localement.
OLLAMA_LLM = Ollama(model="ollama3.2:latest", base_url="http://localhost:11434")

# --- Initialisation de la console Rich ---
console = Console()

console.print("[bold cyan]Initialisation du processus de création du livre...[/bold cyan]")

# --- 1. Initialisation des Outils ---
console.print("[cyan]Initialisation des outils (SerperDevTool)...[/cyan]")
search_tool = SerperDevTool()

# --- 2. Définition des Agents ---
console.print("[bold cyan]Création de l'équipe d'agents...[/bold cyan]")

architecte = Agent(
  role='Architecte de Contenu Équestre',
  goal="Définir une structure de livre complète et cohérente à partir d'une liste de sujets.",
  backstory="Vous êtes un éditeur expérimenté spécialisé dans les manuels techniques et pédagogiques.",
  verbose=True, llm=OLLAMA_LLM, allow_delegation=False
)

chercheur = Agent(
  role='Chercheur Scientifique Équestre',
  goal='Collecter des informations détaillées, factuelles et vérifiables sur des sujets équestres spécifiques.',
  backstory="Vous êtes un documentaliste scientifique passionné par le monde équestre, expert en recherche de sources fiables.",
  verbose=True, llm=OLLAMA_LLM, tools=[search_tool], allow_delegation=False
)

validateur = Agent(
  role='Validateur Scientifique et Vétérinaire',
  goal="Analyser et synthétiser les informations collectées pour en garantir l'exactitude scientifique.",
  backstory="Vous êtes un vétérinaire équin avec une carrière en recherche. Votre mission est de valider les faits.",
  verbose=True, llm=OLLAMA_LLM, allow_delegation=False
)

redacteur = Agent(
  role='Rédacteur Pédagogique Équestre',
  goal="Transformer des synthèses scientifiques en un contenu textuel clair, engageant et accessible.",
  backstory="Vous êtes un auteur et formateur équestre reconnu, doué pour la vulgarisation.",
  verbose=True, llm=OLLAMA_LLM, allow_delegation=True
)

editeur = Agent(
  role='Éditeur en Chef',
  goal="Relire, corriger et harmoniser l'ensemble du texte pour une qualité de publication professionnelle.",
  backstory="Vous êtes le gardien de la qualité, un ancien professeur de lettres qui ne laisse passer aucune erreur.",
  verbose=True, llm=OLLAMA_LLM, allow_delegation=False
)

metteur_en_page = Agent(
    role='Spécialiste de la Mise en Page',
    goal="Formater le contenu final en un document Markdown bien structuré.",
    backstory="Vous êtes un designer de contenu numérique, maître de la syntaxe Markdown.",
    verbose=True, llm=OLLAMA_LLM, allow_delegation=False
)
console.print("[bold green]Agents créés avec succès ![/bold green]")

# --- 3. Définition des Tâches ---
console.print("[bold cyan]Définition des tâches...[/bold cyan]")
sujets_livre = [
    "anatomie_locomotion.md", "osteopathie_equine.md", "assurances_equines.md",
    "principes_apprentissage.md", "1_pansage_quotidien.md", "bases_nutrition.md",
    "choix_selle.md", "principes_parage.md"
]
console.print(f"[cyan]Sujets à traiter:[/cyan] {sujets_livre}")

tache_plan = Task(
  description=f"Crée un plan de livre détaillé à partir de la liste de sujets : {', '.join(sujets_livre)}. Organise-les en chapitres logiques.",
  expected_output="Un plan de livre structuré au format Markdown.",
  agent=architecte
)

tache_recherche = Task(
  description="Pour chaque sujet du plan, effectue une recherche approfondie pour collecter des informations factuelles.",
  expected_output="Un rapport de recherche pour chaque sujet, avec informations et URL des sources.",
  agent=chercheur, context=[tache_plan]
)

tache_validation = Task(
  description="Analyse les rapports de recherche. Vérifie l'exactitude et produis une synthèse scientifique pour chaque sujet.",
  expected_output="Une série de synthèses factuelles et validées, prêtes pour le rédacteur.",
  agent=validateur, context=[tache_recherche]
)

tache_redaction = Task(
  description="Rédige le contenu de chaque section à partir des synthèses. Le ton doit être pédagogique et clair.",
  expected_output="Le corps de texte complet du livre, chapitre par chapitre.",
  agent=redacteur, context=[tache_validation]
)

tache_edition = Task(
  description="Effectue une relecture complète du texte. Corrige toutes les erreurs de grammaire, style et typographie.",
  expected_output="Le texte final du livre, poli et sans erreur.",
  agent=editeur, context=[tache_redaction]
)

tache_mise_en_page = Task(
    description="Prends le texte final et mets-le en page au format Markdown. Ajoute un titre principal, des titres de chapitres (#) et de sections (##, ###).",
    expected_output="Un unique fichier Markdown (.md) contenant l'intégralité du livre, parfaitement structuré.",
    agent=metteur_en_page, context=[tache_edition],
    output_file='livre_equestre_final.md'
)
console.print("[bold green]Tâches définies avec succès ![/bold green]")

# --- 4. Création et Lancement de l'Équipe (Crew) ---
crew = Crew(
  agents=[architecte, chercheur, validateur, redacteur, editeur, metteur_en_page],
  tasks=[tache_plan, tache_recherche, tache_validation, tache_redaction, tache_edition, tache_mise_en_page],
  process=Process.sequential,
  verbose=True
)

console.print("[bold yellow]========================================[/bold yellow]")
console.print("[bold yellow]Lancement de l'équipe de création du livre...[/bold yellow]")
console.print(f"[yellow]Le processus peut prendre beaucoup de temps en fonction de la puissance de votre machine.[/yellow]")
console.print("[bold yellow]========================================[/bold yellow]")

result = crew.kickoff()

console.print("[bold green]========================================[/bold green]")
console.print("[bold green]Travail de l'équipe terminé ![/bold green]")
console.print("[bold cyan]Résultat final :[/bold cyan]")
console.print(result)
console.print("[bold green]========================================[/bold green]")
console.print(f"[green]Le livre a été sauvegardé dans le fichier : [bold]livre_equestre_final.md[/bold][/green]")