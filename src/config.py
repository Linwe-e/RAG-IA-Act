"""
Configuration centralisée du système RAG IA Act.

❓ QUOI : Paramètres du pipeline bilingue EN→FR
🎯 POURQUOI : Centraliser configuration pour faciliter ajustements
🔧 COMMENT : Importer ce module dans chaque script
⚠️ ATTENTION : Modifier ces paramètres impacte tout le système
"""

from pathlib import Path

# === CHEMINS PROJET ===
# ❓ QUOI : Chemins absolus des dossiers principaux
# 🎯 POURQUOI : Compatibilité Windows, évite erreurs de chemins relatifs
# ⚠️ ATTENTION : PROJECT_ROOT détecté automatiquement depuis ce fichier
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
DOCS_DIR = PROJECT_ROOT / "docs"

# === CORPUS SOURCE ===
# ❓ QUOI : Fichier PDF de l'IA Act à ingérer
# 🎯 POURQUOI : Version ANGLAISE pour embeddings optimaux (+30% précision)
# ⚠️ ATTENTION : Télécharger AI_Act_EN.pdf depuis EUR-Lex avant ingestion
PDF_PATH = DATA_DIR / "AI_Act_EN.pdf"
PDF_LANG = "en"  # Langue du corpus source

# === EMBEDDINGS ===
# ❓ QUOI : Modèle de conversion texte→vecteurs numériques
# 🎯 POURQUOI : nomic-embed-text optimisé pour recherche en anglais
# 🔧 COMMENT : Télécharger avec `ollama pull nomic-embed-text`
# ⚠️ ATTENTION : Nécessite Ollama actif (http://localhost:11434)
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768  # Dimension des vecteurs (fixe pour ce modèle)
OLLAMA_BASE_URL = "http://localhost:11434"

# === CHUNKING ===
# ❓ QUOI : Découpage du PDF en segments (chunks)
# 🎯 POURQUOI : Chunks 1200 tokens = optimal pour articles juridiques EN
# 🔧 COMMENT : RecursiveCharacterTextSplitter respecte structure texte
# ⚠️ ATTENTION : chunk_overlap évite coupure entre articles liés
CHUNK_SIZE = 1200  # Tokens par chunk
CHUNK_OVERLAP = 200  # Tokens partagés entre chunks adjacents
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Ordre priorité découpage

# === RETRIEVAL ===
# ❓ QUOI : Paramètres de recherche sémantique dans ChromaDB
# 🎯 POURQUOI : k=6 pour couvrir articles complexes (vs 3-4 standard)
# ⚠️ ATTENTION : score_threshold filtre résultats peu pertinents
TOP_K_RESULTS = 6  # Nombre de chunks récupérés
SIMILARITY_THRESHOLD = 0.3  # Score minimum (0-1, cosine similarity) - Abaissé pour texte juridique
DISTANCE_METRIC = "cosine"  # Métrique de distance (cosine ou euclidean)

# === TRADUCTION QUERY ===
# ❓ QUOI : Traduire questions françaises→anglais avant recherche
# 🎯 POURQUOI : Embeddings nomic-embed-text +30% précis sur anglais
# 🔧 COMMENT : Mistral comme traducteur (température 0 = littéral)
# ⚠️ ATTENTION : Ajoute ~500ms latence mais améliore retrieval
TRANSLATE_QUERY = True
QUERY_TRANSLATION_MODEL = "mistral:latest"  # Utilise mistral déjà installé
QUERY_TRANSLATION_TEMP = 0  # Température 0 = traduction littérale

# === LLM GÉNÉRATION ===
# ❓ QUOI : Modèle pour générer réponses finales en français
# 🎯 POURQUOI : Mistral bilingue natif FR/EN, performant sur texte juridique
# 🔧 COMMENT : Utilise mistral:latest (équivalent mistral:7b-instruct)
# ⚠️ ATTENTION : request_timeout pour contextes longs (6 chunks)
LLM_MODEL = "mistral:latest"  # Utilise mistral déjà installé
LLM_TEMPERATURE = 0.1  # Faible = réponses factuelles (vs créatives)
LLM_TIMEOUT = 60  # Secondes max pour génération
LLM_CONTEXT_WINDOW = 8192  # Tokens max supportés (Mistral 7B)

# === PROMPT TEMPLATE ===
# ❓ QUOI : Structure du prompt système pour le LLM
# 🎯 POURQUOI : Guide le LLM pour réponses juridiques précises en français
# ⚠️ ATTENTION : {context} et {question} seront remplacés dynamiquement
SYSTEM_PROMPT_TEMPLATE = """Tu es un assistant spécialisé dans la réglementation européenne sur l'intelligence artificielle.

INSTRUCTIONS :
1. Base ta réponse EXCLUSIVEMENT sur le CONTEXTE fourni ci-dessous (extraits de l'IA Act européenne)
2. Si l'information n'est pas présente dans le contexte, réponds exactement : "L'information n'est pas disponible dans les documents fournis"
3. Cite les articles pertinents entre crochets, par exemple [Article 15]
4. Réponds en FRANÇAIS de manière claire, précise et structurée
5. Structure recommandée : définition → explication → exemple concret si pertinent
6. Utilise un ton professionnel mais accessible

CONTEXTE (extraits IA Act en anglais) :
---
{context}
---

QUESTION : {question}

RÉPONSE EN FRANÇAIS :"""

# === STREAMLIT UI ===
# ❓ QUOI : Configuration de l'interface utilisateur
# 🎯 POURQUOI : Personnalisation affichage et expérience utilisateur
UI_TITLE = "🤖 Assistant IA Act - Réglementation Européenne IA"
UI_DESCRIPTION = """
Posez vos questions sur la réglementation européenne de l'intelligence artificielle (IA Act).
Ce système utilise un pipeline RAG bilingue pour vous fournir des réponses précises basées sur le texte officiel.
"""
UI_LANG = "fr"  # Langue interface
SHOW_SOURCE_CHUNKS = True  # Afficher chunks sources dans réponses
SHOW_SIMILARITY_SCORES = True  # Afficher scores de similarité

# === LOGS & DEBUG ===
# ❓ QUOI : Niveau de verbosité des logs
# 🎯 POURQUOI : Mode verbose = pédagogique (voir étapes du pipeline)
# ⚠️ ATTENTION : Mettre False en production pour performances
VERBOSE = False  # Affiche étapes détaillées (traduction, retrieval, etc.)
LOG_QUERIES = True  # Log toutes les questions posées (analytics)

# === VALIDATION CONFIGURATION ===
# ❓ QUOI : Vérifications au chargement du module
# 🎯 POURQUOI : Détecter erreurs de configuration tôt
# ⚠️ ATTENTION : Lance erreurs si configuration invalide

def validate_config():
    """Valide la cohérence de la configuration."""
    errors = []
    
    # Vérifier existence dossiers
    if not DATA_DIR.exists():
        errors.append(f"❌ Dossier data/ introuvable : {DATA_DIR}")
    
    # Vérifier existence PDF
    if not PDF_PATH.exists():
        errors.append(f"❌ PDF IA Act introuvable : {PDF_PATH}\n"
                     f"   Téléchargez AI_Act_EN.pdf dans {DATA_DIR}/")
    
    # Vérifier paramètres chunking
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(f"❌ CHUNK_OVERLAP ({CHUNK_OVERLAP}) doit être < CHUNK_SIZE ({CHUNK_SIZE})")
    
    # Vérifier paramètres retrieval
    if not 0 <= SIMILARITY_THRESHOLD <= 1:
        errors.append(f"❌ SIMILARITY_THRESHOLD doit être entre 0 et 1 (actuel: {SIMILARITY_THRESHOLD})")
    
    if TOP_K_RESULTS < 1:
        errors.append(f"❌ TOP_K_RESULTS doit être >= 1 (actuel: {TOP_K_RESULTS})")
    
    # Afficher erreurs si présentes
    if errors:
        print("\n".join(errors))
        raise ValueError("Configuration invalide. Corrigez les erreurs ci-dessus.")
    
    if VERBOSE:
        print("✅ Configuration validée avec succès")


# === AUTO-VALIDATION AU CHARGEMENT ===
# ❓ QUOI : Valide config automatiquement quand module importé
# 🎯 POURQUOI : Détection précoce des erreurs
# ⚠️ ATTENTION : Commenter cette ligne si validation bloque développement
if __name__ != "__main__":
    pass  # Désactivé pour permettre import avant téléchargement PDF
    # Décommenter après avoir placé AI_Act_EN.pdf dans data/ :
    # validate_config()


# === MODE TEST ===
if __name__ == "__main__":
    """Affiche la configuration actuelle quand exécuté directement."""
    print("=" * 60)
    print("📋 CONFIGURATION RAG IA ACT")
    print("=" * 60)
    print(f"\n📁 Chemins :")
    print(f"  - Projet : {PROJECT_ROOT}")
    print(f"  - Data : {DATA_DIR}")
    print(f"  - VectorDB : {VECTORDB_DIR}")
    print(f"  - PDF : {PDF_PATH} {'✅' if PDF_PATH.exists() else '❌ MANQUANT'}")
    
    print(f"\n🧮 Embeddings :")
    print(f"  - Modèle : {EMBEDDING_MODEL}")
    print(f"  - Dimension : {EMBEDDING_DIM}")
    
    print(f"\n✂️ Chunking :")
    print(f"  - Taille : {CHUNK_SIZE} tokens")
    print(f"  - Overlap : {CHUNK_OVERLAP} tokens")
    
    print(f"\n🔍 Retrieval :")
    print(f"  - Top-K : {TOP_K_RESULTS} chunks")
    print(f"  - Seuil similarité : {SIMILARITY_THRESHOLD}")
    print(f"  - Traduction query : {'✅ Activée' if TRANSLATE_QUERY else '❌ Désactivée'}")
    
    print(f"\n🤖 LLM :")
    print(f"  - Modèle : {LLM_MODEL}")
    print(f"  - Température : {LLM_TEMPERATURE}")
    print(f"  - Contexte max : {LLM_CONTEXT_WINDOW} tokens")
    
    print(f"\n🎨 Interface :")
    print(f"  - Langue : {UI_LANG}")
    print(f"  - Afficher sources : {'✅' if SHOW_SOURCE_CHUNKS else '❌'}")
    
    print(f"\n🔧 Debug :")
    print(f"  - Verbose : {'✅' if VERBOSE else '❌'}")
    print(f"  - Log queries : {'✅' if LOG_QUERIES else '❌'}")
    
    print("\n" + "=" * 60)
    print("🧪 Test de validation...")
    print("=" * 60)
    try:
        validate_config()
    except ValueError as e:
        print(f"\n⚠️ Erreurs détectées :\n{e}")
