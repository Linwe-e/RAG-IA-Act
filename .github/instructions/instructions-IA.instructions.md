---
applyTo: '**'
---
# Instructions pour Copilot - Projet RAG IA Act

## Contexte du Projet
Je développe un système RAG (Retrieval-Augmented Generation) pour interroger la loi IA Act européenne. C'est un projet pédagogique pour apprendre LangChain et les embeddings vectoriels.

## Stack Technique
- **Python 3.10+** sur Windows
- **LangChain** : Framework pour chaîner les opérations LLM
- **ChromaDB** : Base de données vectorielle persistante
- **Ollama** : LLM local + embeddings
  - **Embeddings** : nomic-embed-text (optimisé anglais, 768 dim)
  - **LLM** : mistral:7b-instruct (bilingue FR/EN)
- **Streamlit** : Interface utilisateur
- **PyPDF** : Parsing du PDF de l'IA Act

## Ressources 
- https://blog.stephane-robert.info/docs/developper/programmation/python/rag-introduction/

## Stratégie Langue : Pipeline Bilingue Optimisé
**Décision Technique** : Utiliser l'IA Act en **ANGLAIS** avec réponses en français
- **Corpus source** : AI_Act_EN.pdf (version officielle UE)
- **Embeddings** : nomic-embed-text (optimisé anglais, +30% précision vs français)
- **Traduction** : Questions FR → EN avant recherche vectorielle
- **Génération** : LLM bilingue (Mistral 7B) pour réponses en français
- **Justification** : Précision retrieval maximale + stack locale légère

## Structure du Projet
rag-ai-act/
├── data/
│   └── AI_Act_FR_TXT.pdf   # Corpus source FRANÇAIS (non utilisé)
│   └── AI_Act_EN.pdf       # Corpus source ANGLAIS (officiel UE)
├── vectordb/               # Stockage ChromaDB
├── src/
│   ├── config.py          # Configuration centralisée (NEW)
│   ├── query_translator.py # Traduction FR→EN (NEW)
│   ├── ingest.py          # Chargement + chunking + vectorisation
│   ├── retriever.py       # Recherche sémantique
│   └── generator.py       # Génération de réponses
├── app.py                 # Interface Streamlit
├── requirements.txt
├── docs/
│   └── guide-rag-stephane-robert.md
└── .github\instructions    
    └── instructions-IA.instructions.md  # Ce fichier

## Principes de Code
1. **Pédagogie prioritaire** : Commente CHAQUE abstraction LangChain la première fois
2. **Modularité** : Séparer ingestion, retrieval et generation
3. **Offline-first** : Tout doit fonctionner sans Internet
4. **Logs verbeux** : Afficher les étapes du pipeline (nombre de chunks, scores de similarité, etc.)
5. **Open source strict** : Pas d'API propriétaires

## Conventions de Nommage
- **Variables** : `snake_case` (ex: `vector_store`, `chunk_size`)
- **Fonctions** : `snake_case` descriptif (ex: `load_and_split_pdf()`, `create_retrieval_chain()`)
- **Classes** : `PascalCase` si nécessaire (ex: `CustomRetriever`)
- **Constantes** : `UPPER_SNAKE_CASE` (ex: `CHUNK_SIZE = 500`)

## Gestion des Dépendances
Versions fixées dans `requirements.txt` :
```
langchain
langchain-community
chromadb
streamlit
pypdf
```

## Abstractions LangChain à Expliquer
Quand tu génères du code avec ces composants, ajoute un commentaire explicatif :

### 1. Documents
```python
# Document = Unité de base LangChain
# Structure : {page_content: str, metadata: dict}
# Métadonnées auto : source, page number
```

### 2. TextSplitter
```python
# RecursiveCharacterTextSplitter = Découpe intelligente
# Respecte les séparateurs (\n\n, \n, ., " ") dans l'ordre
# chunk_overlap = contexte partagé entre chunks adjacents
```

### 3. Embeddings
```python
# OllamaEmbeddings = Convertit texte en vecteur numérique
# nomic-embed-text = Modèle optimisé pour recherche sémantique
# Dimension : 768 (vs 1536 pour OpenAI)
```

### 4. VectorStore
```python
# Chroma = Base vectorielle avec persistance disque
# Calcule automatiquement similarité cosinus
# Méthodes clés : .from_documents(), .similarity_search()
```

### 5. Chains
```python
# Chain = Pipeline d'opérations enchaînées
# RetrievalQA = Question → Retrieve → Generate → Answer
# Gère automatiquement le prompt engineering
```
## Patterns à Utiliser

### Chargement avec gestion d'erreur
```python
try:
    loader = PyPDFLoader("data/ai_act.pdf")
    docs = loader.load()
    print(f"✅ {len(docs)} pages chargées")
except FileNotFoundError:
    print("❌ Fichier ai_act.pdf introuvable dans data/")
    sys.exit(1)
```

### Logs de progression
```python
print(f"📄 Découpage en chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
chunks = splitter.split_documents(docs)
print(f"✅ {len(chunks)} chunks créés")
```

### Vérification Ollama
```python
# Toujours vérifier que Ollama tourne avant d'appeler les modèles
import requests
try:
    requests.get("http://localhost:11434")
except:
    print("❌ Ollama n'est pas démarré. Lance 'ollama serve'")
    sys.exit(1)
```
## Optimisations Spécifiques

### Chunking pour texte légal
```python
# L'IA Act a une structure hiérarchique (Titres → Chapitres → Articles)
# Privilégier chunk_size=1000-1500 pour garder contexte juridique
# chunk_overlap=200 pour transitions entre articles
```

### Retrieval
```python
# Pour questions juridiques : k=5-7 chunks (vs 3-4 standard)
# Ajouter score_threshold=0.7 pour filtrer résultats peu pertinents
```

### Prompting
```python
# Template système pour réponses juridiques :
# "Tu es un assistant spécialisé en droit européen.
#  Réponds uniquement en te basant sur le contexte fourni.
#  Cite les articles pertinents entre crochets [Article X]."
```
## Points d'Attention

- **Chemins Windows** : Utiliser `Path` de `pathlib` pour compatibilité
- **Encodage** : Spécifier `encoding='utf-8'` pour fichiers texte
- **Mémoire** : ChromaDB peut être gourmand → limiter `k` si nécessaire
- **Ollama timeout** : Ajouter `request_timeout=60` pour gros contextes

## Workflow de Développement

1. Tester chaque module indépendamment avant intégration
2. Utiliser `if __name__ == "__main__":` pour scripts exécutables
3. Ajouter `--verbose` flag pour logs détaillés (optionnel)

## Questions Fréquentes à Anticiper

**Q : Pourquoi ChromaDB et pas FAISS ?**  
→ ChromaDB persiste sur disque automatiquement, FAISS nécessite sauvegarde manuelle

**Q : Pourquoi nomic-embed-text ?**  
→ Optimisé pour recherche (vs génération), dimension réduite (768), licence Apache 2.0

**Q : Chunk overlap à quoi ça sert ?**  
→ Évite de couper une phrase/idée entre 2 chunks, améliore cohérence contextuelle
## Niveau de Compétence de l'Utilisateur

- **Bootcamp IA** : Notions théoriques acquises, pratique débutante
- **Python** : Syntaxe basique maîtrisée, concepts avancés à expliquer
- **Préférences** : Clarté > Concision, analogies bienvenues, jargon décomposé

## Format des Explications Souhaitées
```
# ❓ QUOI : Description courte de l'objet/fonction
# 🎯 POURQUOI : Problème résolu ou avantage apporté
# 🔧 COMMENT : Exemple d'utilisation minimal
# ⚠️ ATTENTION : Pièges courants ou limitations
```

## Objectif Final
Créer un RAG fonctionnel, compréhensible et réutilisable pour d'autres corpus techniques.