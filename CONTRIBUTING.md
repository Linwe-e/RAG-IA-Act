# Contributing Guide

Merci de contribuer a ce projet RAG IA Act.

## Prerequisites
- Python 3.10+
- Ollama installe et demarre
- Modeles Ollama:
  - `nomic-embed-text`
  - `mistral:latest`

## Local Setup
1. Cloner le depot.
2. Creer un environnement virtuel.
3. Installer les dependances:
   - `pip install -r requirements.txt`
4. Verifier Ollama:
   - `ollama list`

## Development Workflow
1. Lancer l'ingestion (une fois):
   - `python src/ingest.py`
2. Lancer l'application:
   - `streamlit run app.py`
3. Tester les modules:
   - `python tests/manual/quick_validation.py`
   - `python tests/manual/expansion_validation.py`
   - `python src/query_translator.py`
   - `python src/retriever.py`
   - `python src/generator.py`

## Pull Requests
- Garder des commits clairs et atomiques.
- Ajouter/mettre a jour la documentation si necessaire.
- Verifier que les scripts de test manuels passent.
- Eviter d'ajouter des artefacts locaux (venv, logs, vectordb).

## Code Style
- Respecter le style Python existant du projet.
- Utiliser des noms explicites en `snake_case`.
- Conserver les commentaires pedagogiques sur les abstractions LangChain.
