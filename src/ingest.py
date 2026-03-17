"""
Module d'ingestion du PDF IA Act dans ChromaDB.

❓ QUOI : Charge, découpe et vectorise le PDF IA Act EN
🎯 POURQUOI : Créer la base vectorielle pour recherche sémantique
🔧 COMMENT : PyPDF → Chunking → Embeddings → ChromaDB
⚠️ ATTENTION : Processus long (~5-10 min pour 100+ pages)
"""

import sys
from pathlib import Path
from typing import List

# ❓ QUOI : Imports LangChain pour RAG
# 🎯 POURQUOI : Outils spécialisés pour loading, splitting, embeddings
# ⚠️ ATTENTION : Utiliser nouvelles API (langchain_ollama vs déprécié)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    PDF_PATH,
    PDF_LANG,
    VECTORDB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    VERBOSE
)


def check_prerequisites() -> bool:
    """
    Vérifie que tous les prérequis sont présents.
    
    ❓ QUOI : Validation environnement avant ingestion
    🎯 POURQUOI : Éviter erreurs en cours de processus
    🔧 COMMENT : Vérifie PDF, Ollama, modèles
    ⚠️ ATTENTION : Arrête le script si erreur détectée
    
    Returns:
        bool: True si tout est OK, False sinon
    """
    errors = []
    
    # Vérifier PDF existe
    if not PDF_PATH.exists():
        errors.append(f"❌ PDF introuvable : {PDF_PATH}")
        errors.append(f"   Téléchargez AI_Act_EN.pdf depuis EUR-Lex")
    
    # Vérifier Ollama accessible
    try:
        import requests
        response = requests.get(OLLAMA_BASE_URL, timeout=2)
        if VERBOSE:
            print(f"✅ Ollama accessible : {OLLAMA_BASE_URL}")
    except Exception as e:
        errors.append(f"❌ Ollama inaccessible : {OLLAMA_BASE_URL}")
        errors.append(f"   Erreur : {e}")
        errors.append(f"   Démarrez Ollama avec 'ollama serve'")
    
    # Vérifier modèle embeddings disponible
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m['name'] for m in response.json().get('models', [])]
        
        # ❓ QUOI : Vérifier modèle avec ou sans tag :latest
        # 🎯 POURQUOI : Ollama peut retourner "model:latest" ou "model"
        # ⚠️ ATTENTION : Accepter les deux formats
        model_found = any(
            EMBEDDING_MODEL in model or model.startswith(f"{EMBEDDING_MODEL}:")
            for model in models
        )
        
        if model_found:
            if VERBOSE:
                print(f"✅ Modèle embeddings disponible : {EMBEDDING_MODEL}")
        else:
            errors.append(f"❌ Modèle embeddings manquant : {EMBEDDING_MODEL}")
            errors.append(f"   Téléchargez avec : ollama pull {EMBEDDING_MODEL}")
            errors.append(f"   Modèles disponibles : {', '.join(models)}")
    except Exception as e:
        errors.append(f"⚠️ Impossible de vérifier modèles Ollama : {e}")
    
    # Afficher erreurs si présentes
    if errors:
        print("\n" + "=" * 70)
        print("⚠️ ERREURS DÉTECTÉES")
        print("=" * 70)
        for error in errors:
            print(error)
        print("=" * 70 + "\n")
        return False
    
    return True


def load_pdf() -> List:
    """
    Charge le PDF IA Act avec PyPDFLoader.
    
    ❓ QUOI : Extraction texte du PDF page par page
    🎯 POURQUOI : PyPDFLoader conserve métadonnées (numéro page)
    🔧 COMMENT : Chaque page = 1 Document LangChain
    ⚠️ ATTENTION : Texte peut contenir headers/footers parasites
    
    Returns:
        List[Document]: Liste de documents (1 par page)
    """
    if VERBOSE:
        print(f"\n📄 Chargement du PDF : {PDF_PATH.name}")
        print(f"   Langue : {PDF_LANG}")
    
    # ❓ QUOI : PyPDFLoader lit PDF et crée Documents
    # 🎯 POURQUOI : Automatise extraction + métadonnées
    # ⚠️ ATTENTION : Peut être lent sur gros PDF (>100 pages)
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()
    
    if VERBOSE:
        print(f"✅ {len(documents)} pages chargées")
        print(f"   Exemple métadonnées : {documents[0].metadata}")
        print(f"   Aperçu page 1 : {documents[0].page_content[:200]}...")
    
    return documents


def split_documents(documents: List) -> List:
    """
    Découpe les documents en chunks optimisés pour texte juridique.
    
    ❓ QUOI : RecursiveCharacterTextSplitter pour chunks intelligents
    🎯 POURQUOI : Respecte structure texte (paragraphes, phrases)
    🔧 COMMENT : Priorité séparateurs : \\n\\n > \\n > . > espace
    ⚠️ ATTENTION : chunk_overlap évite coupure milieu d'article
    
    Args:
        documents (List[Document]): Documents issus du PDF
        
    Returns:
        List[Document]: Chunks avec métadonnées préservées
    """
    if VERBOSE:
        print(f"\n✂️ Découpage en chunks intelligents...")
        print(f"   Taille chunk : {CHUNK_SIZE} caractères")
        print(f"   Overlap : {CHUNK_OVERLAP} caractères")
        print(f"   Séparateurs : {CHUNK_SEPARATORS}")
    
    # ❓ QUOI : RecursiveCharacterTextSplitter = découpe intelligente
    # 🎯 POURQUOI : Essaye séparateurs dans l'ordre (paragraphe > phrase > mot)
    # 🔧 COMMENT : chunk_overlap partage contexte entre chunks adjacents
    # ⚠️ ATTENTION : length_function par défaut = len() (caractères, pas tokens)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,  # Compte caractères (approximation tokens)
        is_separator_regex=False
    )
    
    # ❓ QUOI : split_documents découpe tout en préservant métadonnées
    # 🎯 POURQUOI : Métadonnées (page, source) conservées pour chaque chunk
    # ⚠️ ATTENTION : Peut prendre 30s-1min sur gros documents
    chunks = text_splitter.split_documents(documents)
    
    if VERBOSE:
        print(f"✅ {len(chunks)} chunks créés")
        print(f"   Ratio : {len(chunks) / len(documents):.1f} chunks/page")
        print(f"   Exemple chunk 1 :")
        print(f"   {chunks[0].page_content[:300]}...")
        print(f"   Métadonnées : {chunks[0].metadata}")
    
    return chunks


def create_embeddings():
    """
    Initialise le modèle d'embeddings Ollama.
    
    ❓ QUOI : OllamaEmbeddings pour vectorisation
    🎯 POURQUOI : nomic-embed-text optimisé pour recherche anglais
    🔧 COMMENT : Connexion Ollama locale
    ⚠️ ATTENTION : Nécessite Ollama actif avec modèle téléchargé
    
    Returns:
        OllamaEmbeddings: Instance du modèle d'embeddings
    """
    if VERBOSE:
        print(f"\n🧮 Initialisation modèle embeddings...")
        print(f"   Modèle : {EMBEDDING_MODEL}")
        print(f"   Base URL : {OLLAMA_BASE_URL}")
    
    # ❓ QUOI : OllamaEmbeddings = wrapper LangChain pour Ollama
    # 🎯 POURQUOI : Compatible avec vectorstores LangChain (Chroma)
    # ⚠️ ATTENTION : Doit utiliser base_url, pas model_url
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    if VERBOSE:
        print(f"✅ Modèle embeddings prêt")
    
    return embeddings


def create_vectorstore(chunks: List, embeddings) -> Chroma:
    """
    Crée la base vectorielle ChromaDB et vectorise les chunks.
    
    ❓ QUOI : ChromaDB = base vectorielle persistante
    🎯 POURQUOI : Stocke vecteurs + permet recherche similarité
    🔧 COMMENT : from_documents() vectorise et stocke automatiquement
    ⚠️ ATTENTION : Processus LONG (~5-10min pour 300+ chunks)
    
    Args:
        chunks (List[Document]): Chunks à vectoriser
        embeddings (OllamaEmbeddings): Modèle d'embeddings
        
    Returns:
        Chroma: Instance de la base vectorielle
    """
    if VERBOSE:
        print(f"\n💾 Création base vectorielle ChromaDB...")
        print(f"   Nombre de chunks : {len(chunks)}")
        print(f"   Dossier : {VECTORDB_DIR}")
        print(f"   ⏱️ Temps estimé : ~{len(chunks) * 0.5:.0f}s")
        print(f"\n⏳ Vectorisation en cours (cela peut prendre plusieurs minutes)...")
    
    # ❓ QUOI : Créer dossier vectordb si inexistant
    # 🎯 POURQUOI : ChromaDB nécessite dossier pour persist_directory
    # ⚠️ ATTENTION : parents=True crée dossiers parents si besoin
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
    
    # ❓ QUOI : from_documents() = vectorise + stocke en 1 appel
    # 🎯 POURQUOI : Automatise embed + insertion ChromaDB
    # 🔧 COMMENT : persist_directory = sauvegarde sur disque (réutilisable)
    # ⚠️ ATTENTION : Si dossier existe, peut causer erreurs (supprimer d'abord)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORDB_DIR),
        collection_name="ia_act_en"  # Nom de la collection dans ChromaDB
    )
    
    if VERBOSE:
        print(f"✅ Base vectorielle créée avec succès")
        print(f"   Collection : ia_act_en")
        print(f"   Emplacement : {VECTORDB_DIR}")
    
    return vectorstore


def test_vectorstore(vectorstore: Chroma):
    """
    Teste la base vectorielle avec une requête simple.
    
    ❓ QUOI : Validation que recherche fonctionne
    🎯 POURQUOI : Vérifier ingestion OK avant utilisation
    🔧 COMMENT : similarity_search sur question test
    ⚠️ ATTENTION : Résultats doivent être pertinents
    
    Args:
        vectorstore (Chroma): Base vectorielle à tester
    """
    if VERBOSE:
        print(f"\n🧪 Test de la base vectorielle...")
    
    # ❓ QUOI : Question test en anglais (corpus EN)
    # 🎯 POURQUOI : Vérifier retrieval fonctionne
    # ⚠️ ATTENTION : Question doit matcher contenu IA Act
    test_query = "What is a high-risk AI system?"
    
    # ❓ QUOI : similarity_search = recherche par similarité cosinus
    # 🎯 POURQUOI : Trouve chunks les plus proches sémantiquement
    # 🔧 COMMENT : k=3 = retourne top-3 résultats
    # ⚠️ ATTENTION : Peut être lent première fois (index loading)
    results = vectorstore.similarity_search(test_query, k=3)
    
    if VERBOSE:
        print(f"   Question test : {test_query}")
        print(f"   Résultats trouvés : {len(results)}")
        
        for i, doc in enumerate(results, 1):
            print(f"\n   📄 Résultat {i} :")
            print(f"      Page : {doc.metadata.get('page', 'N/A')}")
            print(f"      Extrait : {doc.page_content[:200]}...")
    
    if len(results) > 0:
        print(f"\n✅ Test réussi - Base vectorielle fonctionnelle")
    else:
        print(f"\n⚠️ Aucun résultat trouvé - Vérifier la base")


def main():
    """
    Point d'entrée principal du module d'ingestion.
    
    ❓ QUOI : Pipeline complet d'ingestion
    🎯 POURQUOI : Centralise toutes les étapes
    🔧 COMMENT : check → load → split → embed → store → test
    ⚠️ ATTENTION : Processus long, ne pas interrompre
    """
    print("=" * 70)
    print("🚀 INGESTION PDF IA ACT DANS CHROMADB")
    print("=" * 70)
    
    # Étape 1 : Vérifications prérequis
    print("\n📋 Étape 1/5 : Vérification des prérequis")
    print("-" * 70)
    if not check_prerequisites():
        print("\n❌ Ingestion annulée - Corrigez les erreurs ci-dessus")
        sys.exit(1)
    
    # Étape 2 : Chargement PDF
    print("\n📋 Étape 2/5 : Chargement du PDF")
    print("-" * 70)
    try:
        documents = load_pdf()
    except Exception as e:
        print(f"❌ Erreur chargement PDF : {e}")
        sys.exit(1)
    
    # Étape 3 : Découpage en chunks
    print("\n📋 Étape 3/5 : Découpage en chunks")
    print("-" * 70)
    try:
        chunks = split_documents(documents)
    except Exception as e:
        print(f"❌ Erreur découpage : {e}")
        sys.exit(1)
    
    # Étape 4 : Création embeddings et vectorisation
    print("\n📋 Étape 4/5 : Vectorisation (ÉTAPE LONGUE)")
    print("-" * 70)
    try:
        embeddings = create_embeddings()
        vectorstore = create_vectorstore(chunks, embeddings)
    except Exception as e:
        print(f"❌ Erreur vectorisation : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Étape 5 : Test de validation
    print("\n📋 Étape 5/5 : Test de validation")
    print("-" * 70)
    try:
        test_vectorstore(vectorstore)
    except Exception as e:
        print(f"⚠️ Erreur test : {e}")
        print(f"   (La base est probablement OK malgré l'erreur de test)")
    
    # Récapitulatif final
    print("\n" + "=" * 70)
    print("✅ INGESTION TERMINÉE AVEC SUCCÈS")
    print("=" * 70)
    print(f"\n📊 Statistiques :")
    print(f"   Pages traitées : {len(documents)}")
    print(f"   Chunks créés : {len(chunks)}")
    print(f"   Base vectorielle : {VECTORDB_DIR}")
    print(f"   Collection : ia_act_en")
    print(f"\n🎯 Prochaine étape : Utiliser retriever.py pour recherches")
    print("=" * 70 + "\n")


# === POINT D'ENTRÉE ===
if __name__ == "__main__":
    main()
