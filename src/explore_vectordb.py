"""
Script d'exploration de la base vectorielle ChromaDB.

❓ QUOI : Interface interactive pour tester la recherche sémantique
🎯 POURQUOI : Valider qualité retrieval et explorer contenu
🔧 COMMENT : Queries en anglais ou français avec traduction auto
⚠️ ATTENTION : Base vectorielle doit exister (lancer ingest.py d'abord)
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    VECTORDB_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    TOP_K_RESULTS,
    VERBOSE
)
from query_translator import translate_query_to_english


def load_vectorstore() -> Chroma:
    """
    Charge la base vectorielle existante.
    
    ❓ QUOI : Connexion à ChromaDB existant
    🎯 POURQUOI : Réutiliser base sans revectoriser
    🔧 COMMENT : Chroma() avec persist_directory
    ⚠️ ATTENTION : Dossier vectordb/ doit exister
    
    Returns:
        Chroma: Instance de la base vectorielle
    """
    print("📂 Chargement de la base vectorielle...")
    print(f"   Emplacement : {VECTORDB_DIR}")
    
    if not VECTORDB_DIR.exists():
        raise FileNotFoundError(
            f"❌ Base vectorielle introuvable : {VECTORDB_DIR}\n"
            f"   Lancez d'abord : python src/ingest.py"
        )
    
    # ❓ QUOI : Réinitialiser embeddings (même config que ingestion)
    # 🎯 POURQUOI : ChromaDB nécessite même modèle pour recherche
    # ⚠️ ATTENTION : Doit être identique à celui utilisé pour vectorisation
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    # ❓ QUOI : Chroma() charge base existante (vs from_documents créer)
    # 🎯 POURQUOI : Accès lecture seule, pas de modification
    # ⚠️ ATTENTION : collection_name doit matcher celui de l'ingestion
    vectorstore = Chroma(
        persist_directory=str(VECTORDB_DIR),
        embedding_function=embeddings,
        collection_name="ia_act_en"
    )
    
    # Obtenir stats collection
    collection = vectorstore._collection
    count = collection.count()
    
    print(f"✅ Base chargée avec succès")
    print(f"   Collection : ia_act_en")
    print(f"   Nombre de vecteurs : {count}")
    
    return vectorstore


def search_similarity(vectorstore: Chroma, query: str, k: int = None, lang: str = "auto"):
    """
    Recherche par similarité avec affichage détaillé.
    
    ❓ QUOI : Similarity search + affichage résultats
    🎯 POURQUOI : Tester retrieval et voir scores
    🔧 COMMENT : similarity_search_with_score pour scores
    ⚠️ ATTENTION : lang="auto" détecte si traduction nécessaire
    
    Args:
        vectorstore (Chroma): Base vectorielle
        query (str): Question utilisateur
        k (int): Nombre de résultats (défaut: config.TOP_K_RESULTS)
        lang (str): Langue query ("fr", "en", "auto")
    """
    if k is None:
        k = TOP_K_RESULTS
    
    print("\n" + "=" * 70)
    print("🔍 RECHERCHE SÉMANTIQUE")
    print("=" * 70)
    
    # Détecter langue et traduire si nécessaire
    query_search = query
    if lang == "auto":
        # Heuristique simple : si caractères français détectés, traduire
        french_chars = any(c in query for c in "àâäéèêëïîôùûüÿç")
        french_words = any(w in query.lower() for w in ["qu'est", "quel", "quelle", "comment", "pourquoi"])
        
        if french_chars or french_words:
            print(f"🌐 Langue détectée : Français")
            print(f"   Question originale : {query}")
            query_search = translate_query_to_english(query)
            print(f"   Question traduite : {query_search}")
        else:
            print(f"🌐 Langue détectée : Anglais")
            print(f"   Question : {query}")
    elif lang == "fr":
        print(f"🌐 Traduction FR→EN...")
        print(f"   Question FR : {query}")
        query_search = translate_query_to_english(query)
        print(f"   Question EN : {query_search}")
    else:
        print(f"🔍 Question EN : {query}")
    
    print(f"\n📊 Recherche top-{k} résultats...")
    
    # ❓ QUOI : similarity_search_with_score retourne (doc, score)
    # 🎯 POURQUOI : Voir scores de similarité pour évaluer pertinence
    # 🔧 COMMENT : Score = distance cosinus (0=identique, 1=opposé)
    # ⚠️ ATTENTION : ChromaDB retourne distance, pas similarité (1 - distance)
    results = vectorstore.similarity_search_with_score(query_search, k=k)
    
    print(f"✅ {len(results)} résultats trouvés\n")
    
    # Afficher résultats
    for i, (doc, score) in enumerate(results, 1):
        similarity = 1 - score  # Convertir distance en similarité
        
        print("─" * 70)
        print(f"📄 Résultat {i}/{len(results)}")
        print("─" * 70)
        print(f"📊 Score similarité : {similarity:.4f} (distance: {score:.4f})")
        print(f"📖 Page : {doc.metadata.get('page', 'N/A')}")
        print(f"📝 Source : {doc.metadata.get('source', 'N/A').split('\\')[-1]}")
        
        # Afficher extrait (limité pour lisibilité)
        content = doc.page_content
        if len(content) > 400:
            content = content[:400] + "..."
        print(f"\n💬 Contenu :\n{content}\n")
    
    return results


def explore_metadata(vectorstore: Chroma):
    """
    Explore les métadonnées disponibles dans la collection.
    
    ❓ QUOI : Afficher infos sur structure collection
    🎯 POURQUOI : Comprendre ce qui est stocké
    🔧 COMMENT : Accès direct à collection ChromaDB
    ⚠️ ATTENTION : Fonctions internes ChromaDB peuvent changer
    """
    print("\n" + "=" * 70)
    print("📊 MÉTADONNÉES DE LA COLLECTION")
    print("=" * 70)
    
    collection = vectorstore._collection
    
    # Stats générales
    count = collection.count()
    print(f"\n📈 Statistiques :")
    print(f"   Nombre total de chunks : {count}")
    
    # Peek quelques documents pour voir métadonnées
    print(f"\n🔍 Aperçu métadonnées (5 premiers chunks) :")
    results = collection.peek(limit=5)
    
    if results and 'metadatas' in results and results['metadatas']:
        for i, metadata in enumerate(results['metadatas'], 1):
            print(f"\n   Chunk {i} :")
            for key, value in metadata.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"      {key}: {value}")
    
    # Résumer champs métadonnées disponibles
    if results and 'metadatas' in results and results['metadatas']:
        all_keys = set()
        for metadata in results['metadatas']:
            all_keys.update(metadata.keys())
        
        print(f"\n📋 Champs métadonnées disponibles :")
        for key in sorted(all_keys):
            print(f"   • {key}")


def interactive_mode(vectorstore: Chroma):
    """
    Mode interactif pour explorer la base.
    
    ❓ QUOI : Boucle questions/réponses
    🎯 POURQUOI : Tester plusieurs requêtes facilement
    🔧 COMMENT : Input utilisateur en boucle
    ⚠️ ATTENTION : 'quit' ou 'exit' pour sortir
    """
    print("\n" + "=" * 70)
    print("🎮 MODE INTERACTIF")
    print("=" * 70)
    print("\nCommandes :")
    print("  - Tapez votre question (FR ou EN)")
    print("  - 'meta' : Voir métadonnées collection")
    print("  - 'quit' ou 'exit' : Quitter")
    print("=" * 70)
    
    while True:
        try:
            query = input("\n💬 Question : ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir !")
                break
            
            if query.lower() == 'meta':
                explore_metadata(vectorstore)
                continue
            
            # Recherche
            search_similarity(vectorstore, query, lang="auto")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted - Au revoir !")
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            import traceback
            traceback.print_exc()


def main():
    """Point d'entrée principal."""
    print("=" * 70)
    print("🔍 EXPLORATEUR BASE VECTORIELLE - RAG IA ACT")
    print("=" * 70)
    
    # Charger base vectorielle
    try:
        vectorstore = load_vectorstore()
    except Exception as e:
        print(f"\n❌ Erreur chargement base : {e}")
        return
    
    # Menu principal
    print("\n" + "=" * 70)
    print("🎯 MENU PRINCIPAL")
    print("=" * 70)
    print("1. Mode interactif (poser plusieurs questions)")
    print("2. Test rapide (question prédéfinie)")
    print("3. Explorer métadonnées")
    print("4. Quitter")
    print("=" * 70)
    
    choice = input("\nVotre choix (1-4) : ").strip()
    
    if choice == "1":
        interactive_mode(vectorstore)
    
    elif choice == "2":
        # Tests prédéfinis
        test_queries = [
            ("What is a high-risk AI system?", "en"),
            ("Qu'est-ce qu'un système d'IA à haut risque ?", "fr"),
            ("What are the obligations of AI providers?", "en"),
            ("Quelles sont les sanctions en cas de non-conformité ?", "fr")
        ]
        
        print(f"\n🧪 Test avec {len(test_queries)} questions...\n")
        
        for query, lang in test_queries:
            search_similarity(vectorstore, query, k=3, lang=lang)
            input("\n⏸️ Appuyez sur Entrée pour la question suivante...")
    
    elif choice == "3":
        explore_metadata(vectorstore)
    
    elif choice == "4":
        print("👋 Au revoir !")
    
    else:
        print("❌ Choix invalide")


if __name__ == "__main__":
    main()
