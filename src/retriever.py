"""
Module de retrieval orchestré pour le RAG IA Act.

❓ QUOI : Recherche sémantique avec traduction automatique
🎯 POURQUOI : Centraliser pipeline retrieval (traduction + vectorDB)
🔧 COMMENT : Question FR → EN → ChromaDB → Chunks pertinents
⚠️ ATTENTION : Nécessite base vectorielle créée (ingest.py)
"""

from typing import List, Tuple, Dict, Any
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from config import (
    VECTORDB_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    TRANSLATE_QUERY,
    VERBOSE
)
from query_translator import translate_query_to_english, expand_citizen_query


class IAActRetriever:
    """
    Retriever orchestré pour recherche dans IA Act.
    
    ❓ QUOI : Classe encapsulant logique retrieval
    🎯 POURQUOI : Interface propre, réutilisable, testable
    🔧 COMMENT : Initialisation → load vectorstore → retrieve()
    ⚠️ ATTENTION : Singleton recommandé (chargement vectorstore coûteux)
    """
    
    def __init__(self):
        """
        Initialise le retriever et charge la base vectorielle.
        
        ❓ QUOI : Setup embeddings + connexion ChromaDB
        🎯 POURQUOI : Préparer infrastructure recherche
        ⚠️ ATTENTION : Peut prendre quelques secondes au premier appel
        """
        if VERBOSE:
            print("🔧 Initialisation du retriever...")
        
        self.vectorstore = self._load_vectorstore()
        
        if VERBOSE:
            print("✅ Retriever prêt")
    
    def _load_vectorstore(self) -> Chroma:
        """
        Charge la base vectorielle ChromaDB.
        
        ❓ QUOI : Connexion à base existante
        🎯 POURQUOI : Réutiliser vecteurs sans recalcul
        🔧 COMMENT : Même config embeddings que ingestion
        ⚠️ ATTENTION : Doit matcher config ingest.py
        
        Returns:
            Chroma: Instance de la base vectorielle
        """
        if not VECTORDB_DIR.exists():
            raise FileNotFoundError(
                f"❌ Base vectorielle introuvable : {VECTORDB_DIR}\n"
                f"   Lancez d'abord : python src/ingest.py"
            )
        
        # Initialiser embeddings (même config que ingestion)
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        # Charger base existante
        vectorstore = Chroma(
            persist_directory=str(VECTORDB_DIR),
            embedding_function=embeddings,
            collection_name="ia_act_en"
        )
        
        if VERBOSE:
            count = vectorstore._collection.count()
            print(f"   📂 Base chargée : {count} vecteurs")
        
        return vectorstore
    
    def retrieve(
        self,
        question: str,
        k: int = None,
        score_threshold: float = None,
        return_scores: bool = True
    ) -> List[Tuple[Document, float]] | List[Document]:
        """
        Récupère les chunks les plus pertinents pour une question.
        
        ❓ QUOI : Pipeline complet retrieval
        🎯 POURQUOI : Point d'entrée unique pour recherche
        🔧 COMMENT : Traduction (si FR) → Recherche → Filtrage
        ⚠️ ATTENTION : Question peut être FR ou EN (auto-détecté)
        
        Args:
            question (str): Question utilisateur (FR ou EN)
            k (int): Nombre de résultats (défaut: config.TOP_K_RESULTS)
            score_threshold (float): Seuil similarité min (défaut: config.SIMILARITY_THRESHOLD)
            return_scores (bool): Retourner scores avec documents
            
        Returns:
            List[Tuple[Document, float]] si return_scores=True
            List[Document] sinon
            
        Example:
            >>> retriever = IAActRetriever()
            >>> results = retriever.retrieve("Qu'est-ce qu'un système à haut risque ?")
            >>> for doc, score in results:
            ...     print(f"Page {doc.metadata['page']}: {score:.3f}")
        """
        if k is None:
            k = TOP_K_RESULTS
        
        if score_threshold is None:
            score_threshold = SIMILARITY_THRESHOLD
        
        if VERBOSE:
            print(f"\n🔍 Retrieval pour : {question}")
        
        # Étape 1 : Traduction/normalisation si nécessaire
        query_normalized = self._prepare_query(question)
        
        # Étape 2 : Query Expansion (vocabulaire citoyen → juridique)
        query_expanded = expand_citizen_query(query_normalized)
        
        # Étape 3 : Recherche vectorielle avec question optimisée
        results_with_scores = self._search_vectordb(query_expanded, k)
        
        # Étape 4 : Filtrage par score
        filtered_results = self._filter_by_score(results_with_scores, score_threshold)
        
        if VERBOSE:
            print(f"✅ {len(filtered_results)} résultats retournés (seuil: {score_threshold})")
        
        # Retourner avec ou sans scores
        if return_scores:
            return filtered_results
        else:
            return [doc for doc, score in filtered_results]
    
    def _prepare_query(self, question: str) -> str:
        """
        Prépare la question pour recherche (traduction systématique).
        
        ❓ QUOI : Traduit TOUTES les questions FR→EN
        🎯 POURQUOI : Évite faux négatifs détection + corpus EN
        🔧 COMMENT : Mistral détecte automatiquement si déjà EN
        ⚠️ ATTENTION : +500ms même pour questions EN (mais plus fiable)
        
        Args:
            question (str): Question originale (FR ou EN)
            
        Returns:
            str: Question en anglais pour recherche optimale
        """
        if not TRANSLATE_QUERY:
            if VERBOSE:
                print("   ⚠️ Traduction désactivée dans config")
            return question
        
        # Traduction/normalisation systématique (améliore précision)
        if VERBOSE:
            print(f"   🔄 Normalisation question : {question[:50]}...")
        
        query_en = translate_query_to_english(question)
        
        if VERBOSE:
            print(f"   ✅ Question optimisée : {query_en[:50]}...")
        
        return query_en
    
    def _search_vectordb(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        Effectue recherche dans ChromaDB.
        
        ❓ QUOI : Similarity search avec scores
        🎯 POURQUOI : Trouver chunks sémantiquement proches
        🔧 COMMENT : similarity_search_with_score (cosine distance)
        ⚠️ ATTENTION : Scores ChromaDB = distance (0=identique)
        
        Args:
            query (str): Question préparée (EN)
            k (int): Nombre de résultats
            
        Returns:
            List[Tuple[Document, float]]: Documents avec distances
        """
        if VERBOSE:
            print(f"   🔎 Recherche top-{k} dans ChromaDB...")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        if VERBOSE:
            print(f"   ✅ {len(results)} résultats bruts trouvés")
        
        return results
    
    def _filter_by_score(
        self,
        results: List[Tuple[Document, float]],
        threshold: float
    ) -> List[Tuple[Document, float]]:
        """
        Filtre résultats par seuil de similarité.
        
        ❓ QUOI : Éliminer résultats peu pertinents
        🎯 POURQUOI : Qualité > Quantité
        🔧 COMMENT : Convertir distance→similarité puis filtrer
        ⚠️ ATTENTION : ChromaDB distance ≠ similarité (inverser)
        
        Args:
            results (List[Tuple[Document, float]]): Résultats bruts
            threshold (float): Seuil similarité minimum (0-1)
            
        Returns:
            List[Tuple[Document, float]]: Résultats filtrés
        """
        # Convertir distances en similarités et filtrer
        filtered = []
        
        for doc, distance in results:
            similarity = 1 - distance  # ChromaDB retourne distance, pas similarité
            
            if similarity >= threshold:
                filtered.append((doc, similarity))
                
                if VERBOSE:
                    page = doc.metadata.get('page', 'N/A')
                    print(f"      ✓ Page {page}: similarité {similarity:.3f}")
            else:
                if VERBOSE:
                    page = doc.metadata.get('page', 'N/A')
                    print(f"      ✗ Page {page}: similarité {similarity:.3f} < seuil")
        
        return filtered
    
    def retrieve_with_metadata(self, question: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Récupère résultats avec métadonnées formatées.
        
        ❓ QUOI : Retrieve + formatage pour affichage/génération
        🎯 POURQUOI : Structure pratique pour generator.py
        🔧 COMMENT : Dictionnaires avec champs standardisés
        ⚠️ ATTENTION : Ajoute overhead formatage
        
        Args:
            question (str): Question utilisateur
            k (int): Nombre de résultats
            
        Returns:
            List[Dict]: Résultats formatés
                - content (str): Contenu du chunk
                - page (int): Numéro de page
                - similarity (float): Score similarité
                - metadata (dict): Métadonnées complètes
        """
        results = self.retrieve(question, k=k, return_scores=True)
        
        formatted_results = []
        for doc, similarity in results:
            formatted_results.append({
                'content': doc.page_content,
                'page': doc.metadata.get('page', -1),
                'page_label': doc.metadata.get('page_label', 'N/A'),
                'similarity': similarity,
                'metadata': doc.metadata
            })
        
        return formatted_results
    
    def retrieve_with_transformation_details(
        self, 
        question: str, 
        k: int = None
    ) -> Dict[str, Any]:
        """
        Récupère résultats + détails des transformations (transparence).
        
        ❓ QUOI : Retrieve + métadonnées pipeline complet
        🎯 POURQUOI : Transparence pour UI Streamlit (query expansion visible)
        🔧 COMMENT : Capture chaque étape transformation
        ⚠️ ATTENTION : Nécessite mode VERBOSE pour capturer transformations
        
        Args:
            question (str): Question utilisateur
            k (int): Nombre de résultats
            
        Returns:
            Dict avec:
                - results (List[Dict]): Résultats formatés
                - transformations (Dict): Étapes du pipeline
                    - original (str): Question originale
                    - normalized (str): Après traduction
                    - expanded (str): Après query expansion
        """
        if k is None:
            k = TOP_K_RESULTS
        
        # Capturer transformations
        transformations = {
            'original': question
        }
        
        # Étape 1 : Normalisation (traduction)
        from query_translator import translate_query_to_english
        normalized = translate_query_to_english(question)
        transformations['normalized'] = normalized
        
        # Étape 2 : Expansion
        expanded = expand_citizen_query(normalized)
        transformations['expanded'] = expanded
        
        # Étape 3 : Recherche avec question finale
        results_with_scores = self._search_vectordb(expanded, k)
        filtered_results = self._filter_by_score(results_with_scores, SIMILARITY_THRESHOLD)
        
        # Formater résultats
        formatted_results = []
        for doc, similarity in filtered_results:
            formatted_results.append({
                'content': doc.page_content,
                'page': doc.metadata.get('page', -1),
                'page_label': doc.metadata.get('page_label', 'N/A'),
                'similarity': similarity,
                'metadata': doc.metadata
            })
        
        return {
            'results': formatted_results,
            'transformations': transformations
        }


# === FONCTIONS UTILITAIRES ===

def create_retriever() -> IAActRetriever:
    """
    Factory function pour créer un retriever.
    
    ❓ QUOI : Point d'entrée simple pour créer retriever
    🎯 POURQUOI : Pattern factory, facilite tests
    🔧 COMMENT : Instancie IAActRetriever
    ⚠️ ATTENTION : Chargement vectorstore peut être lent
    
    Returns:
        IAActRetriever: Instance prête à l'emploi
    """
    return IAActRetriever()


def quick_search(question: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Recherche rapide one-liner.
    
    ❓ QUOI : Interface simple pour tests rapides
    🎯 POURQUOI : Éviter boilerplate pour tests
    🔧 COMMENT : Créer retriever + retrieve + formater
    ⚠️ ATTENTION : Recharge vectorstore à chaque appel (lent)
    
    Args:
        question (str): Question
        k (int): Nombre de résultats
        
    Returns:
        List[Dict]: Résultats formatés
        
    Example:
        >>> results = quick_search("What is high-risk AI?", k=3)
        >>> print(results[0]['page'])
    """
    retriever = create_retriever()
    return retriever.retrieve_with_metadata(question, k=k)


# === POINT D'ENTRÉE TEST ===

if __name__ == "__main__":
    print("=" * 70)
    print("🔍 TEST MODULE RETRIEVER - RAG IA ACT")
    print("=" * 70)
    
    # Questions de test
    test_questions = [
        ("Qu'est-ce qu'un système d'IA à haut risque ?", "fr"),
        ("What are the obligations of AI providers?", "en"),
        ("Quelles sont les sanctions en cas de non-conformité ?", "fr"),
    ]
    
    print(f"\n🧪 Test avec {len(test_questions)} questions\n")
    
    # Créer retriever (une seule fois)
    print("📂 Initialisation du retriever...")
    try:
        retriever = create_retriever()
    except Exception as e:
        print(f"❌ Erreur initialisation : {e}")
        import sys
        sys.exit(1)
    
    # Tester chaque question
    for i, (question, lang) in enumerate(test_questions, 1):
        print("\n" + "=" * 70)
        print(f"📝 Question {i}/{len(test_questions)}")
        print("=" * 70)
        print(f"Langue : {lang.upper()}")
        print(f"Question : {question}")
        
        try:
            # Retrieve avec métadonnées
            results = retriever.retrieve_with_metadata(question, k=3)
            
            print(f"\n📊 {len(results)} résultats trouvés :\n")
            
            for j, result in enumerate(results, 1):
                print(f"   {j}. Page {result['page_label']} "
                      f"(similarité: {result['similarity']:.3f})")
                print(f"      Extrait : {result['content'][:100]}...")
            
        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(test_questions):
            input("\n⏸️ Appuyez sur Entrée pour continuer...")
    
    print("\n" + "=" * 70)
    print("✅ Tests terminés")
    print("=" * 70)
