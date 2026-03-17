"""
Module de génération de réponses pour le RAG IA Act.

❓ QUOI : Génère réponses en français à partir de chunks anglais
🎯 POURQUOI : Combiner retrieval + LLM pour réponses contextualisées
🔧 COMMENT : Chunks EN → Prompt → Mistral → Réponse FR
⚠️ ATTENTION : Nécessite Ollama avec Mistral installé
"""

from typing import List, Dict, Any, Optional
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    SYSTEM_PROMPT_TEMPLATE,
    VERBOSE
)
from retriever import IAActRetriever, create_retriever


# ❓ QUOI : Template de prompt pour génération
# 🎯 POURQUOI : Guider le LLM pour réponses juridiques précises
# 🔧 COMMENT : Inject contexte + question dans template structuré
# ⚠️ ATTENTION : Le contexte est en anglais, la réponse doit être en français
ANSWER_PROMPT_TEMPLATE = """Tu es un assistant juridique spécialisé dans le Règlement européen sur l'Intelligence Artificielle (IA Act - Règlement UE 2024/1689).

CONTEXTE EXTRAIT DU RÈGLEMENT (en anglais) :
{context}

QUESTION DE L'UTILISATEUR :
{question}

INSTRUCTIONS :
1. Réponds UNIQUEMENT en te basant sur le contexte fourni ci-dessus
2. Réponds en FRANÇAIS, même si le contexte est en anglais
3. Cite les articles pertinents entre crochets [Article X] quand possible
4. Si l'information n'est pas dans le contexte, dis clairement "Je ne trouve pas cette information dans les extraits fournis"
5. Structure ta réponse de manière claire avec des paragraphes si nécessaire
6. Reste factuel et précis - c'est un texte juridique

RÉPONSE :"""


class IAActGenerator:
    """
    ❓ QUOI : Générateur de réponses RAG pour l'IA Act
    🎯 POURQUOI : Orchestrer retrieval + génération avec contexte
    🔧 COMMENT : 
        1. Récupère chunks via retriever
        2. Formate contexte pour prompt
        3. Appelle Mistral pour génération
        4. Retourne réponse structurée
    ⚠️ ATTENTION : Gère cas où aucun chunk pertinent trouvé
    """
    
    def __init__(self, retriever: Optional[IAActRetriever] = None):
        """
        Initialise le générateur avec retriever et LLM.
        
        Args:
            retriever: Instance de IAActRetriever (créé auto si None)
        """
        if VERBOSE:
            print("🔧 Initialisation du generator...")
        
        # Retriever pour chercher chunks
        self.retriever = retriever or create_retriever()
        
        # ❓ QUOI : LLM Mistral via Ollama
        # 🎯 POURQUOI : Génération bilingue (comprend EN, répond FR)
        # 🔧 COMMENT : temperature=0.1 pour réponses factuelles
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=OLLAMA_BASE_URL
        )
        
        # Template de prompt
        self.prompt = PromptTemplate(
            template=ANSWER_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        if VERBOSE:
            print(f"   🤖 LLM : {LLM_MODEL} (temp={LLM_TEMPERATURE})")
            print("✅ Generator prêt")
    
    def generate(
        self, 
        question: str,
        return_sources: bool = True,
        return_transformations: bool = False
    ) -> Dict[str, Any]:
        """
        Génère une réponse complète à partir de la question.
        
        ❓ QUOI : Pipeline complet question → réponse
        🎯 POURQUOI : Interface simple pour obtenir réponse + sources
        🔧 COMMENT :
            1. Retrieval des chunks pertinents
            2. Construction du contexte
            3. Génération via LLM
            4. Packaging réponse + métadonnées
        
        Args:
            question: Question de l'utilisateur (FR ou EN)
            return_sources: Inclure les sources dans la réponse
            return_transformations: Inclure détails transformations query
        
        Returns:
            Dict avec:
                - answer: Réponse générée (str)
                - sources: Liste des chunks sources (List[Dict])
                - question: Question originale (str)
                - has_sources: Booléen si des sources trouvées
                - transformations: Étapes transformation (si return_transformations=True)
        
        ⚠️ ATTENTION : Si aucun chunk trouvé, génère message approprié
        """
        if VERBOSE:
            print(f"\n🤖 Génération pour : {question[:80]}...")
        
        # 1. Retrieval des chunks pertinents (avec ou sans transformations)
        if return_transformations:
            retrieval_result = self.retriever.retrieve_with_transformation_details(question)
            retrieved_chunks = retrieval_result['results']
            transformations = retrieval_result['transformations']
        else:
            retrieved_chunks = self.retriever.retrieve_with_metadata(question)
            transformations = None
        
        if not retrieved_chunks:
            if VERBOSE:
                print("⚠️ Aucun chunk pertinent trouvé")
            return {
                "answer": "Je n'ai pas trouvé d'information pertinente dans le règlement IA Act pour répondre à cette question. Pourriez-vous reformuler ou poser une question plus spécifique sur le règlement européen sur l'IA ?",
                "sources": [],
                "question": question,
                "has_sources": False
            }
        
        if VERBOSE:
            print(f"   📚 {len(retrieved_chunks)} chunks trouvés")
        
        # 2. Construction du contexte à partir des chunks
        context = self._format_context(retrieved_chunks)
        
        if VERBOSE:
            print(f"   📝 Contexte : {len(context)} caractères")
        
        # 3. Génération de la réponse
        prompt_text = self.prompt.format(
            context=context,
            question=question
        )
        
        if VERBOSE:
            print("   🎯 Génération de la réponse...")
        
        answer = self.llm.invoke(prompt_text)
        
        if VERBOSE:
            print(f"   ✅ Réponse générée : {len(answer)} caractères")
        
        # 4. Packaging de la réponse
        result = {
            "answer": answer.strip(),
            "question": question,
            "has_sources": True
        }
        
        if return_sources:
            result["sources"] = retrieved_chunks
        
        if return_transformations and transformations:
            result["transformations"] = transformations
        
        return result
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formate les chunks récupérés en contexte pour le prompt.
        
        ❓ QUOI : Transforme liste de chunks en texte structuré
        🎯 POURQUOI : LLM a besoin de contexte bien formaté
        🔧 COMMENT : Ajoute numéros, pages, séparateurs
        
        Args:
            chunks: Liste de dicts avec content, page, similarity
        
        Returns:
            Contexte formaté pour injection dans prompt
        
        ⚠️ ATTENTION : Garde infos de page pour citations
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # ❓ QUOI : Format "EXTRAIT X (Page Y)"
            # 🎯 POURQUOI : Permet au LLM de citer les sources
            page = chunk.get('page', 'N/A')
            content = chunk.get('content', '')
            similarity = chunk.get('similarity', 0.0)
            
            part = f"""EXTRAIT {i} (Page {page}, Pertinence: {similarity:.2f}) :
{content}

---"""
            context_parts.append(part)
        
        return "\n".join(context_parts)
    
    def generate_with_details(self, question: str) -> Dict[str, Any]:
        """
        Génère une réponse avec détails complets pour debugging.
        
        ❓ QUOI : Version verbose de generate()
        🎯 POURQUOI : Utile pour debugging et pédagogie
        🔧 COMMENT : Retourne toutes les métadonnées intermédiaires
        
        Args:
            question: Question utilisateur
        
        Returns:
            Dict avec answer, sources, context, prompt, stats
        """
        # Récupération
        retrieved_chunks = self.retriever.retrieve_with_metadata(question)
        
        # Construction contexte
        context = self._format_context(retrieved_chunks) if retrieved_chunks else ""
        
        # Génération
        if retrieved_chunks:
            prompt_text = self.prompt.format(context=context, question=question)
            answer = self.llm.invoke(prompt_text)
        else:
            prompt_text = ""
            answer = "Aucune information pertinente trouvée."
        
        # Stats
        stats = {
            "num_chunks": len(retrieved_chunks),
            "context_length": len(context),
            "prompt_length": len(prompt_text),
            "answer_length": len(answer),
            "avg_similarity": sum(c.get('similarity', 0) for c in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0
        }
        
        return {
            "answer": answer.strip(),
            "sources": retrieved_chunks,
            "context": context,
            "prompt": prompt_text,
            "question": question,
            "stats": stats
        }


# ========================================
# FONCTIONS UTILITAIRES
# ========================================

def create_generator(retriever: Optional[IAActRetriever] = None) -> IAActGenerator:
    """
    Factory function pour créer un générateur.
    
    ❓ QUOI : Crée instance de IAActGenerator
    🎯 POURQUOI : Interface simple pour initialisation
    🔧 COMMENT : Appelle constructeur avec paramètres
    
    Args:
        retriever: Retriever optionnel (créé auto si None)
    
    Returns:
        Instance configurée de IAActGenerator
    """
    return IAActGenerator(retriever=retriever)


def quick_answer(question: str, verbose: bool = False) -> str:
    """
    Génère une réponse rapide sans métadonnées.
    
    ❓ QUOI : Interface one-liner pour obtenir réponse
    🎯 POURQUOI : Utilisation simple dans scripts/tests
    🔧 COMMENT : Crée generator, appelle generate, retourne answer
    
    Args:
        question: Question utilisateur
        verbose: Afficher logs de progression
    
    Returns:
        Réponse générée (string)
    
    Example:
        >>> answer = quick_answer("Qu'est-ce qu'un système IA à haut risque ?")
        >>> print(answer)
    """
    # Temporairement override VERBOSE si nécessaire
    import config
    old_verbose = config.VERBOSE
    if verbose:
        config.VERBOSE = True
    
    try:
        generator = create_generator()
        result = generator.generate(question, return_sources=False)
        return result["answer"]
    finally:
        config.VERBOSE = old_verbose


def display_answer(result: Dict[str, Any], show_sources: bool = True):
    """
    Affiche une réponse formatée dans le terminal.
    
    ❓ QUOI : Pretty-print de la réponse RAG
    🎯 POURQUOI : Lisibilité dans tests/CLI
    🔧 COMMENT : Formate avec emojis et séparateurs
    
    Args:
        result: Dict retourné par generate()
        show_sources: Afficher les sources utilisées
    """
    print("\n" + "="*70)
    print("❓ QUESTION")
    print("="*70)
    print(result["question"])
    
    print("\n" + "="*70)
    print("💬 RÉPONSE")
    print("="*70)
    print(result["answer"])
    
    if show_sources and result.get("has_sources") and result.get("sources"):
        print("\n" + "="*70)
        print(f"📚 SOURCES ({len(result['sources'])} extraits)")
        print("="*70)
        for i, source in enumerate(result["sources"], 1):
            page = source.get('page', 'N/A')
            similarity = source.get('similarity', 0.0)
            content = source.get('content', '')[:200] + "..."
            
            print(f"\n{i}. Page {page} (similarité: {similarity:.3f})")
            print(f"   {content}")
    
    print("\n" + "="*70)


# ========================================
# TESTS
# ========================================

if __name__ == "__main__":
    """
    Tests du module generator avec questions variées.
    """
    print("="*70)
    print("🤖 TEST MODULE GENERATOR - RAG IA ACT")
    print("="*70)
    
    # Questions de test
    test_questions = [
        "Qu'est-ce qu'un système d'IA à haut risque selon le règlement ?",
        "Quelles sont les obligations des fournisseurs d'IA ?",
        "What are the penalties for non-compliance?"  # Test EN aussi
    ]
    
    print(f"\n🧪 Test avec {len(test_questions)} questions\n")
    
    # Créer générateur une fois
    print("📂 Initialisation du generator...")
    generator = create_generator()
    
    # Tester chaque question
    for i, question in enumerate(test_questions, 1):
        print("\n" + "="*70)
        print(f"📝 Question {i}/{len(test_questions)}")
        print("="*70)
        
        # Génération
        result = generator.generate(question, return_sources=True)
        
        # Affichage
        display_answer(result, show_sources=True)
        
        # Pause entre questions
        if i < len(test_questions):
            input("\n⏸️  Appuyez sur Entrée pour continuer...")
    
    print("\n" + "="*70)
    print("✅ Tests terminés")
    print("="*70)
    
    # Test fonction quick_answer
    print("\n" + "="*70)
    print("🚀 Test quick_answer()")
    print("="*70)
    
    quick_q = "Quels sont les systèmes d'IA interdits ?"
    print(f"Question : {quick_q}\n")
    answer = quick_answer(quick_q, verbose=True)
    print(f"\nRéponse : {answer}")
