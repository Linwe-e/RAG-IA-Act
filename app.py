"""
Interface Streamlit pour le RAG IA Act.

❓ QUOI : Application web interactive pour interroger l'IA Act
🎯 POURQUOI : Interface user-friendly pour le système RAG
🔧 COMMENT : Streamlit + Generator + Retriever
⚠️ ATTENTION : Nécessite Ollama actif et base vectorielle créée
"""

import streamlit as st
from pathlib import Path
import sys

# Ajouter src/ au path pour imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.generator import create_generator, IAActGenerator
from src.config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
    VECTORDB_DIR
)


# ========================================
# CONFIGURATION STREAMLIT
# ========================================

st.set_page_config(
    page_title="RAG IA Act - Assistant Juridique",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #f0f2f6;
        color: #333;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .stat-box {
        background-color: #e8f4f8;
        color: #1f77b4;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .answer-box {
        background-color: #f9f9f9;
        color: #333;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .question-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ========================================
# INITIALISATION SESSION STATE
# ========================================

if 'generator' not in st.session_state:
    st.session_state.generator = None
    st.session_state.initialized = False

if 'history' not in st.session_state:
    st.session_state.history = []


# ========================================
# FONCTIONS UTILITAIRES
# ========================================

@st.cache_resource
def load_generator():
    """
    Charge le générateur (mis en cache pour éviter rechargements).
    
    ❓ QUOI : Initialise generator avec retriever
    🎯 POURQUOI : Cache pour performance (1 seule init)
    🔧 COMMENT : @st.cache_resource garde en mémoire
    """
    try:
        return create_generator()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement : {str(e)}")
        return None


def check_prerequisites():
    """
    Vérifie que tous les prérequis sont OK.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    # Vérifier base vectorielle
    if not Path(VECTORDB_DIR).exists():
        return False, f"❌ Base vectorielle introuvable : {VECTORDB_DIR}\n\nLance d'abord : `python src/ingest.py`"
    
    # TODO: Vérifier Ollama (optionnel, ferait une requête)
    
    return True, "✅ Prérequis OK"


def display_source(source: dict, index: int):
    """
    Affiche une source de manière élégante.
    
    Args:
        source: Dict avec content, page, similarity
        index: Numéro de la source
    """
    page = source.get('page', 'N/A')
    similarity = source.get('similarity', 0.0)
    content = source.get('content', '')
    
    with st.container():
        st.markdown(f"""
        <div class="source-card">
            <strong>📄 Source {index} - Page {page}</strong><br>
            <small>🎯 Pertinence: {similarity:.1%}</small>
            <hr style="margin: 0.5rem 0;">
            <p style="margin: 0; font-size: 0.9rem; color: #333;">
                {content[:300]}{"..." if len(content) > 300 else ""}
            </p>
        </div>
        """, unsafe_allow_html=True)


def add_to_history(question: str, answer: str, sources: list):
    """
    Ajoute une Q&A à l'historique.
    """
    st.session_state.history.append({
        'question': question,
        'answer': answer,
        'sources': sources,
        'timestamp': st.session_state.get('timestamp_counter', 0)
    })
    st.session_state.timestamp_counter = st.session_state.get('timestamp_counter', 0) + 1


# ========================================
# SIDEBAR - CONFIGURATION & INFO
# ========================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Infos système
    with st.expander("🤖 Modèles utilisés", expanded=False):
        st.markdown(f"""
        **LLM de génération:**  
        `{LLM_MODEL}`
        
        **Embeddings:**  
        `{EMBEDDING_MODEL}`
        
        **Seuil de similarité:**  
        `{SIMILARITY_THRESHOLD}`
        
        **Nombre de chunks:**  
        `{TOP_K_RESULTS}`
        """)
    
    # Statistiques
    if st.session_state.history:
        st.markdown("### 📊 Statistiques")
        st.metric("Questions posées", len(st.session_state.history))
        
        # Moyenne de sources par question
        avg_sources = sum(len(h['sources']) for h in st.session_state.history) / len(st.session_state.history)
        st.metric("Sources moy. par question", f"{avg_sources:.1f}")
    
    # Actions
    st.markdown("### 🔧 Actions")
    
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    if st.button("🔄 Recharger le système", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.generator = None
        st.session_state.initialized = False
        st.rerun()
    
    # Informations
    st.markdown("---")
    st.markdown("""
    ### ℹ️ À propos
    
    **RAG IA Act** est un assistant juridique basé sur le Règlement européen sur l'Intelligence Artificielle (UE 2024/1689).
    
    **Fonctionnalités:**
    - 🔍 Recherche sémantique dans le règlement
    - 🌐 Questions en français ou anglais
    - 📚 Citations des articles sources
    - ⚡ 100% local (Ollama)
    
    **Stack technique:**
    - LangChain + ChromaDB
    - Mistral 7B (local)
    - Streamlit
    """)


# ========================================
# HEADER PRINCIPAL
# ========================================

st.markdown('<div class="main-header">⚖️ Assistant Juridique IA Act</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interrogez le Règlement européen sur l\'Intelligence Artificielle (UE 2024/1689)</div>',
    unsafe_allow_html=True
)


# ========================================
# VÉRIFICATION & INITIALISATION
# ========================================

# Vérifier prérequis
prereq_ok, prereq_msg = check_prerequisites()

if not prereq_ok:
    st.error(prereq_msg)
    st.stop()

# Initialiser générateur
if not st.session_state.initialized:
    with st.spinner("🔧 Initialisation du système RAG..."):
        generator = load_generator()
        
        if generator is None:
            st.error("❌ Impossible d'initialiser le système. Vérifie que :\n- Ollama est lancé (`ollama serve`)\n- Les modèles sont installés\n- La base vectorielle existe")
            st.stop()
        
        st.session_state.generator = generator
        st.session_state.initialized = True
        st.success("✅ Système RAG prêt !")


# ========================================
# INTERFACE PRINCIPALE - QUESTION
# ========================================

st.markdown("### 💬 Posez votre question")

# Exemples de questions
with st.expander("💡 Exemples de questions", expanded=False):
    example_questions = [
        "Qu'est-ce qu'un système d'IA à haut risque ?",
        "Quelles sont les obligations des fournisseurs d'IA ?",
        "Quels sont les systèmes d'IA interdits ?",
        "Quelles sont les sanctions en cas de non-conformité ?",
        "Qu'est-ce qu'un modèle d'IA à usage général ?",
        "Quels sont les droits des personnes concernées ?",
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"📌 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.current_question = example

# Input question
question = st.text_input(
    "Votre question :",
    value=st.session_state.get('current_question', ''),
    placeholder="Ex: Qu'est-ce qu'un système d'IA à haut risque ?",
    help="Posez votre question en français ou en anglais",
    key="question_input"
)

# Options avancées
with st.expander("🔧 Options avancées", expanded=False):
    show_sources = st.checkbox("Afficher les sources", value=True, help="Affiche les extraits du règlement utilisés pour générer la réponse")
    show_debug = st.checkbox("Mode debug", value=False, help="Affiche des informations techniques détaillées")

# Bouton recherche
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_button = st.button("🔍 Rechercher", type="primary", use_container_width=True)


# ========================================
# TRAITEMENT & AFFICHAGE RÉPONSE
# ========================================

if search_button and question.strip():
    
    # Afficher la question
    st.markdown(f"""
    <div class="question-box">
        <strong>❓ Votre question :</strong><br>
        {question}
    </div>
    """, unsafe_allow_html=True)
    
    # Génération
    with st.spinner("🤔 Recherche dans le règlement et génération de la réponse..."):
        try:
            result = st.session_state.generator.generate(
                question, 
                return_sources=True,
                return_transformations=True  # ⬅️ NOUVEAU : Capturer transformations
            )
            
            # ⬇️ NOUVEAU : Afficher transformations query (transparence)
            if result.get('transformations'):
                trans = result['transformations']
                
                with st.expander("🔍 Transparence : Optimisation de la question", expanded=False):
                    st.markdown("**Pipeline de transformation appliqué à votre question :**")
                    
                    # Étape 1 : Question originale
                    st.markdown(f"""
                    **1️⃣ Question originale :**  
                    `{trans['original']}`
                    """)
                    
                    # Étape 2 : Normalisation/Traduction
                    if trans['normalized'] != trans['original']:
                        st.markdown(f"""
                        **2️⃣ Normalisation/Traduction :**  
                        `{trans['normalized']}`  
                        <small style="color: #666;">→ Traduction en anglais pour recherche optimale</small>
                        """, unsafe_allow_html=True)
                    
                    # Étape 3 : Expansion (vocabulaire juridique)
                    if trans['expanded'] != trans['normalized']:
                        st.markdown(f"""
                        **3️⃣ Expansion juridique :**  
                        `{trans['expanded']}`  
                        <small style="color: #666;">→ Vocabulaire citoyen transformé en terminologie officielle IA Act</small>
                        """, unsafe_allow_html=True)
                        
                        # Highlight des transformations clés
                        st.info("💡 **Transformation appliquée :** Votre langage courant a été converti en vocabulaire juridique pour améliorer la précision de la recherche.")
                    else:
                        st.success("✅ Votre question utilise déjà la terminologie officielle de l'IA Act.")
            
            # Afficher la réponse
            st.markdown("### 💡 Réponse")
            st.markdown(f"""
            <div class="answer-box">
                {result['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Ajouter à l'historique
            add_to_history(question, result['answer'], result.get('sources', []))
            
            # Afficher les sources
            if show_sources and result.get('has_sources') and result.get('sources'):
                st.markdown("### 📚 Sources utilisées")
                st.markdown(f"*{len(result['sources'])} extrait(s) du règlement*")
                
                for i, source in enumerate(result['sources'], 1):
                    display_source(source, i)
            
            elif not result.get('has_sources'):
                st.warning("⚠️ Aucune source pertinente trouvée pour cette question.")
            
            # Mode debug
            if show_debug:
                st.markdown("### 🐛 Informations de debug")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="stat-box">
                        <strong>{len(result.get('sources', []))}</strong><br>
                        <small>Sources trouvées</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if result.get('sources'):
                        avg_sim = sum(s['similarity'] for s in result['sources']) / len(result['sources'])
                        st.markdown(f"""
                        <div class="stat-box">
                            <strong>{avg_sim:.1%}</strong><br>
                            <small>Similarité moyenne</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="stat-box">
                        <strong>{len(result['answer'])} car.</strong><br>
                        <small>Longueur réponse</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("🔍 Détails techniques", expanded=False):
                    st.json({
                        'question': question,
                        'num_sources': len(result.get('sources', [])),
                        'has_sources': result.get('has_sources'),
                        'answer_length': len(result['answer']),
                        'sources_pages': [s.get('page') for s in result.get('sources', [])],
                        'sources_similarities': [f"{s.get('similarity', 0):.3f}" for s in result.get('sources', [])]
                    })
        
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération : {str(e)}")
            if show_debug:
                st.exception(e)

elif search_button:
    st.warning("⚠️ Veuillez entrer une question.")


# ========================================
# HISTORIQUE
# ========================================

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Historique des questions")
    
    # Afficher les 5 dernières questions
    for i, entry in enumerate(reversed(st.session_state.history[-5:])):
        with st.expander(f"❓ {entry['question'][:80]}...", expanded=False):
            st.markdown(f"**Réponse :**\n\n{entry['answer']}")
            
            if entry.get('sources'):
                st.markdown(f"\n**Sources :** {len(entry['sources'])} extrait(s)")


# ========================================
# FOOTER
# ========================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>
        💡 <strong>Conseil :</strong> Pour de meilleurs résultats, posez des questions précises sur des aspects spécifiques du règlement.<br>
        ⚠️ <strong>Avertissement :</strong> Cette application est à but pédagogique. Pour des conseils juridiques officiels, consultez un avocat.
    </p>
    <p style="margin-top: 1rem;">
        Développé avec ❤️ | LangChain + ChromaDB + Mistral + Streamlit | 🇪🇺 Règlement UE 2024/1689
    </p>
</div>
""", unsafe_allow_html=True)
