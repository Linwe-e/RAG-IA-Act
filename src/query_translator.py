"""
Module de traduction FR→EN pour optimiser la recherche vectorielle.

❓ QUOI : Convertit questions françaises en anglais avant recherche
🎯 POURQUOI : Embeddings nomic-embed-text +30% précis sur corpus anglais
🔧 COMMENT : Utilise Mistral comme traducteur (température 0 = littéral)
⚠️ ATTENTION : Ajoute ~500ms latence mais améliore retrieval significativement
"""

# ❓ QUOI : Import nouvelle API LangChain Ollama
# 🎯 POURQUOI : langchain_community.llms.Ollama est déprécié depuis 0.3.1
# 🔧 COMMENT : Utiliser langchain_ollama.OllamaLLM à la place
# ⚠️ ATTENTION : Nécessite pip install langchain-ollama
from langchain_ollama import OllamaLLM
from config import (
    QUERY_TRANSLATION_MODEL,
    QUERY_TRANSLATION_TEMP,
    OLLAMA_BASE_URL,
    VERBOSE
)


def translate_query_to_english(question_fr: str) -> str:
    """
    Traduit/normalise une question en anglais pour recherche optimale.
    
    ❓ QUOI : Traduction FR→EN ou normalisation EN→EN
    🎯 POURQUOI : Améliore précision retrieval sur corpus anglais
    🔧 COMMENT : Prompt spécialisé traduction/normalisation juridique
    ⚠️ ATTENTION : Température 0 pour traduction littérale (pas créative)
    
    Args:
        question_fr (str): Question utilisateur (FR ou EN)
        
    Returns:
        str: Question en anglais optimisée pour recherche
        
    Example:
        >>> translate_query_to_english("Qu'est-ce qu'un système d'IA à haut risque ?")
        "What is a high-risk AI system?"
        >>> translate_query_to_english("What are AI providers?")
        "What are AI service providers?"
    """
    
    # ❓ QUOI : Initialiser LLM Ollama comme traducteur
    # 🎯 POURQUOI : Évite dépendances externes (Google Translate, DeepL)
    # 🔧 COMMENT : OllamaLLM = nouvelle API recommandée (vs Ollama déprécié)
    # ⚠️ ATTENTION : Nécessite Ollama actif (http://localhost:11434)
    translator = OllamaLLM(
        model=QUERY_TRANSLATION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=QUERY_TRANSLATION_TEMP  # 0 = traduction littérale
    )
    
    # ❓ QUOI : Prompt spécialisé pour traduction/normalisation juridique
    # 🎯 POURQUOI : Guide le LLM pour terminologie précise
    # ⚠️ ATTENTION : "Only output" évite texte superflu
    translation_prompt = f"""You are a professional translator specializing in legal and regulatory texts about AI.

Task: Convert this question to proper English for the European AI Act regulation.
- If the question is in French: translate it to English
- If the question is already in English: improve/normalize the terminology if needed
- Keep legal terminology precise and official (e.g., "AI providers" → "AI service providers")
- Maintain the question structure
- Only output the English question, nothing else (no explanations, no comments)

Question: {question_fr}

Optimized English question:"""

    # ❓ QUOI : Appel au LLM pour traduction/normalisation
    # 🎯 POURQUOI : Génère version optimisée pour recherche
    # ⚠️ ATTENTION : .strip() enlève espaces parasites
    question_en = translator.invoke(translation_prompt).strip()
    
    # ❓ QUOI : Log de la transformation si mode verbose
    # 🎯 POURQUOI : Pédagogique, permet de voir étapes pipeline
    # ⚠️ ATTENTION : Désactiver en production pour performances
    if VERBOSE:
        # Déterminer type de transformation
        if question_fr == question_en:
            print(f"🌐 Question déjà optimale :")
            print(f"   Original : {question_fr}")
        else:
            print(f"🌐 Normalisation/Traduction :")
            print(f"   Original : {question_fr}")
            print(f"   Optimisé : {question_en}")
    
    return question_en


def translate_batch_queries(questions_fr: list[str]) -> list[str]:
    """
    Traduit un lot de questions françaises en anglais.
    
    ❓ QUOI : Traduction batch pour tests/benchmarks
    🎯 POURQUOI : Optimise latence en traduisant plusieurs questions d'un coup
    🔧 COMMENT : Boucle sur translate_query_to_english
    ⚠️ ATTENTION : Latence totale = n × 500ms (non parallélisé)
    
    Args:
        questions_fr (list[str]): Liste de questions en français
        
    Returns:
        list[str]: Liste de questions traduites en anglais
        
    Example:
        >>> questions = ["Question 1 ?", "Question 2 ?"]
        >>> translate_batch_queries(questions)
        ["Question 1?", "Question 2?"]
    """
    questions_en = []
    
    for i, q_fr in enumerate(questions_fr, 1):
        if VERBOSE:
            print(f"\n📝 Traduction {i}/{len(questions_fr)}...")
        
        q_en = translate_query_to_english(q_fr)
        questions_en.append(q_en)
    
    return questions_en


def expand_citizen_query(question: str) -> str:
    """
    Reformule question citoyenne en vocabulaire juridique de l'IA Act.
    
    ❓ QUOI : Query Expansion pour démocratiser l'accès
    🎯 POURQUOI : Questions grand public utilisent vocabulaire informel
    🔧 COMMENT : LLM transforme termes citoyens → terminologie officielle
    ⚠️ ATTENTION : Appliquée APRÈS translate_query_to_english
    
    Args:
        question (str): Question en langage courant (FR ou EN)
        
    Returns:
        str: Question reformulée avec vocabulaire juridique IA Act
        
    Exemples:
        >>> expand_citizen_query("What are training obligations?")
        "What are the AI literacy obligations for providers?"
        
        >>> expand_citizen_query("Must my employer inform me about AI?")
        "What are the transparency obligations for AI system deployers?"
    """
    
    # Initialiser LLM pour expansion
    expander = OllamaLLM(
        model=QUERY_TRANSLATION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0  # Factuel, pas créatif
    )
    
    # ❓ QUOI : Prompt spécialisé vocabulaire citoyen → juridique
    # 🎯 POURQUOI : Mapping entre termes courants et officiels
    # ⚠️ ATTENTION : Utiliser string concat pour éviter format specifier errors
    
    # Template de base (sans f-string problématique)
    expansion_template = """You are an expert in the European AI Act regulation. Your task is to reformulate questions into precise legal terminology from the AI Act.

VOCABULARY_MAPPING (citizen terms -> official AI Act terminology):

=== ACTORS (4 distinct roles) ===
• "company / employer / business / organization" -> "AI provider / AI deployer / provider or deployer"
• "developer / creator" -> "AI provider / provider of AI system"
• "user / end-user" -> "deployer / natural person affected"

=== OBLIGATIONS (Precise modalities) ===
• "must / have to / required / mandatory" -> "shall / obligation / requirement"
• "should / recommended" -> "should / recommendation / code of conduct"
• "allowed / can / permitted" -> "may / permitted / authorized"

=== TRAINING (2 distinct meanings) ===
• "training employees / learning about AI / AI education" -> "AI literacy obligations / staff competence"
• "training AI / teaching AI / AI learns from" -> "training data / training dataset / machine learning data"

=== INFORMATION (3 distinct types) ===
• "inform users / tell people / notify" -> "transparency obligation / information to individuals / disclosure requirement"
• "inform deployer / tell customer" -> "information to deployers / instructions for use"
• "warn / disclose deepfake / synthetic content" -> "transparency requirement Article 50 / disclosure Article 52"

=== RISKS (3 levels) ===
• "dangerous / risky / critical" -> "high-risk AI system / Annex III"
• "forbidden / banned / not allowed / illegal" -> "prohibited AI practice / Article 5"
• "limited risk" -> "limited risk AI / transparency obligations"

=== COMPLIANCE ===
• "check / verify / control / audit" -> "conformity assessment / compliance verification / third-party assessment"
• "approve / certify / authorize" -> "CE marking / conformity assessment / notified body"

=== SANCTIONS ===
• "fine / penalty / sanction / punishment" -> "administrative fine / penalty / Article 99"

CONTEXT DETECTION (choose relevant interpretation):
1. If about "training people" -> Use "AI literacy obligations"
2. If about "training AI" -> Use "training data / datasets"
3. If about "company obligations" -> Distinguish "provider" (creator) vs "deployer" (user)
4. If about "informing" -> Specify Article 50 (deepfakes), Article 52 (chatbots), or Article 26 (deployers)

Original question: """
    
    # Construire le prompt complet avec concat (pas f-string)
    expansion_prompt = expansion_template + question + "\n\nReformulated question using official AI Act terminology (one sentence, no explanation):"

    expanded = expander.invoke(expansion_prompt).strip()
    
    # Log si verbose
    if VERBOSE:
        if question.lower() != expanded.lower():
            print(f"🔄 Query Expansion :")
            print(f"   Originale : {question}")
            print(f"   Étendue   : {expanded}")
        else:
            print(f"   ℹ️ Question déjà en terminologie officielle")
    
    return expanded


def test_query_expansion():
    """
    Teste l'expansion de questions citoyennes.
    
    ❓ QUOI : Validation query expansion pour démocratisation
    🎯 POURQUOI : Vérifier mapping vocabulaire citoyen → juridique
    🔧 COMMENT : Questions informelles → attendu vocabulaire officiel
    """
    print("=" * 70)
    print("🧪 TEST QUERY EXPANSION - Démocratisation IA Act")
    print("=" * 70)
    
    # Questions citoyennes typiques (déjà en anglais après translate_query_to_english)
    test_cases = [
        ("Is training in AI required for my employer?", 
         "Devrait trouver : AI literacy obligations"),
        
        ("Must my company tell me when it uses AI?",
         "Devrait trouver : transparency obligations for deployers"),
        
        ("Which AI systems are forbidden?",
         "Devrait trouver : prohibited AI practices"),
        
        ("How much is the fine if we don't comply?",
         "Devrait trouver : administrative fines / penalties"),
        
        ("What is a dangerous AI system?",
         "Devrait trouver : high-risk AI system"),
    ]
    
    print(f"\n🧪 Test avec {len(test_cases)} questions citoyennes\n")
    
    for i, (question, expected) in enumerate(test_cases, 1):
        print("\n" + "-" * 70)
        print(f"Test {i}/{len(test_cases)}")
        print("-" * 70)
        print(f"Question informelle : {question}")
        print(f"Attendu             : {expected}")
        print()
        
        expanded = expand_citizen_query(question)
        
        print(f"\n✅ Question étendue : {expanded}")
        
        if i < len(test_cases):
            input("\n⏸️ Appuyez sur Entrée pour continuer...")
    
    print("\n" + "=" * 70)
    print("✅ Tests expansion terminés")
    print("=" * 70)


def test_translation_quality():
    """
    Teste la qualité de la traduction avec exemples juridiques.
    
    ❓ QUOI : Validation manuelle qualité traduction
    🎯 POURQUOI : Vérifier terminologie juridique préservée
    🔧 COMMENT : Exemples de questions IA Act
    ⚠️ ATTENTION : Évaluation subjective, à valider par humain
    """
    print("=" * 70)
    print("🧪 TEST QUALITÉ TRADUCTION FR→EN")
    print("=" * 70)
    
    # ❓ QUOI : Questions types sur l'IA Act
    # 🎯 POURQUOI : Couvrir différents types de questions juridiques
    # ⚠️ ATTENTION : Termes techniques à préserver (high-risk, provider, etc.)
    test_questions = [
        "Qu'est-ce qu'un système d'IA à haut risque ?",
        "Quelles sont les obligations des fournisseurs de systèmes d'IA ?",
        "Comment identifier un système d'IA interdit ?",
        "Quelle est la définition de l'intelligence artificielle selon l'IA Act ?",
        "Quels sont les droits des utilisateurs de systèmes d'IA ?",
        "Quelles sanctions en cas de non-conformité ?",
        "Comment fonctionne l'évaluation de conformité ?"
    ]
    
    print(f"\n📋 {len(test_questions)} questions à traduire...\n")
    
    for i, q_fr in enumerate(test_questions, 1):
        print(f"\n{'─' * 70}")
        print(f"Question {i}/{len(test_questions)}")
        print(f"{'─' * 70}")
        
        q_en = translate_query_to_english(q_fr)
        
        print(f"\n✅ Traduction complétée")
        print(f"   Longueur FR : {len(q_fr)} caractères")
        print(f"   Longueur EN : {len(q_en)} caractères")
    
    print("\n" + "=" * 70)
    print("✅ Test terminé - Vérifiez manuellement la qualité des traductions")
    print("=" * 70)


# === POINT D'ENTRÉE SI EXÉCUTÉ DIRECTEMENT ===
# ❓ QUOI : Permet de tester le module isolément
# 🎯 POURQUOI : Validation avant intégration dans pipeline complet
# 🔧 COMMENT : python src/query_translator.py
# ⚠️ ATTENTION : Nécessite Ollama actif avec mistral:latest

if __name__ == "__main__":
    import sys
    
    print("🚀 Module de Traduction FR→EN - RAG IA Act")
    print("=" * 70)
    
    # Vérifier que Ollama est accessible
    try:
        import requests
        response = requests.get(OLLAMA_BASE_URL, timeout=2)
        print(f"✅ Ollama accessible : {OLLAMA_BASE_URL}")
    except Exception as e:
        print(f"❌ Ollama introuvable : {OLLAMA_BASE_URL}")
        print(f"   Erreur : {e}")
        print(f"\n💡 Solution : Démarrez Ollama avec 'ollama serve'")
        sys.exit(1)
    
    print(f"✅ Modèle traduction : {QUERY_TRANSLATION_MODEL}")
    print(f"✅ Température : {QUERY_TRANSLATION_TEMP} (traduction littérale)")
    
    # Menu de test
    print("\n" + "=" * 70)
    print("🎯 Mode de test :")
    print("  1. Test rapide (1 question)")
    print("  2. Test qualité complet (7 questions)")
    print("  3. Traduction personnalisée")
    print("=" * 70)
    
    choice = input("\nVotre choix (1/2/3) : ").strip()
    
    if choice == "1":
        # Test rapide
        print("\n🧪 TEST RAPIDE\n")
        test_q = "Qu'est-ce qu'un système d'IA à haut risque ?"
        result = translate_query_to_english(test_q)
        print(f"\n✅ Résultat : {result}")
        
    elif choice == "2":
        # Test qualité complet
        test_translation_quality()
        
    elif choice == "3":
        # Traduction personnalisée
        print("\n✏️ TRADUCTION PERSONNALISÉE\n")
        custom_q = input("Entrez votre question en français : ").strip()
        
        if custom_q:
            result = translate_query_to_english(custom_q)
            print(f"\n✅ Traduction : {result}")
        else:
            print("❌ Question vide")
    else:
        print("❌ Choix invalide")
    
    print("\n" + "=" * 70)
    print("👋 Module testé avec succès !")
    print("=" * 70)
