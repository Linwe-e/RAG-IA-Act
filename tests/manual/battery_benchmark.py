"""
🎯 Batterie de Tests RAG - Certification Qualité
Exécute les 33 questions de test et génère un rapport détaillé.

❓ QUOI : Script automatique pour valider qualité RAG
🎯 POURQUOI : Certification système avec questions de tous niveaux
🔧 COMMENT : Utilise generator.py pour capturer articles, scores, réponses
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Fix encodage Windows pour emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Ajouter src/ au path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from generator import IAActGenerator
from config import SIMILARITY_THRESHOLD, VERBOSE

# Désactiver verbose pour logs propres
import config
config.VERBOSE = False


# ========== DÉFINITION DES QUESTIONS ==========

TEST_QUESTIONS = {
    # NIVEAU 1 : Questions Directes (Vocabulaire Juridique)
    "NIVEAU_1": [
        {
            "id": "Q1",
            "question": "What are the obligations of AI providers under Article 16?",
            "expected_article": "16",
            "min_score": 0.60,
            "category": "Acteurs & Rôles"
        },
        {
            "id": "Q2",
            "question": "What must AI deployers do according to Article 26?",
            "expected_article": "26",
            "min_score": 0.55,
            "category": "Acteurs & Rôles"
        },
        {
            "id": "Q3",
            "question": "What is the difference between a provider and a deployer?",
            "expected_article": "3",  # Définitions
            "min_score": 0.45,
            "category": "Acteurs & Rôles"
        },
        {
            "id": "Q4",
            "question": "Which AI practices are prohibited by the AI Act?",
            "expected_article": "5",
            "min_score": 0.65,
            "category": "Systèmes Interdits"
        },
        {
            "id": "Q5",
            "question": "Is social scoring by public authorities allowed?",
            "expected_article": "5",
            "min_score": 0.55,
            "category": "Systèmes Interdits"
        },
        {
            "id": "Q6",
            "question": "Can AI manipulate human behavior through subliminal techniques?",
            "expected_article": "5",
            "min_score": 0.50,
            "category": "Systèmes Interdits"
        },
        {
            "id": "Q7",
            "question": "What makes an AI system high-risk?",
            "expected_article": "6",
            "min_score": 0.60,
            "category": "Haut Risque"
        },
        {
            "id": "Q8",
            "question": "What are the conformity assessment obligations for high-risk AI?",
            "expected_article": "43",
            "min_score": 0.55,
            "category": "Haut Risque"
        },
        {
            "id": "Q9",
            "question": "Must high-risk AI systems have human oversight?",
            "expected_article": "14",
            "min_score": 0.60,
            "category": "Haut Risque"
        },
    ],
    
    # NIVEAU 2 : Questions Citoyennes (Langage Naturel)
    "NIVEAU_2": [
        {
            "id": "Q10",
            "question": "Mon patron doit-il me former à l'intelligence artificielle ?",
            "expected_article": "4",  # AI literacy
            "min_score": 0.40,
            "category": "Obligations Employeur"
        },
        {
            "id": "Q11",
            "question": "My company uses AI for hiring. What are their legal obligations?",
            "expected_article": "6",  # High-risk employment
            "min_score": 0.45,
            "category": "Obligations Employeur"
        },
        {
            "id": "Q12",
            "question": "Can my employer use AI to monitor my work performance?",
            "expected_article": "6",  # Annex III point 4
            "min_score": 0.40,
            "category": "Obligations Employeur"
        },
        {
            "id": "Q13",
            "question": "Ai-je le droit de savoir quand une IA prend une décision me concernant ?",
            "expected_article": "26",  # Transparency
            "min_score": 0.35,
            "category": "Droits des Personnes"
        },
        {
            "id": "Q14",
            "question": "Can I refuse to interact with an AI system at work?",
            "expected_article": "26",  # Human oversight
            "min_score": 0.35,
            "category": "Droits des Personnes"
        },
        {
            "id": "Q15",
            "question": "Puis-je demander une explication sur une décision prise par une IA ?",
            "expected_article": "86",  # Right to explanation
            "min_score": 0.40,
            "category": "Droits des Personnes"
        },
        {
            "id": "Q16",
            "question": "Est-ce qu'un chatbot doit dire qu'il est une IA ?",
            "expected_article": "52",  # Transparency chatbots
            "min_score": 0.45,
            "category": "Cas d'Usage"
        },
        {
            "id": "Q17",
            "question": "My doctor uses AI to diagnose illnesses. Is this allowed?",
            "expected_article": "6",  # Medical devices
            "min_score": 0.40,
            "category": "Cas d'Usage"
        },
        {
            "id": "Q18",
            "question": "Can police use facial recognition in public spaces?",
            "expected_article": "5",  # Prohibited + exceptions
            "min_score": 0.45,
            "category": "Cas d'Usage"
        },
    ],
    
    # NIVEAU 3 : Questions Pièges (Concepts Complexes)
    "NIVEAU_3": [
        {
            "id": "Q19",
            "question": "What training data requirements exist for AI systems?",
            "expected_article": "10",  # Training data (PAS Article 4)
            "min_score": 0.40,
            "category": "Ambiguïtés",
            "piege": "training = data, pas literacy"
        },
        {
            "id": "Q20",
            "question": "Who must train AI systems to ensure safety?",
            "expected_article": "9",  # Risk management
            "min_score": 0.35,
            "category": "Ambiguïtés",
            "piege": "train systems = développement"
        },
        {
            "id": "Q21",
            "question": "My company trains employees on AI. What's required?",
            "expected_article": "4",  # AI literacy ET Article 26
            "min_score": 0.35,
            "category": "Ambiguïtés",
            "piege": "Distinguer literacy vs deployer"
        },
        {
            "id": "Q22",
            "question": "What are ALL the steps to put a high-risk AI on the EU market?",
            "expected_article": "16",  # Multi-articles nécessaires
            "min_score": 0.40,
            "category": "Multi-Articles",
            "piege": "Nécessite 3+ articles"
        },
        {
            "id": "Q23",
            "question": "What happens if a company violates the AI Act?",
            "expected_article": "99",  # Fines
            "min_score": 0.50,
            "category": "Multi-Articles"
        },
        {
            "id": "Q24",
            "question": "Can I develop AI in a regulatory sandbox and what are the conditions?",
            "expected_article": "57",  # Sandbox
            "min_score": 0.45,
            "category": "Multi-Articles"
        },
        {
            "id": "Q25",
            "question": "Does the AI Act apply to open-source AI models?",
            "expected_article": "2",  # Scope
            "min_score": 0.35,
            "category": "Edge Cases",
            "piege": "Concept dans Recitals"
        },
        {
            "id": "Q26",
            "question": "What if an AI system from the US is used in France?",
            "expected_article": "2",  # Territorial scope
            "min_score": 0.40,
            "category": "Edge Cases"
        },
        {
            "id": "Q27",
            "question": "When does the AI Act enter into force?",
            "expected_article": "113",  # Entry into force
            "min_score": 0.55,
            "category": "Edge Cases"
        },
    ],
    
    # NIVEAU 4 : Stress Test (Questions "Impossibles")
    "NIVEAU_4": [
        {
            "id": "Q28",
            "question": "How do I build a neural network in Python?",
            "expected_article": None,  # Hors périmètre
            "min_score": 0.30,
            "category": "Hors Périmètre",
            "should_fail": True
        },
        {
            "id": "Q29",
            "question": "What's the GDPR regulation about personal data?",
            "expected_article": None,  # Confusion possible
            "min_score": 0.30,
            "category": "Hors Périmètre",
            "should_fail": True
        },
        {
            "id": "Q30",
            "question": "Combien coûte une formation en IA ?",
            "expected_article": None,
            "min_score": 0.30,
            "category": "Hors Périmètre",
            "should_fail": True
        },
        {
            "id": "Q31",
            "question": "Tell me about AI.",
            "expected_article": "3",  # Trop vague
            "min_score": 0.40,
            "category": "Questions Vagues"
        },
        {
            "id": "Q32",
            "question": "Is AI dangerous?",
            "expected_article": "5",  # Ambiguë
            "min_score": 0.35,
            "category": "Questions Vagues"
        },
        {
            "id": "Q33",
            "question": "What's the AI Act?",
            "expected_article": "1",  # Subject matter
            "min_score": 0.50,
            "category": "Questions Vagues"
        },
    ]
}


# ========== FONCTIONS DE TEST ==========

def extract_articles_from_sources(sources: List[Dict]) -> List[str]:
    """
    Extrait les numéros d'articles des métadonnées sources.
    Format sources : [{'content': str, 'page': int, 'similarity': float, 'metadata': dict}, ...]
    
    ⚠️ ATTENTION : Le PDF a des espaces dans les mots : "Ar ticle" au lieu de "Article"
    Regex accepte espaces optionnels entre lettres
    """
    articles = []
    import re
    
    for source in sources:
        # Chercher dans le contenu du chunk
        content = source.get('content', '')
        
        # Regex permissif : A r t i c l e (espaces optionnels) + numéro
        # Accepte : "Article 16", "Ar ticle 16", "Art. 16", "Art 16"
        matches = re.findall(
            r'(?i)A\s*r\s*t\s*(?:i\s*c\s*l\s*e|\.?)\s+(\d+)', 
            content
        )
        articles.extend(matches)
    
    return list(set(articles))  # Unique


def get_best_score_from_sources(sources: List[Dict]) -> float:
    """
    Récupère le meilleur score de similarité des sources.
    """
    if not sources:
        return 0.0
    return max([s.get('similarity', 0.0) for s in sources])


def test_question(generator: IAActGenerator, test_case: Dict) -> Dict:
    """
    Teste une question et retourne les résultats détaillés.
    """
    question = test_case['question']
    expected_article = test_case.get('expected_article')
    min_score = test_case.get('min_score', 0.30)
    should_fail = test_case.get('should_fail', False)
    
    print(f"\n{'='*70}")
    print(f"🧪 {test_case['id']}: {question}")
    print(f"   Catégorie: {test_case['category']}")
    if test_case.get('piege'):
        print(f"   ⚠️ Piège: {test_case['piege']}")
    print(f"   Attendu: Article {expected_article if expected_article else 'Aucun'} (score > {min_score})")
    
    try:
        # Générer réponse
        result = generator.generate(question, return_sources=True, return_transformations=True)
        
        # Extraire données
        answer = result.get('answer', '')
        sources = result.get('sources', [])
        transformations = result.get('transformations', {})
        
        # Calculer meilleur score (corrigé)
        best_score = get_best_score_from_sources(sources)
        
        # Extraire articles trouvés
        articles_found = extract_articles_from_sources(sources)
        
        # Vérifier si article attendu trouvé
        article_match = expected_article in articles_found if expected_article else None
        
        # Évaluer pertinence
        if should_fail:
            # Pour questions hors périmètre, succès = score faible OU message explicite
            is_success = (best_score < min_score) or ("hors" in answer.lower()) or ("pas" in answer.lower())
            status = "✅" if is_success else "❌"
        else:
            # Pour questions normales, succès = score ET article trouvé
            is_success = (best_score >= min_score) and (article_match if expected_article else True)
            status = "✅" if is_success else "⚠️" if best_score >= (min_score - 0.05) else "❌"
        
        # Affichage résultats
        print(f"\n   {status} Score: {best_score:.3f} (seuil: {min_score})")
        print(f"   📄 Articles trouvés: {', '.join(articles_found) if articles_found else 'Aucun'}")
        
        if transformations:
            print(f"   🔄 Transformation:")
            print(f"      Original: {transformations.get('original', 'N/A')[:60]}...")
            if transformations.get('expanded') != transformations.get('normalized'):
                print(f"      Enrichie: {transformations.get('expanded', 'N/A')[:60]}...")
        
        # Vérifier citations
        has_citation = any(f"[Article {a}]" in answer or f"article {a}" in answer.lower() 
                          for a in articles_found)
        print(f"   💬 Citations: {'Oui' if has_citation else 'Non'}")
        
        return {
            "id": test_case['id'],
            "question": question,
            "category": test_case['category'],
            "expected_article": expected_article,
            "articles_found": articles_found,
            "article_match": article_match,
            "best_score": best_score,
            "min_score": min_score,
            "has_citation": has_citation,
            "is_success": is_success,
            "status": status,
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
            "transformations": transformations
        }
        
    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        return {
            "id": test_case['id'],
            "question": question,
            "error": str(e),
            "is_success": False,
            "status": "❌"
        }


def run_test_battery():
    """
    Exécute la batterie complète de tests et génère le rapport.
    """
    print("="*70)
    print("🎯 BATTERIE DE TESTS RAG - CERTIFICATION QUALITÉ")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️ Seuil similarité: {SIMILARITY_THRESHOLD}")
    print(f"📊 Total questions: 33 (4 niveaux)")
    
    # Initialiser generator
    print("\n🔧 Initialisation du generator...")
    generator = IAActGenerator()
    print("✅ Generator prêt\n")
    
    # Résultats par niveau
    results_by_level = {}
    all_results = []
    
    # Exécuter tests par niveau
    for level_name, questions in TEST_QUESTIONS.items():
        print(f"\n{'#'*70}")
        print(f"{'#'*70}")
        print(f"## {level_name} ({len(questions)} questions)")
        print(f"{'#'*70}")
        
        level_results = []
        for test_case in questions:
            result = test_question(generator, test_case)
            level_results.append(result)
            all_results.append(result)
        
        results_by_level[level_name] = level_results
    
    # Générer rapport final
    print("\n" + "="*70)
    print("📊 RAPPORT FINAL")
    print("="*70)
    
    total_success = sum(1 for r in all_results if r.get('is_success'))
    total_questions = len(all_results)
    success_rate = (total_success / total_questions) * 100
    
    print(f"\n🎯 SCORE GLOBAL: {total_success}/{total_questions} ({success_rate:.1f}%)")
    
    # Score par niveau
    for level_name, results in results_by_level.items():
        level_success = sum(1 for r in results if r.get('is_success'))
        level_total = len(results)
        level_rate = (level_success / level_total) * 100
        
        status_icon = "🥇" if level_rate >= 85 else "🥈" if level_rate >= 70 else "🥉" if level_rate >= 60 else "⚠️"
        print(f"\n{status_icon} {level_name}: {level_success}/{level_total} ({level_rate:.1f}%)")
    
    # Certification
    print(f"\n{'='*70}")
    print("🏆 CERTIFICATION QUALITÉ")
    print(f"{'='*70}")
    
    if success_rate >= 85:
        certification = "🥇 GOLD"
        message = "Excellent ! Système production-ready."
    elif success_rate >= 73:
        certification = "🥈 SILVER"
        message = "Très bon ! Quelques optimisations possibles."
    elif success_rate >= 61:
        certification = "🥉 BRONZE"
        message = "Correct. Amélioration recommandée."
    else:
        certification = "⚠️ À AMÉLIORER"
        message = "Score insuffisant. Révision nécessaire."
    
    print(f"\n{certification}")
    print(f"{message}")
    
    # Sauvegarder rapport détaillé
    save_detailed_report(all_results, results_by_level, success_rate, certification)
    
    return all_results, success_rate, certification


def save_detailed_report(all_results: List[Dict], results_by_level: Dict, 
                        success_rate: float, certification: str):
    """
    Sauvegarde un rapport détaillé en markdown.
    """
    report_path = Path("docs/test-report.md")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 📊 Rapport de Tests RAG - Certification Qualité\n\n")
        f.write(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Score Global** : {certification} - {success_rate:.1f}%\n\n")
        f.write(f"**Seuil Similarité** : {SIMILARITY_THRESHOLD}\n\n")
        f.write("---\n\n")
        
        # Résumé par niveau
        f.write("## 📈 Résumé par Niveau\n\n")
        f.write("| Niveau | Succès | Total | Taux | Status |\n")
        f.write("|--------|--------|-------|------|--------|\n")
        
        for level_name, results in results_by_level.items():
            level_success = sum(1 for r in results if r.get('is_success'))
            level_total = len(results)
            level_rate = (level_success / level_total) * 100
            status_icon = "🥇" if level_rate >= 85 else "🥈" if level_rate >= 70 else "🥉"
            
            f.write(f"| {level_name} | {level_success} | {level_total} | {level_rate:.1f}% | {status_icon} |\n")
        
        f.write("\n---\n\n")
        
        # Détails par question
        f.write("## 📋 Détails par Question\n\n")
        
        for level_name, results in results_by_level.items():
            f.write(f"### {level_name}\n\n")
            
            for r in results:
                f.write(f"#### {r['status']} {r['id']}: {r['category']}\n\n")
                f.write(f"**Question** : {r['question']}\n\n")
                
                if r.get('error'):
                    f.write(f"**❌ Erreur** : {r['error']}\n\n")
                else:
                    f.write(f"- **Article attendu** : {r.get('expected_article', 'N/A')}\n")
                    f.write(f"- **Articles trouvés** : {', '.join(r.get('articles_found', [])) or 'Aucun'}\n")
                    f.write(f"- **Score** : {r.get('best_score', 0):.3f} (seuil: {r.get('min_score', 0)})\n")
                    f.write(f"- **Citations** : {'✅ Oui' if r.get('has_citation') else '❌ Non'}\n")
                    
                    if r.get('transformations'):
                        trans = r['transformations']
                        if trans.get('expanded') != trans.get('normalized'):
                            f.write(f"- **Query Expansion** : ✅\n")
                
                f.write("\n---\n\n")
    
    print(f"\n💾 Rapport détaillé sauvegardé : {report_path}")


# ========== POINT D'ENTRÉE ==========

if __name__ == "__main__":
    try:
        results, success_rate, certification = run_test_battery()
        
        print("\n" + "="*70)
        print("✅ Tests terminés avec succès !")
        print("📄 Consulte docs/test-report.md pour le rapport détaillé")
        print("="*70)
        
        # Code de sortie selon certification
        if success_rate >= 73:
            sys.exit(0)  # Succès
        else:
            sys.exit(1)  # À améliorer
            
    except KeyboardInterrupt:
        print("\n\n⏸️ Tests interrompus par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
