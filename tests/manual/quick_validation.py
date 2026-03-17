"""
⚡ Test Rapide RAG - Validation Core (9 questions)
Version allégée pour validation rapide du fix
"""

import sys
import os
from pathlib import Path

# Fix encodage Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from generator import IAActGenerator
from config import SIMILARITY_THRESHOLD
import config
config.VERBOSE = False

import re
from typing import Dict, List

# ========== QUESTIONS TEST RAPIDE ==========
QUICK_TESTS = [
    # NIVEAU 1 - Vocabulaire juridique (3 questions)
    {
        "id": "Q1",
        "question": "What are the obligations of AI providers under Article 16?",
        "expected_article": "16",
        "min_score": 0.60,
        "category": "Niveau 1"
    },
    {
        "id": "Q4",
        "question": "Which AI practices are prohibited by the AI Act?",
        "expected_article": "5",
        "min_score": 0.65,
        "category": "Niveau 1"
    },
    {
        "id": "Q7",
        "question": "What makes an AI system high-risk?",
        "expected_article": "6",
        "min_score": 0.60,
        "category": "Niveau 1"
    },
    
    # NIVEAU 2 - Questions citoyennes (3 questions)
    {
        "id": "Q10",
        "question": "Mon patron doit-il me former à l'intelligence artificielle ?",
        "expected_article": "4",
        "min_score": 0.40,
        "category": "Niveau 2"
    },
    {
        "id": "Q16",
        "question": "Est-ce qu'un chatbot doit dire qu'il est une IA ?",
        "expected_article": "52",
        "min_score": 0.45,
        "category": "Niveau 2"
    },
    {
        "id": "Q18",
        "question": "Can police use facial recognition in public spaces?",
        "expected_article": "5",
        "min_score": 0.45,
        "category": "Niveau 2"
    },
    
    # NIVEAU 4 - Hors périmètre (3 questions)
    {
        "id": "Q28",
        "question": "How do I build a neural network in Python?",
        "expected_article": None,
        "min_score": 0.30,
        "category": "Niveau 4",
        "should_fail": True
    },
    {
        "id": "Q29",
        "question": "What's the GDPR regulation about personal data?",
        "expected_article": None,
        "min_score": 0.30,
        "category": "Niveau 4",
        "should_fail": True
    },
    {
        "id": "Q33",
        "question": "What's the AI Act?",
        "expected_article": "1",
        "min_score": 0.50,
        "category": "Niveau 4"
    },
]


def extract_articles(sources: List[Dict]) -> List[str]:
    """Extrait numéros d'articles avec regex amélioré + debug
    
    ⚠️ ATTENTION : Le PDF a des espaces dans les mots : "Ar ticle" au lieu de "Article"
    Regex accepte espaces optionnels entre lettres
    """
    articles = []
    for i, source in enumerate(sources):
        content = source.get('content', '')
        
        # Regex permissif : A r t i c l e (espaces optionnels) + numéro
        # Accepte : "Article 16", "Ar ticle 16", "Art. 16", "Art 16"
        matches = re.findall(
            r'(?i)A\s*r\s*t\s*(?:i\s*c\s*l\s*e|\.?)\s+(\d+)', 
            content
        )
        articles.extend(matches)
        
        # Debug premier chunk si aucun article trouvé
        if i == 0 and not matches:
            preview = content[:200].replace('\n', ' ')
            print(f"   🔍 DEBUG: Pas d'article dans chunk 1")
            print(f"      Preview: {preview}...")
    
    return list(set(articles))  # Unique


def get_best_score(sources: List[Dict]) -> float:
    if not sources:
        return 0.0
    return max([s.get('similarity', 0.0) for s in sources])


def test_question(generator, test_case: Dict) -> Dict:
    question = test_case['question']
    expected = test_case.get('expected_article')
    min_score = test_case.get('min_score', 0.30)
    should_fail = test_case.get('should_fail', False)
    
    print(f"\n{'='*70}")
    print(f"🧪 {test_case['id']}: {question[:60]}...")
    print(f"   Attendu: Article {expected if expected else 'Aucun'} (score > {min_score})")
    
    try:
        result = generator.generate(question, return_sources=True, return_transformations=True)
        
        sources = result.get('sources', [])
        best_score = get_best_score(sources)
        articles_found = extract_articles(sources)
        
        # Évaluation
        if should_fail:
            # Questions hors périmètre : succès = score faible OU pas de sources
            is_success = (best_score < min_score) or (len(sources) == 0)
            status = "✅" if is_success else "❌"
        else:
            article_match = expected in articles_found if expected else True
            is_success = (best_score >= min_score) and article_match
            status = "✅" if is_success else "⚠️" if best_score >= (min_score - 0.05) else "❌"
        
        # Affichage résultat
        print(f"   {status} Score: {best_score:.3f} | Articles: {', '.join(articles_found) if articles_found else 'Aucun'}")
        
        return {
            "id": test_case['id'],
            "status": status,
            "is_success": is_success,
            "score": best_score,
            "articles": articles_found,
            "expected": expected
        }
        
    except KeyboardInterrupt:
        raise  # Propager interruption clavier
    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        return {"id": test_case['id'], "status": "❌", "is_success": False, "error": str(e)}


def main():
    print("="*70)
    print("⚡ TEST RAPIDE RAG - VALIDATION FIX (9 questions)")
    print("="*70)
    print(f"⚙️ Seuil: {SIMILARITY_THRESHOLD}\n")
    
    print("🔧 Initialisation generator...")
    generator = IAActGenerator()
    print("✅ Prêt\n")
    
    results = []
    for test_case in QUICK_TESTS:
        result = test_question(generator, test_case)
        results.append(result)
    
    # Rapport final
    print("\n" + "="*70)
    print("📊 RÉSULTATS")
    print("="*70)
    
    success_count = sum(1 for r in results if r.get('is_success'))
    total = len(results)
    rate = (success_count / total) * 100
    
    print(f"\n🎯 Score: {success_count}/{total} ({rate:.1f}%)")
    
    # Détails par catégorie
    categories = {}
    for r in results:
        cat = [t['category'] for t in QUICK_TESTS if t['id'] == r['id']][0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    for cat, cat_results in categories.items():
        cat_success = sum(1 for r in cat_results if r.get('is_success'))
        cat_total = len(cat_results)
        print(f"\n   {cat}: {cat_success}/{cat_total}")
        for r in cat_results:
            print(f"      {r['status']} {r['id']}: score={r.get('score', 0):.3f}, articles={r.get('articles', [])}")
    
    # Certification simplifiée
    if rate >= 80:
        print(f"\n✅ SUCCÈS ! Le fix fonctionne correctement.")
        return 0
    elif rate >= 60:
        print(f"\n⚠️ PARTIEL - Besoin d'optimisations supplémentaires")
        return 1
    else:
        print(f"\n❌ ÉCHEC - Problème persistant à investiguer")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏸️ Test interrompu")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
