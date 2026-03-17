"""Test Query Expansion avec question citoyenne"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from retriever import create_retriever

print("=" * 70)
print("🧪 TEST QUERY EXPANSION - Question Citoyenne")
print("=" * 70)

# Question citoyenne typique
question = "Mon entreprise doit-elle me dire quand elle utilise l'IA ?"

print(f"\n❓ Question citoyenne : {question}\n")

# Créer retriever
retriever = create_retriever()

# Retrieve avec Query Expansion activée
results = retriever.retrieve_with_metadata(question, k=3)

print(f"\n📊 {len(results)} résultats trouvés :\n")
for i, r in enumerate(results, 1):
    print(f"   {i}. Page {r['page']} - Similarité: {r['similarity']:.3f}")
    print(f"      Extrait: {r['content'][:100]}...")

print("\n" + "=" * 70)
