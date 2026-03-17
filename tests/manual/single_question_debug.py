"""
⚡⚡ Test Ultra-Rapide - 1 question debug
"""

import sys
from pathlib import Path

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    # Désactiver buffering pour affichage immédiat
    sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from generator import IAActGenerator
import re

def print_flush(msg):
    """Print avec flush immédiat"""
    print(msg)
    sys.stdout.flush()

print_flush("="*70)
print_flush("⚡⚡ TEST DEBUG - 1 QUESTION")
print_flush("="*70)

print_flush("\n🔧 Initialisation generator...")
gen = IAActGenerator()
print_flush("✅ Generator prêt")

question = "What are the obligations of AI providers under Article 16?"
print_flush(f"\n📝 Question: {question}")
print_flush("🔄 Génération en cours...")

result = gen.generate(question, return_sources=True)

sources = result.get('sources', [])
print_flush(f"\n✅ {len(sources)} sources retournées")

# Analyser premier chunk
if sources:
    first = sources[0]
    print_flush(f"\n📊 PREMIER CHUNK:")
    print_flush(f"   Score: {first.get('similarity', 0):.3f}")
    print_flush(f"   Page: {first.get('page')}")
    
    content = first.get('content', '')
    print_flush(f"\n📄 CONTENU (200 premiers caractères):")
    print_flush(f"   {content[:200].replace(chr(10), ' ')}")
    
    # Test regex
    matches = re.findall(r'(?i)(?:Article|Art\.?)\s+(\d+)', content)
    print_flush(f"\n🔍 Articles trouvés avec regex: {matches if matches else 'AUCUN'}")
    
    # Chercher manuellement "16" dans le texte
    if "16" in content:
        print_flush(f"   ✅ Le nombre '16' EST présent dans le texte")
        # Extraire contexte autour de "16"
        idx = content.find("16")
        context = content[max(0, idx-30):idx+50]
        print_flush(f"   Contexte: ...{context}...")
    else:
        print_flush(f"   ❌ Le nombre '16' N'EST PAS dans ce chunk")
    
    print_flush(f"\n💡 ANALYSE:")
    if matches:
        print_flush(f"   ✅ Le regex fonctionne ! Articles détectés: {matches}")
    else:
        print_flush(f"   ❌ Le regex ne match pas. Problème de format.")
        print_flush(f"   → Chunk entier ci-dessous pour analyse:")
        print_flush(f"\n{'-'*70}")
        print_flush(content)
        print_flush(f"{'-'*70}")

print_flush("\n" + "="*70)
