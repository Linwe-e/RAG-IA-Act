"""
🔍 Inspection directe ChromaDB - INSTANTANÉ
Vérifie le format des chunks sans appeler le LLM
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL, VECTORDB_DIR
import re

print("="*70)
print("🔍 INSPECTION CHROMADB - Format des Articles")
print("="*70)

# Vérifier que le dossier existe
from pathlib import Path
vectordb_path = Path(VECTORDB_DIR)
print(f"\n📁 Chemin base vectorielle: {vectordb_path}")
print(f"   Existe ? {vectordb_path.exists()}")
if vectordb_path.exists():
    print(f"   Contenu: {list(vectordb_path.iterdir())}")

# Charger ChromaDB directement
print("\n📂 Chargement base vectorielle...")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=VECTORDB_DIR,
    embedding_function=embeddings,
    collection_name="ia_act_en"  # ⚠️ IMPORTANT : Même nom que dans ingest.py
)

# Compter le nombre total de documents
try:
    collection = vectorstore._collection
    total_docs = collection.count()
    print(f"✅ Base chargée : {total_docs} documents au total")
except Exception as e:
    print(f"⚠️ Impossible de compter les documents : {e}")
    total_docs = 0

# Rechercher chunks contenant "Article 16"
print("\n🔎 Recherche chunks avec 'Article 16'...")
results = vectorstore.similarity_search("Article 16", k=3)

print(f"✅ {len(results)} chunks trouvés\n")

if not results and total_docs > 0:
    print("⚠️ Aucun résultat pour 'Article 16', essai recherche générique...")
    results = vectorstore.similarity_search("AI Act provider obligations", k=3)
    print(f"   Recherche générique : {len(results)} chunks trouvés\n")

if not results and total_docs > 0:
    print("⚠️ Essai recherche ultra-générique...")
    results = vectorstore.similarity_search("artificial intelligence", k=3)
    print(f"   Recherche ultra-générique : {len(results)} chunks trouvés\n")

for i, doc in enumerate(results, 1):
    print(f"\n{'='*70}")
    print(f"CHUNK {i}")
    print(f"{'='*70}")
    print(f"Page: {doc.metadata.get('page', 'N/A')}")
    
    content = doc.page_content
    print(f"Longueur: {len(content)} caractères")
    
    # Tester regex
    matches = re.findall(r'(?i)(?:Article|Art\.?)\s+(\d+)', content)
    print(f"\n🔍 Articles détectés par regex: {matches if matches else 'AUCUN'}")
    
    # Chercher "16" manuellement
    if "16" in content:
        print(f"✅ Le texte contient '16'")
        idx = content.find("16")
        context = content[max(0, idx-50):idx+100]
        print(f"\nContexte autour de '16':")
        print(f"   ...{context}...")
    
    # Afficher début du chunk
    print(f"\n📄 DÉBUT DU CHUNK:")
    print("-"*70)
    print(content[:500])
    print("-"*70)

print("\n" + "="*70)
print("✅ Inspection terminée !")
