# 🎯 Batterie de Tests RAG - Certification Qualité

## 📋 Protocole de Test

Pour **chaque question**, note :
1. ✅ **Article(s) trouvé(s)** : Numéro exact
2. 📊 **Meilleur score** : Similarité max
3. 🎯 **Pertinence** : Réponse correcte ? (Oui/Non/Partiel)
4. 💬 **Citation** : Le LLM cite-t-il l'article source ?

---

## 🥇 NIVEAU 1 : Questions Directes (Vocabulaire Juridique)

### Test 1.1 : Acteurs & Rôles
```
Q1: What are the obligations of AI providers under Article 16?
✅ Attendu: Article 16 (score > 0.60)

Q2: What must AI deployers do according to Article 26?
✅ Attendu: Article 26 (score > 0.55)

Q3: What is the difference between a provider and a deployer?
✅ Attendu: Articles 3 (définitions) + 16 + 26 (score > 0.45)
```

---

### Test 1.2 : Systèmes Interdits
```
Q4: Which AI practices are prohibited by the AI Act?
✅ Attendu: Article 5 (score > 0.65)

Q5: Is social scoring by public authorities allowed?
✅ Attendu: Article 5(1)(c) (score > 0.55)

Q6: Can AI manipulate human behavior through subliminal techniques?
✅ Attendu: Article 5(1)(a) (score > 0.50)
```

---

### Test 1.3 : Systèmes à Haut Risque
```
Q7: What makes an AI system high-risk?
✅ Attendu: Article 6 + Annex III (score > 0.60)

Q8: What are the conformity assessment obligations for high-risk AI?
✅ Attendu: Article 43 (score > 0.55)

Q9: Must high-risk AI systems have human oversight?
✅ Attendu: Article 14 (score > 0.60)
```

---

## 🥈 NIVEAU 2 : Questions Citoyennes (Langage Naturel)

### Test 2.1 : Obligations Employeur

```
Q10: Mon patron doit-il me former à l'intelligence artificielle ?
✅ Attendu: Article 4 (AI literacy) (score > 0.40)

Q11: My company uses AI for hiring. What are their legal obligations?
✅ Attendu: Article 6 + Annex III (employment) + Article 26 (score > 0.45)

Q12: Can my employer use AI to monitor my work performance?
✅ Attendu: Article 6 + Annex III (point 4) (score > 0.40)
```

---

### Test 2.2 : Droits des Personnes

```
Q13: Ai-je le droit de savoir quand une IA prend une décision me concernant ?
✅ Attendu: Article 26 (deployer transparency) + Article 86 (score > 0.35)

Q14: Can I refuse to interact with an AI system at work?
✅ Attendu: Article 26(8) (human oversight) (score > 0.35)

Q15: Puis-je demander une explication sur une décision prise par une IA ?
✅ Attendu: Article 86 (right to explanation) (score > 0.40)
```

---

### Test 2.3 : Cas d'Usage Spécifiques

```
Q16: Est-ce qu'un chatbot doit dire qu'il est une IA ?
✅ Attendu: Article 52(1) (transparency chatbots) (score > 0.45)

Q17: My doctor uses AI to diagnose illnesses. Is this allowed?
✅ Attendu: Article 6 + Annex III (medical devices) (score > 0.40)

Q18: Can police use facial recognition in public spaces?
✅ Attendu: Article 5(1)(d) + exceptions Article 5(2) (score > 0.45)
```

---

## 🥉 NIVEAU 3 : Questions Pièges (Concepts Complexes)

### Test 3.1 : Ambiguïtés Terminologiques

```
Q19: What training data requirements exist for AI systems?
⚠️ Piège: "training" = data, pas literacy
✅ Attendu: Article 10 (training data) (PAS Article 4)

Q20: Who must train AI systems to ensure safety?
⚠️ Piège: "train systems" = développement
✅ Attendu: Article 9 (risk management) + Article 15 (accuracy)

Q21: My company trains employees on AI. What's required?
⚠️ Piège: Distinguer AI literacy vs deployer obligations
✅ Attendu: Article 4 (literacy) ET Article 26(5) (deployer staff)
```

---

### Test 3.2 : Multi-Articles (Réponse Nécessite Plusieurs Sources)

```
Q22: What are ALL the steps to put a high-risk AI on the EU market?
✅ Attendu: Article 16 (provider obligations) + Article 43 (conformity) + Article 49 (CE marking) (score > 0.40 chacun)

Q23: What happens if a company violates the AI Act?
✅ Attendu: Article 99 (fines) + Article 73 (market surveillance) (score > 0.50)

Q24: Can I develop AI in a regulatory sandbox and what are the conditions?
✅ Attendu: Article 57 (sandbox definition) + Article 58 (conditions) (score > 0.45)
```

---

### Test 3.3 : Edge Cases Juridiques

```
Q25: Does the AI Act apply to open-source AI models?
✅ Attendu: Article 2 (scope) + Recital 60 (open-source) (score > 0.35)
⚠️ Difficulté: Concept dans Recitals, pas articles principaux

Q26: What if an AI system from the US is used in France?
✅ Attendu: Article 2(1) (territorial scope) (score > 0.40)

Q27: When does the AI Act enter into force?
✅ Attendu: Article 113 (entry into force) (score > 0.55)
```

---

## 🏆 NIVEAU 4 : Stress Test (Questions "Impossibles")

### Test 4.1 : Hors Périmètre (Doit Échouer Gracieusement)

```
Q28: How do I build a neural network in Python?
❌ Attendu: Aucun chunk > 0.30 → Réponse "hors périmètre"

Q29: What's the GDPR regulation about personal data?
❌ Attendu: Peut matcher "personal data" mais contexte IA Act
→ Vérifier que LLM ne confond pas GDPR et AI Act

Q30: Combien coûte une formation en IA ?
❌ Attendu: Aucun chunk pertinent → "Information non disponible"
```

---

### Test 4.2 : Questions Vagues (Test Robustesse)

```
Q31: Tell me about AI.
⚠️ Trop vague
✅ Attendu: Article 3 (definitions) OU message "question trop générale"

Q32: Is AI dangerous?
⚠️ Ambiguë
✅ Attendu: Article 5 (prohibited) + Article 6 (high-risk)

Q33: What's the AI Act?
✅ Attendu: Article 1 (subject matter) + Recital 1 (score > 0.50)
```

---

## 📊 Grille d'Évaluation

### Score par Niveau

| Niveau | Questions | Seuil Succès | Ton Objectif |
|--------|-----------|--------------|--------------|
| **Niveau 1** | 9 questions | 8/9 (89%) | Score > 0.55 |
| **Niveau 2** | 9 questions | 7/9 (78%) | Score > 0.40 |
| **Niveau 3** | 9 questions | 6/9 (67%) | Score > 0.35 |
| **Niveau 4** | 6 questions | 4/6 (67%) | Gestion erreur |

### Certification Qualité

- **🥇 Gold** : 28+/33 (85%) réponses pertinentes
- **🥈 Silver** : 24+/33 (73%) réponses pertinentes
- **🥉 Bronze** : 20+/33 (61%) réponses pertinentes

---

## 🎯 Format de Test Recommandé

### Option A : Test Automatique (Si Temps)

```python
# test_rag.py
QUESTIONS = [
    ("Q1", "What are the obligations...", 16, 0.60),
    # (id, question, article_attendu, score_min)
]

for id, q, expected_article, min_score in QUESTIONS:
    result = retrieve_context(q)
    score = max(result['distances'][0])
    
    print(f"{id}: {'✅' if score >= min_score else '❌'} (score={score:.2f})")
```

---

### Option B : Test Manuel (Recommandé Pour Toi)

**Spreadsheet** :

| ID | Question | Article Trouvé | Score | ✅/❌ | Notes |
|----|----------|----------------|-------|-------|-------|
| Q1 | What are... | 16 | 0.62 | ✅ | Parfait |
| Q10 | Mon patron... | 4 | 0.38 | ✅ | Score limite mais OK |

---

## 💡 Conseil Testing

### Commence Par :
1. **3 questions Niveau 1** → Valide base technique
2. **3 questions Niveau 2** → Valide query expansion
3. **1 question Niveau 4** → Valide gestion erreur

### Si > 6/7 Succès :
→ Lance le test complet ! 🚀

---

📄 **Mode Doc** - Commence par les **9 premières questions** (Niveaux 1-2) et partage tes résultats (format libre). On analyse ensemble ! 🎯