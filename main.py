import json
import chromadb
from sentence_transformers import SentenceTransformer

# Krok 1: Načtení JSON bloků
with open("data/hypoteky_cs_knowledge_blocks.json", "r", encoding="utf-8") as f:
    knowledge_blocks = json.load(f)

# Krok 2: Inicializace modelu
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Krok 3: Inicializace ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection(name="hypoteky_cs")

# Krok 4: Uložení dokumentů s embeddingy
for i, block in enumerate(knowledge_blocks):
    embedding = model.encode(block["obsah"]).tolist()
    metadata = {
        "dokument": block["dokument"],
        "strana": block["strana"]
    }
    collection.add(
        documents=[block["obsah"]],
        embeddings=[embedding],
        ids=[f"block_{i}"],
        metadatas=[metadata]
    )

# Krok 5: Vyhledávání (ukázka)
dotaz = input("Zadej dotaz: ")
dotaz_embedding = model.encode(dotaz).tolist()

results = collection.query(query_embeddings=[dotaz_embedding], n_results=3)

print("\n🔍 Nejrelevantnější odpovědi:")
for result in results["documents"][0]:
    print("-" * 50)
    print(result)

input("\nStiskni Enter pro ukončení...")
