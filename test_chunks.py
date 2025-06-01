from chromadb import PersistentClient

# Připojení k databázi
client = PersistentClient(path="./chroma_db")
collection = client.get_collection("hypoteky_all")  # Ujisti se, že název odpovídá

# Získání dokumentů podle názvu PDF
results = collection.get(
    where={"dokument": {"$eq": "Hypoteky_RB_ucely_rekonstrukce.pdf"}}
)

# Výpis obsahu
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(f"📄 {meta.get('dokument')} | strana {meta.get('strana')} | kapitola: {meta.get('kapitola', '?')}")
    print(doc)
    print("=" * 80)