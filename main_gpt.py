import os
from dotenv import load_dotenv
import openai
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
model = SentenceTransformer("intfloat/multilingual-e5-large")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="hypoteky_all")

print(f"🔍 Počet uložených záznamů v kolekci: {len(collection.get()['ids'])}")

dotaz = input("Zadej dotaz: ")
embedding = model.encode(f"query: {dotaz}").tolist()

results = collection.query(
    query_embeddings=[embedding],
    n_results=15,
    include=["documents", "metadatas"]
)

relevant_chunks = []
citace = []
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    relevant_chunks.append(doc)
    if "strana" not in meta:
        print(f"⚠️  Chybí klíč 'strana' v metadatech: {meta}")
    citace.append(
        f"(dokument: {meta.get('dokument', '?')}, strana: {meta.get('strana', '?')}, kapitola: {meta.get('kapitola', '?')})"
    )

messages = [
    {
        "role": "system",
        "content": (
            "Jsi expertní asistent na hypotéky a posuzování bonity klientů podle interních metodik banky. "
            "Tvým úkolem je odpovídat pouze na základě poskytnutých úryvků z těchto metodik. "
            "Nepoužívej žádné obecné znalosti nebo domněnky mimo tyto dokumenty. "
            "Pokud některé části úryvků odpovídají dotazu jen částečně nebo jsou zformulovány jinými slovy, použij je. "
            "Pokud odpověď nelze najít ve výňatcích, jasně uveď, že není možné odpovědět bez dalších informací. "
            "V odpovědi vždy uveď konkrétní citaci ve formátu: (dokument: <název>, strana: <číslo>, kapitola: <číslo>). "
            "Drž se pouze faktů z úryvků a neuváděj doporučení nebo obecné poznámky, které nejsou přímo v textech."
        )
    },
    {
        "role": "user",
        "content": f"Dotaz: {dotaz}\n\nZde jsou úryvky z dokumentů:\n\n" +
                   "\n\n".join([f"{chunk}\nUmístění: {cit}" for chunk, cit in zip(relevant_chunks, citace)])
    }
]

from openai import OpenAI

client = OpenAI(api_key=openai.api_key)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
)

print("📘 Odpověď GPT-4o:")
print("-" * 60)
print(response.choices[0].message.content)
input("\nStiskni Enter pro ukončení...")
