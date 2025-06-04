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
            "Jsi expertní asistent na hypotéky a posuzování bonity klientů podle interních metodik bank.\n\n"
            "Odpovídej výhradně na základě úryvků z dodaných dokumentů. Pokud odpověď není v úryvcích výslovně obsažena, řekni to jasně.\n\n"
            "Pokud ale dotaz není výslovně zodpovězen, ale lze ho logicky odvodit na základě výpočtových metod, tabulek, nebo popsaných postupů bank, uveď odvozenou odpověď spolu s důvodem a citací.\n\n"
            "Například: pokud dokument obsahuje návod, jak banka počítá příjem z obratu, považuj to za důkaz, že banka tento typ příjmu akceptuje.\n\n"
            "U dotazů typu „které banky akceptují…“ nejprve zjisti, zda dokumenty obsahují jakýkoli výpočet, postup nebo metodiku související s tímto případem – pokud ano, banku uveď jako relevantní.\n\n"
            "Odpověď strukturu jako výčet bank a pro každou uveď související výrok + citaci:\n\n"
            "Název banky\n"
            "Shrnutí, jak se daná věc posuzuje nebo vypočítává.\n"
            "(dokument: <název>, strana: <číslo>, kapitola: <číslo>)\n\n"
            "Nepoužívej žádné vymyšlené informace. Neodkazuj na web ani na neexistující zdroje."
        )
    },
    {"role": "user", "content": dotaz}
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
