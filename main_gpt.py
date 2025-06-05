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
    n_results=50,
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
    "Pokud ale dotaz není výslovně zodpovězen, ale lze jej logicky odvodit na základě výpočtových metod, posuzovacích pravidel, nebo postupů popsaných v dokumentech, tak odpověď uveď. Přidej zdůvodnění a citaci.\n\n"
    "Pokud dotaz obsahuje konkrétní pojem (např. typ příjmu, forma zaměstnání, struktura smlouvy) a dokumenty tento pojem neobsahují doslova, ale popisují výpočet, pravidlo nebo situaci s tímto pojmem logicky související, považuj to za platnou informaci a uveď ji.\n\n"
    "Příklad: Pokud dokument uvádí, že banka pro výpočet příjmu používá obrat dělený 12 nebo průměr plateb na účtu, považuj to za důkaz, že banka akceptuje příjem z obratu.\n\n"
    "U dotazů typu „Které banky akceptují…“ vyhledej, zda některá z bank v úryvcích popisuje výpočet nebo metodu posuzování související s tímto případem – pokud ano, tuto banku uveď jako relevantní.\n\n"
    "Pokud je v dotazu uveden konkrétní název banky (např. „Komerční banka“), prioritně prohledej dokumenty, které jsou s touto bankou přímo spojeny.\n"
    "Použij tento seznam vazeb mezi bankami a dokumenty:\n"
    "- Komerční banka → Hypoteky_KB.pdf\n"
    "- Raiffeisenbank → Hypoteky_RB_bonita_podnikani.pdf\n"
    "- Česká spořitelna → Hypoteky_CS.pdf\n"
    "- ČSOB Hypoteční banka → Hypoteky_ČSOBHB.pdf\n"
    "- mBank → Hypoteky_mB.pdf\n"
    "- UniCredit Bank → Hypoteky_UCB.pdf\n"
    "- Oberbank AG → Hypoteky_OB.pdf\n\n"
    "Ostatní dokumenty zohledni pouze tehdy, pokud daná banka nemá vlastní dokument nebo pokud je v nich daná banka výslovně jmenována.\n"
    "Nikdy nepřiřazuj informace mezi různými bankami jen na základě tematické podobnosti.\n\n"
    "Pokud je dotaz srovnávacího typu (např. „Která banka nabízí nejvyšší LTV…“, „Která banka umožňuje nejdelší splatnost…“), vyhledej a porovnej všechny relevantní hodnoty ve všech dostupných úryvcích. Uveď výsledek s nejvyšší (nebo nejnižší) hodnotou a doplň jej citací. Pokud více bank nabízí stejnou maximální hodnotu, vypiš všechny.\n"
    "Nikdy nevynechávej banku, pokud se v úryvku objevuje s hodnotou, která odpovídá dotazu.\n\n"
    "Struktura odpovědi:\n"
    "- Název banky\n"
    "- Shrnutí, jak se daná věc posuzuje nebo vypočítává\n"
    "- Přesná citace (dokument: <název>, strana: <číslo>, kapitola: <číslo>)\n\n"
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
