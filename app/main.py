import markdown
import re
import urllib.parse

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import chromadb
import openai
import os
import secrets
import csv
from dotenv import load_dotenv
from fulltext_validator import analyzuj_relevantni_banky_fulltextem

# --- Přidáno: zpřístupnění modulu z root složky ---
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fulltext_validator import analyzuj_relevantni_banky_fulltextem

# === Zvýraznění a prolinkování citací ===
def highlight_citations(text: str) -> str:
    import re, urllib.parse

    def replace(match):
        citation = match.group(0)
        doc_match = re.search(r'dokument: ([^)]+)', citation)

        if not doc_match:
            return citation

        filename = doc_match.group(1).strip()
        safe_filename = urllib.parse.quote(filename)
        url = f"/metodiky/{safe_filename}"

        return f"<a href='{url}' target='_blank' class='citation'>{citation}</a>"

    pattern = r"\(dokument: [^)]+\)"
    return re.sub(pattern, replace, text)

# === Načtení .env proměnných ===
load_dotenv()
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("APP_PASSWORD")

# === Inicializace aplikace ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/metodiky", StaticFiles(directory="metodiky_bank"), name="metodiky")

templates = Jinja2Templates(directory="app/templates")
security = HTTPBasic()

# === Embed model a vektorová DB ===
model = SentenceTransformer("intfloat/multilingual-e5-large")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("hypoteky_all")

# === Autentizace ===
def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not (
        secrets.compare_digest(credentials.username, USERNAME)
        and secrets.compare_digest(credentials.password, PASSWORD)
    ):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Přístup odepřen",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# === Model zpětné vazby ===
class Feedback(BaseModel):
    question: str
    answer: str
    feedback: str
    comment: str = ""

@app.post("/feedback")
async def receive_feedback(data: Feedback):
    feedback_file = "feedback.csv"
    file_exists = os.path.isfile(feedback_file)

    with open(feedback_file, mode="a", encoding="cp1250", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["question", "answer", "feedback", "comment"])
        writer.writerow([data.question, data.answer, data.feedback, data.comment])

    return JSONResponse(content={"status": "success"})

# === GET ===
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request, username: str = Depends(check_auth)):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# === POST ===
@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, dotaz: str = Form(...), username: str = Depends(check_auth)):
    embedding = model.encode(f"query: {dotaz.strip()}").tolist()
    results = collection.query(query_embeddings=[embedding], n_results=80, include=["documents", "metadatas"])

    # 🔍 DIAGNOSTICKÝ TEST: hledání "výživné"
    vsechny = collection.get(limit=None)
    print("📦 Test: Fulltextově hledám 'výživné' v databázi...")

    for doc, meta in zip(vsechny.get("documents", []), vsechny.get("metadatas", [])):
        if "výživné" in doc.lower():
            print(f"✅ Nalezeno: banka={meta.get('banka')}, dokument={meta.get('document_source')}")

    from fulltext_validator import (
        analyzuj_relevantni_banky_fulltextem,
        zjisti_banky_z_embeddingu,
        porovnej_pokryti,
        analyzuj_banky_z_fulltextu  # <- přidáme pro výpis
    )

    banky_z_fulltextu = analyzuj_banky_z_fulltextu(vsechny.get("metadatas", []))
    print("📊 Banky ve fulltext výstupu:", banky_z_fulltextu)

    # === Validace pokrytí pomocí fulltextu ===
    from fulltext_validator import (
        analyzuj_relevantni_banky_fulltextem,
        zjisti_banky_z_embeddingu,
        porovnej_pokryti
    )

    hledane_slovo = dotaz.strip().lower()
    fulltext_banky = analyzuj_relevantni_banky_fulltextem(collection, hledane_slovo)
    embedding_banky = zjisti_banky_z_embeddingu(results["metadatas"][0])

    # 💬 DEBUG výpisy:
    print("📊 Banky ve fulltext výstupu:", fulltext_banky)
    print("📊 Banky ve vektorovém výstupu:", embedding_banky)

    chybejici = porovnej_pokryti(fulltext_banky, embedding_banky)

    if chybejici:
        print("⚠️ Chybějící banky podle fulltext analýzy:", chybejici)

    def detect_bank(text):
        text = text.lower()
        if "kb" in text or "komercni" in text or "komerční" in text:
            return "Hypoteky_KB.pdf"
        elif "čsob" in text or "csob" in text:
            return "Hypoteky_ČSOBHB.pdf"
        elif "rb" in text or "raiffeisen" in text:
            return "Hypoteky_RB_bonita_podnikani.pdf"
        elif "ucb" in text or "unicredit" in text:
            return "Hypoteky_UCB.pdf"
        elif "mbank" in text or "mb" in text:
            return "Hypoteky_mB.pdf"
        elif "čs" in text or "ceska sporitelna" in text:
            return "Hypoteky_CS.pdf"
        elif "ob" in text or "oberbank" in text:
            return "Hypoteky_OB.pdf"
        else:
            return None

    requested_doc = detect_bank(dotaz)

    if requested_doc:
        found = any(
            requested_doc.lower() in (meta.get("document_source") or "").lower()
            for meta in results["metadatas"][0]
        )
        if not found:
            fallback = collection.get(limit=None)
            for doc, meta in zip(fallback["documents"], fallback["metadatas"]):
                if requested_doc.lower() in (meta.get("document_source") or "").lower():
                    results["documents"][0].append(doc)
                    results["metadatas"][0].append(meta)
                    print(f"✅ Fallback přidal dokument: {requested_doc}")
                    break

    # 💡 DIAGNOSTIKA: banky obsažené ve výsledcích (embedding)
    banky_v_odpovedi = {
        meta.get("banka", "Neznámá banka").lower()
        for meta in results["metadatas"][0]
    }
    print("📋 Banky zahrnuté do odpovědi GPT:", banky_v_odpovedi)
    
    relevant_chunks = results["documents"][0]
    citace = [
        f"(dokument: {meta.get('document_source', '?')}, strana: {meta.get('strana', '?')}, kapitola: {meta.get('kapitola', '?')})"
        for meta in results["metadatas"][0]
    ]
    
    messages = [
        {
            "role": "system",
            "content": (
                "Jsi expertní asistent na hypotéky a posuzování bonity klientů podle interních metodik bank.\n\n"
    "🔍 Nejprve zjisti, co je vstupem uživatele:\n"
    "1. Pokud jde o plnohodnotný dotaz, klasifikuj ho interně do jedné z těchto kategorií:\n"
    "   - výčtový\n"
    "   - srovnávací\n"
    "   - faktický\n"
    "   - podmínkový\n"
    "   - kombinovaný\n"
    "2. Pokud vstup není úplným dotazem (např. jen fragment jako „výživné jako příjem žadatele“), logicky odvoď, co uživatel pravděpodobně zjišťuje, a pokračuj podle odpovídající logiky.\n"
    "3. Pokud dotaz neobsahuje název konkrétní banky, agreguj odpovědi napříč všemi dostupnými dokumenty. Nikdy se nespokojuj pouze s jedním úryvkem nebo jednou bankou.\n"
    "4. Pokud dotaz obsahuje konkrétní banku, pracuj primárně s dokumenty této banky. Ostatní dokumenty zvaž pouze tehdy, pokud je tato banka výslovně zmíněna jinde nebo pokud vlastní dokument chybí.\n\n"
    "🧩 Instrukce podle typu dotazu:\n"
    "- Výčtový: Vypiš každou banku, která podmínku splňuje. Každou zvlášť se stručným shrnutím a citací.\n"
    "  ➕ Pokud máš chunk pro danou banku, ale nenacházíš v něm přímou zmínku k dotazu, zvaž možnost odpovědi založené na kombinaci dotazu a názvu banky. Shrň i nepřímé nebo kontextové informace, pokud jsou v chuncích uvedeny.\n"
    "- Srovnávací: Porovnej hodnoty napříč bankami a uveď pouze tu nejlepší (nebo několik s nejvyšší hodnotou).\n"
    "- Faktický: Odpověz přesně a s citací. Pokud informace chybí, napiš to jasně.\n"
    "- Podmínkový: Popiš okolnosti, za kterých situace nastává. Přidej citace.\n"
    "- Kombinovaný: Vyfiltruj banky splňující podmínku a mezi nimi srovnej výhodnost. Výsledek uveď jen pro ty nejlepší.\n\n"
    "🛑 Pravidla přesnosti:\n"
    "- Vycházej výhradně z úryvků z dokumentů v databázi (ChromaDB).\n"
    "- Nevymýšlej informace. Nepoužívej web ani obecné znalosti.\n"
    "- Nepřiřazuj informace k bankám, které je výslovně neuvádějí.\n"
    "- V odpovědi používej názvy bank přesně dle dokumentů:\n"
    "  • Hypoteky_KB.pdf → Komerční banka\n"
    "  • Hypoteky_mB.pdf → mBank\n"
    "  • Hypoteky_CS.pdf → Česká spořitelna\n"
    "  • Hypoteky_ČSOBHB.pdf → ČSOB Hypoteční banka\n"
    "  • Hypoteky_UCB.pdf → UniCredit Bank\n"
    "  • Hypoteky_OB.pdf → Oberbank AG\n"
    "  • Hypoteky_RB_bonita_podnikani.pdf → Raiffeisenbank\n\n"
    "♻️ Zaměnitelné výrazy:\n"
    "- „americká hypotéka“ = „neúčelový hypoteční úvěr“ = „neúčelová hypotéka“ = „neúčelová část hypotečního úvěru“\n"
    "- „účelová hypotéka“ není totéž jako „americká hypotéka“. Nezaměňuj tyto pojmy.\n"
    "  Pokud je v dotazu zmíněna americká hypotéka, ignoruj informace o účelových hypotékách.\n\n"
    "📋 Struktura odpovědi:\n"
    "- Použij přehledný formát ve stylu Markdown:\n"
    "  • Každou banku začni nadpisem třetí úrovně: ### 🏦 [Název banky]\n"
    "  • Každou část označ tučně: **Podmínky:**, **Výpočet:**, **Doložení:** apod.\n"
    "  • Podmínky a detaily strukturovaně ve formě odrážek: - ...\n"
    "  • Pokud existuje více oblastí, rozděl je logicky a vizuálně\n"
    "  • Na konec každého bloku přidej citaci: 📄 Citace: (dokument: <název>, strana: <číslo>, kapitola: <číslo>)\n\n"
    "🧠 Poznámka:\n"
    "- Interní úvahy (např. „Dotaz je výčtový“) nezobrazuj uživateli.\n"
    "- Odpověď začni rovnou užitečnou informací.\n"
    "  Například místo:\n"
    "  „Dotaz je výčtový. Uživatel se ptá, které banky akceptují výživné...“\n"
    "  napiš přímo:\n"
    "  „Banky, které akceptují výživné jako příjem žadatele:“\n"
            )
        },
        {
            "role": "user",
            "content": f"Dotaz: {dotaz}\n\nZde jsou úryvky z dokumentů:\n\n" +
                    "\n\n".join([f"{chunk}\nUmístění: {cit}" for chunk, cit in zip(relevant_chunks, citace)])
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )

    odpoved = response.choices[0].message.content
    odpoved_html = markdown.markdown(highlight_citations(odpoved))

    return templates.TemplateResponse("index.html", {"request": request, "result": odpoved_html, "dotaz": dotaz})
