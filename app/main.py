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

import re
from collections import defaultdict

def agreguj_banky_v_odpovedi(odpoved_markdown):
    """
    Najde všechny bloky typu "### 🏦 Název banky" ... až do další banky nebo konce.
    Slije Podmínky a Citace od stejné banky do jednoho společného bloku.
    """
    banky = defaultdict(list)

    # Najdi bloky od ### 🏦 až po další ### 🏦 nebo konec
    pattern = r"(### 🏦[^\n]+(?:\n.*?)+?)(?=\n### 🏦|\Z)"
    for blok in re.findall(pattern, odpoved_markdown, flags=re.DOTALL):
        # Název banky
        nazev_match = re.match(r"### 🏦\s*([^\n]+)", blok)
        if not nazev_match:
            continue
        banka = nazev_match.group(1).strip()

        # Oddělíme "Podmínky", "Doložení" apod. a citace
        podminky = []
        citace = []
        lines = blok.split("\n")
        for line in lines[1:]:  # [1:] protože [0] je název banky
            # Pokud je to citace, schováme ji
            if line.strip().startswith("📄 Citace:"):
                citace.append(line.strip())
            # Ostatní necháme jako podmínky (včetně případných „Doložení“ atd.)
            elif line.strip():
                podminky.append(line.rstrip())
        # Uložíme do dictu
        banky[banka].append({
            "podminky": podminky,
            "citace": citace
        })

    # Finální slévání – pro každou banku vypiš všechny podmínky i citace
    vystup = []
    for banka, bloky in banky.items():
        vystup.append(f"### 🏦 {banka}")
        vsechny_podminky = []
        vsechny_citace = []
        for blok in bloky:
            vsechny_podminky.extend(blok["podminky"])
            vsechny_citace.extend(blok["citace"])
        # Odstraníme duplicitní řádky (zachová pořadí)
        podminky_unique = []
        for p in vsechny_podminky:
            if p not in podminky_unique:
                podminky_unique.append(p)
        citace_unique = []
        for c in vsechny_citace:
            if c not in citace_unique:
                citace_unique.append(c)
        vystup.extend(podminky_unique)
        vystup.extend(citace_unique)
        vystup.append("")  # mezera mezi bankami
    return "\n".join(vystup).strip()


# Slovník převodů všech možných názvů na oficiální název banky:
BANK_NAME_MAP = {
    "komercnibanka": "Komerční banka",
    "kb": "Komerční banka",
    "ceskasporitelna": "Česká spořitelna",
    "cs": "Česká spořitelna",
    "csob": "ČSOB Hypoteční banka",
    "csobhypotecnibanka": "ČSOB Hypoteční banka",
    "mbank": "mBank",
    "oberbank": "Oberbank AG",
    "oberbankag": "Oberbank AG",
    "unicreditbank": "UniCredit Bank",
    "ucb": "UniCredit Bank",
    "raiffeisenbank": "Raiffeisenbank",
    "rb": "Raiffeisenbank",
    # přidej další podle potřeby
}

def normalizuj_nazev_banky(banka_raw):
    import unicodedata
    if not banka_raw or "neznámá" in banka_raw.lower():
        return "Neznámá banka"
    banka = banka_raw.strip().lower()
    banka = ''.join(
        c for c in unicodedata.normalize('NFD', banka)
        if unicodedata.category(c) != 'Mn'
    )
    banka = banka.replace(" ", "")
    return BANK_NAME_MAP.get(banka, banka_raw.strip())

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

    # Krok 2: příprava chunků a metadat
    relevant_chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    for i, meta in enumerate(metadatas[:3]):
        print(f"Metadata {i}: {meta}")

    # === Agregace chunků podle banky (už žádné duplicity) ===
    banky_map = {}
    for chunk, meta in zip(relevant_chunks, metadatas):
        banka_raw = meta.get("banka", "Neznámá banka")
        banka_nazev = normalizuj_nazev_banky(banka_raw)
        if banka_nazev not in banky_map:
            banky_map[banka_nazev] = {
                "chunks": [],
                "citace": []
            }
        banky_map[banka_nazev]["chunks"].append(chunk)
        banky_map[banka_nazev]["citace"].append(
            f"(dokument: {meta.get('document_source', '?')}, strana: {meta.get('strana', '?')}, kapitola: {meta.get('kapitola', '?')})"
        )

    # === Odpovědi pro každou banku právě jednou ===
    pridane_banky = set()
    odpovedi_po_bankach = []

    for banka_nazev, banka_data in banky_map.items():
        chunks_banky = banka_data["chunks"]
        citace_banky = banka_data["citace"]

        if not chunks_banky:
            continue

        # GPT prompt pro výběr nejrelevantnějšího chunku
        select_prompt = [
            {
                "role": "system",
                "content": (
                    "Jsi asistent pro výběr nejrelevantnějšího úryvku textu k danému dotazu. "
                    "Dostaneš dotaz a několik úryvků s citacemi. Vyber jen ten jeden úryvek, "
                    "který je pro zodpovězení dotazu nejrelevantnější. Pokud je to možné, upřednostni chunk, "
                    "který obsahuje nejvíce konkrétních informací k dotazu. V odpovědi vypiš přesně vybraný úryvek "
                    "a jeho citaci ve formátu: <chunk>\nUmístění: <citace>."
                )
            },
            {
                "role": "user",
                "content":
                    f"Dotaz: {dotaz}\n\nÚryvky:\n\n" +
                    "\n\n".join([f"{chunk}\nUmístění: {cit}" for chunk, cit in zip(chunks_banky, citace_banky)])
            }
        ]

        select_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=select_prompt,
            temperature=0
        )
        vybrany_chunk_a_citace = select_response.choices[0].message.content.strip()

        # Hlavní prompt pro banku (zůstává stejný)
        messages = [
            {
                "role": "system",
                "content": (
                    "Jsi expertní asistent na hypotéky a posuzování bonity klientů podle interních metodik bank.\n\n"
                        "🔍 Nejprve zjisti, co je vstupem uživatele:\n"
                        "1. Pokud jde o plnohodnotný dotaz, klasifikuj ho interně do jedné z těchto kategorií:\n"
                        "   - výčtový\n"
                        "   - Dotazy typu „které banky…“ vždy považuj za výčtové bez ohledu na další strukturu dotazu.\n"
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
                        "  ➕ Pokud dotaz směřuje na to, **které banky něco umožňují, akceptují, podporují, tolerují nebo zohledňují**, vždy jej považuj za výčtový – i když se zdá být podmínkový nebo faktický.\n"
                        "- Srovnávací: Porovnej hodnoty napříč bankami a uveď pouze tu nejlepší (nebo několik s nejvyšší hodnotou).\n"
                        "- Faktický: Odpověz přesně a s citací. Pokud informace chybí, napiš to jasně.\n"
                        "- Podmínkový: Popiš okolnosti, za kterých situace nastává. Přidej citace.\n"
                        "  ➕ Pokud dotaz obsahuje podmínku („pokud...“, „za jakých podmínek...“), ale cílí na více subjektů (např. „které banky“), nejprve vyfiltruj všechny relevantní banky jako ve výčtovém dotazu a pak u každé z nich uveď podmínky.\n"
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
                "content": f"Dotaz: {dotaz}\n\nZde je nejrelevantnější úryvek pro banku {banka_nazev}:\n\n{vybrany_chunk_a_citace}"
            }
        ]

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )
        if banka_nazev in pridane_banky:
                continue
        
        odpovedi_po_bankach.append(response.choices[0].message.content)
        pridane_banky.add(banka_nazev)
        
    # Krok 5: Vytvoř HTML výstup
    final_answer = "\n\n".join(odpovedi_po_bankach)
    final_answer = agreguj_banky_v_odpovedi(final_answer)   # <-- agregace bloků
    odpoved_html = markdown.markdown(highlight_citations(final_answer))

    return templates.TemplateResponse("index.html", {"request": request, "result": odpoved_html, "dotaz": dotaz})
