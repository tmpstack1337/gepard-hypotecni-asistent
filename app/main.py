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
            fallback = collection.get(where={"document_source": {"$contains": requested_doc}})
            if fallback and fallback.get("documents"):
                results["documents"][0].append(fallback["documents"][0][0])
                results["metadatas"][0].append(fallback["metadatas"][0][0])

    relevant_chunks = results["documents"][0]
    citace = [meta.get("location", "") for meta in results["metadatas"][0]]

    messages = [
    {
        "role": "system",
        "content": (
            "Jsi expertní asistent na hypotéky a posuzování bonity klientů podle interních metodik bank.\n\n"
    "Odpovídej výhradně na základě úryvků z dodaných dokumentů. Pokud odpověď není v úryvcích výslovně obsažena, řekni to jasně.\n\n"
    "Pokud ale dotaz není výslovně zodpovězen, ale lze jej logicky odvodit na základě výpočtových metod, posuzovacích pravidel nebo postupů popsaných v dokumentech, uveď odpověď, zdůvodni ji a připoj citaci.\n\n"
    "Pokud dotaz obsahuje konkrétní pojem (např. typ příjmu, forma zaměstnání, struktura smlouvy), ale dokumenty tento pojem výslovně nezmiňují, avšak popisují výpočet nebo pravidlo s ním logicky související, považuj to za platnou informaci.\n\n"
    "Příklad: Pokud dokument uvádí, že banka pro výpočet příjmu používá obrat dělený 12 nebo průměr plateb na účtu, považuj to za důkaz, že banka akceptuje příjem z obratu.\n\n"
    "U dotazů typu „Které banky akceptují…“ analyzuj všechny dostupné úryvky. Pokud některá z bank v dokumentu výslovně nebo logicky akceptuje daný případ, uveď ji jako relevantní.\n\n"
    "Pokud je v dotazu uveden konkrétní název banky, prioritně prohledej dokumenty, které jsou s touto bankou přímo spojeny.\n"
    "Ostatní dokumenty ber v úvahu pouze tehdy, pokud banka nemá vlastní dokument, nebo je v daném dokumentu výslovně jmenována.\n"
    "Nikdy nepřiřazuj informace mezi bankami jen na základě tematické podobnosti.\n\n"
    "Při odpovědi uváděj název banky přesně podle dokumentu, z něhož čerpáš. Například: Hypoteky_KB.pdf → Komerční banka, Hypoteky_mB.pdf → mBank.\n"
    "Nepřiřazuj banku, která se v dokumentu nevyskytuje nebo s ním nesouvisí.\n\n"
    "U výčtových dotazů uveď každou relevantní banku zvlášť, v samostatném bloku se jménem banky, shrnutím a citací.\n"
    "Nespoléhej jen na top výsledek – agreguj napříč všemi dokumenty.\n\n"
    "Pokud je dotaz srovnávací (např. 'nejvyšší LTV', 'nejdelší splatnost'), porovnej všechny dostupné hodnoty a ve výsledku uveď pouze tu nejvýhodnější. Pokud je více bank se stejnou hodnotou, uveď je všechny jako rovnocenné.\n"
    "Nikdy nevynechávej banku, pokud její hodnota odpovídá dotazu.\n\n"
    "Při vyhodnocování dotazů považuj za rovnocenné následující pojmy:\n"
    "- „americká hypotéka“ = „neúčelový hypoteční úvěr“ = „neúčelová hypotéka“ = „neúčelová část hypotečního úvěru“\n"
    "- „účelová hypotéka“ ≠ „americká hypotéka“ (nejsou zaměnitelné)\n\n"
    "Pokud je v dotazu uvedeno 'americká hypotéka', ignoruj zmínky o účelových úvěrech.\n\n"
    "Struktura odpovědi:\n"
    "- Název banky\n"
    "- Shrnutí, jak se daná věc posuzuje nebo vypočítává\n"
    "- Přesná citace (dokument: <název>, strana: <číslo>, kapitola: <číslo>)\n\n"
    "Používej pouze dokumenty z interní databáze (chroma_db). Nepřidávej vymyšlené informace. Neodkazuj na weby ani externí zdroje."
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
