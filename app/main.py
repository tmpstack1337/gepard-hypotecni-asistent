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
    def replace(match):
        citation = match.group(0)

        doc_match = re.search(r'dokument: ([^,]+)', citation)
        page_match = re.search(r'strana: (\d+)', citation)

        if not doc_match:
            return citation

        filename = doc_match.group(1).strip()
        page_number = page_match.group(1) if page_match else "1"

        safe_filename = urllib.parse.quote(filename)

        if filename.lower().endswith(".pdf"):
            url = f"/metodiky/{safe_filename}#page={page_number}"
        else:
            url = f"/metodiky/{safe_filename}"

        return f"<a href='{url}' target='_blank' class='citation'>{citation}</a>"

    pattern = r"\(dokument: [^,]+, strana: \d+(?:, kapitola: [^)]+)?\)"
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
    embedding = model.encode(f"query: {dotaz}").tolist()
    results = collection.query(query_embeddings=[embedding], n_results=70, include=["documents", "metadatas"])

    relevant_chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    citace = [
        f"(dokument: {meta.get('dokument', '?')}, strana: {meta.get('strana', '?')}, kapitola: {meta.get('kapitola') or '?'})"
        for meta in metadatas
    ]

    messages = [
        {
            "role": "system",
            "content": ("Jsi expertní asistent na hypotéky a posuzování bonity klientů podle interních metodik bank.\n\n"
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
    "Při generování odpovědi musí být název banky uveden v souladu s dokumentem, ze kterého citace pochází. Pokud je citace z dokumentu Hypoteky_mB.pdf, název banky musí být uveden jako mBank. Pokud je z Hypoteky_KB.pdf, pak Komerční banka. Nikdy nepřiřazuj banku, která se v dokumentu nevyskytuje nebo s ním nesouvisí.\n\n"
"U dotazů typu 'Které banky akceptují...' důsledně analyzuj všechny dostupné úryvky. Pokud více bank splňuje podmínky uvedené v dotazu, vypiš každou z nich samostatně v samostatném bloku s názvem banky, shrnutím a citací. Nespoléhej pouze na nejrelevantnější výsledek – agreguj napříč všemi dokumenty. Pokud je výživné akceptováno více bankami, uveď všechny, které to výslovně nebo logicky potvrzují."
    "Pokud je dotaz srovnávacího typu (např. „Která banka nabízí nejvyšší LTV…“, „Která banka umožňuje nejdelší splatnost…“), vyhledej a porovnej všechny relevantní hodnoty ve všech dostupných úryvcích.\n"
    "Ve výsledku uveď pouze tu banku (nebo banky), která nabízí nejvyšší (nebo nejvýhodnější) hodnotu.\n"
    "Neuváděj všechny banky – pouze tu nejlepší podle dotazu. Pokud je více bank se stejnou hodnotou, uveď je všechny, ale jasně označ, že jsou rovnocenné.\n"
    "Nikdy nevynechávej banku, pokud se v úryvku objevuje s hodnotou, která odpovídá dotazu.\n\n"
    "Považuj tyto pojmy za rovnocenné a zaměnitelné při vyhodnocování dotazu:\n"
    "- „americká hypotéka“ = „neúčelový hypoteční úvěr“ = „neúčelová hypotéka“ = „neúčelová část hypotečního úvěru“\n"
    "- „účelová hypotéka“ ≠ „americká hypotéka“ (nejsou zaměnitelné)\n"
    "Pokud je v dotazu uvedeno „americká hypotéka“, zahrnuj pouze informace týkající se neúčelových hypotečních úvěrů a ignoruj jakékoliv zmínky o účelových hypotékách.\n\n"
    "Struktura odpovědi:\n"
    "- Název banky\n"
    "- Shrnutí, jak se daná věc posuzuje nebo vypočítává\n"
    "- Přesná citace (dokument: <název>, strana: <číslo>, kapitola: <číslo>)\n\n"
    "Nepoužívej žádné vymyšlené informace. Neodkazuj na web ani na neexistující zdroje."
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
