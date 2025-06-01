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
    results = collection.query(query_embeddings=[embedding], n_results=15, include=["documents", "metadatas"])

    relevant_chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    citace = [
        f"(dokument: {meta.get('dokument', '?')}, strana: {meta.get('strana', '?')}, kapitola: {meta.get('kapitola') or '?'})"
        for meta in metadatas
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "Jsi expertní asistent na hypotéky a posuzování bonity klientů podle interních metodik bank. "
                "Odpovídej výhradně na základě úryvků. Pokud odpověď není obsažena, řekni to jasně. "
                "Uváděj citace jako (dokument: <název>, strana: <číslo>, kapitola: <číslo>)."
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
