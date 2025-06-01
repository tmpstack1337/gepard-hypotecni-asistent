import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import re

# Načtení modelu a databáze
model = SentenceTransformer("intfloat/multilingual-e5-large")
client = chromadb.PersistentClient(path="./chroma_db")
if "hypoteky_all" in [c.name for c in client.list_collections()]:
    client.delete_collection("hypoteky_all")
collection = client.create_collection(name="hypoteky_all")

# Parametry zpracování
chunk_size = 1200
chunk_overlap = 150

# Pomocná funkce: dělení textu na překrývající se úseky
def split_text(text, size, overlap):
    chunks = []
    i = 0
    while i < len(text):
        end = i + size
        chunks.append(text[i:end])
        i += size - overlap
    return chunks

# Pomocná funkce: určení banky z názvu nebo obsahu
def detect_bank(filename, text):
    patterns = ["ČSOB", "Česká spořitelna", "KB", "Moneta", "Raiffeisen", "UniCredit", "mBank", "Hypoteční banka"]
    for bank in patterns:
        if bank.lower() in filename.lower() or bank.lower() in text.lower():
            return bank
    return "Neznámá banka"

# Cesta ke složce se soubory
folder_path = "./metodiky_bank"

# Procházení souborů
for fname in os.listdir(folder_path):
    path = os.path.join(folder_path, fname)
    ext = os.path.splitext(fname)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                t = page.extract_text()
                if t:
                    text += t + "\n"

        elif ext == ".docx":
            doc = Document(path)
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif ext == ".xlsx":
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                for col in df.columns:
                    col_text = df[col].astype(str).str.cat(sep=" ", na_rep="")
                    text += col_text + "\n"

        else:
            print(f"❌ Nepodporovaný formát: {fname}")
            continue

        if not text.strip():
            print(f"⚠️ Prázdný obsah: {fname}")
            continue

        banka = detect_bank(fname, text)
        chunks = split_text(text, chunk_size, chunk_overlap)

        for i, chunk in enumerate(chunks):
            kapitola = "?"
            for line in chunk.splitlines():
                if line.strip().startswith(tuple(str(k) for k in range(1, 10))) and '.' in line:
                    kapitola = line.strip().split()[0]
                    break

            metadata = {
                "dokument": fname,
                "banka": banka,
                "kapitola": kapitola,
                "cast": i + 1,
                "strana": i + 1  # nově přidáno — lze vylepšit přesnější logikou dle formátu
            }
            embedding = model.encode(f"passage: {chunk}").tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[f"{fname}_{i}"]
            )
        print(f"✅ Zpracováno: {fname} ({banka}) — {len(chunks)} bloků")

    except Exception as e:
        print(f"❌ Chyba při zpracování {fname}: {e}")

print("\n✅ Znalostní databáze úspěšně rozšířena.")
input("\nStiskni Enter pro ukončení...")
