# === Mapování dokumentů na názvy bank ===
dokument_to_banka = {
    "Hypoteky_KB.pdf": "Komerční banka",
    "Hypoteky_CS.pdf": "Česká spořitelna",
    "Hypoteky_RB_bonita_podnikani.pdf": "Raiffeisen",
    "Hypoteky_UCB.pdf": "UniCredit",
    "Hypoteky_mB.pdf": "mBank",
    "Hypoteky_OB.pdf": "Oberbank",
    "Hypoteky_ČSOBHB.pdf": "ČSOB",
}

def extrahuj_banku(meta: dict) -> str:
    if meta.get("banka"):
        return meta["banka"].strip()
    doc_name = meta.get("document_source", "").strip()
    return dokument_to_banka.get(doc_name, "Neznámá banka")

def analyzuj_banky_z_fulltextu(metadatas: list[dict]) -> set[str]:
    return {extrahuj_banku(meta) for meta in metadatas}

def analyzuj_relevantni_banky_fulltextem(collection, hledane_slovo: str) -> set:
    """Vrátí množinu bank (nebo názvů dokumentů), které obsahují hledané slovo."""
    vsechny = collection.get(limit=None)
    banky = set()

    for doc, meta in zip(vsechny.get("documents", []), vsechny.get("metadatas", [])):
        if hledane_slovo in doc.lower():
            banka = meta.get("banka")
            if not banka:
                banka = meta.get("document_source", "neznámý").replace(".pdf", "")
            print(f"✅ Fulltext: {banka} – dokument: {meta.get('document_source')}")
            banky.add(banka.lower())

    return banky


def zjisti_banky_z_embeddingu(embedding_metadatas: list) -> set:
    """Vrátí množinu bank, které byly vráceny embedding dotazem."""
    return {
        meta.get("banka").lower()
        for meta in embedding_metadatas
        if meta.get("banka")
    }


def porovnej_pokryti(fulltext_banky: set, embedding_banky: set) -> set:
    """Vrátí banky, které embedding neobsáhl, ale fulltext je detekoval."""
    return fulltext_banky - embedding_banky
