
def analyzuj_relevantni_banky_fulltextem(collection, hledane_slovo: str) -> set:
    """Vrátí množinu bank, které ve svých dokumentech obsahují hledané slovo."""
    vsechny = collection.get()
    banky = set()

    for doc, meta in zip(vsechny.get("documents", []), vsechny.get("metadatas", [])):
        if hledane_slovo in doc.lower():
            banka = meta.get("banka")
            if banka:
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
