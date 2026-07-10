import itertools
import random
from typing import List, Dict

random.seed(7) 


WIKI_SUBJECTS = [
    "el rio Parana", "la cordillera de los Andes", "el lago Nahuel Huapi",
    "la provincia de Misiones", "el desierto de Atacama", "la meseta patagonica",
    "el golfo San Matias", "la sierra de Cordoba", "el valle de Punilla",
    "la laguna Mar Chiquita", "el cerro Aconcagua", "la peninsula Valdes",
]
WIKI_ATTRIBUTES = [
    "se extiende por varios kilometros", "presenta un clima templado",
    "alberga una gran diversidad de especies", "fue declarado reserva natural",
    "atraviesa tres provincias", "tiene una altitud considerable",
    "es un destino turistico reconocido", "posee una flora autoctona variada",
    "registra fuertes vientos durante el invierno", "cuenta con varias rutas de acceso",
    "fue estudiado por geografos durante decadas", "limita con una zona protegida",
]

NEWS_DAYS = ["El lunes", "El martes", "El miercoles", "El jueves", "El viernes",
             "El sabado", "El domingo"]
NEWS_EVENTS = [
    "el dolar oficial subio levemente", "se registraron lluvias en la zona metropolitana",
    "el banco central no realizo cambios en la tasa", "el transito se vio demorado en accesos",
    "se informaron nuevas medidas economicas", "continuo la suba de los combustibles",
    "el mercado cerro sin grandes variaciones", "se esperan temperaturas estables",
    "se anuncio un nuevo cronograma de pagos", "el indice bursatil cerro en terreno mixto",
    "se reforzo el operativo de control en rutas", "bajaron las exportaciones del sector",
]

BOOK_CHARACTERS = [
    "el viejo molinero", "la nina del bosque", "el zorro plateado",
    "la reina sin corona", "el marinero ciego", "el ultimo jardinero",
    "el relojero mudo", "la sombra del campanario", "el pastor de las nubes",
    "la tejedora de cuentos", "el guardian del puente", "la voz del rio",
]
BOOK_PHRASES = [
    "no hay nada que temer en la noche", "el camino siempre vuelve a casa",
    "el silencio guarda mas secretos que las palabras", "todo rio encuentra su mar",
    "la espera tambien es una forma de viaje", "nadie escapa del eco de su nombre",
    "el fuego recuerda lo que el viento olvida", "una promesa pesa mas que una piedra",
    "el tiempo no perdona pero tampoco olvida", "cada puerta cerrada abre otra pregunta",
    "la memoria es el unico mapa que no se rompe", "quien siembra silencio cosecha dudas",
]


def _gen_a(subjects, attributes, n, template, repeats=(3, 4, 5)):
    out = []
    combos = list(itertools.product(subjects, attributes, repeats))
    random.shuffle(combos)
    seen = set()
    for subj, attr, rep in combos:
        if len(out) >= n:
            break
        text = (template.format(subj=subj.capitalize() if not subj[0].isupper() else subj,
                                 attr=attr) + " ") * rep
        text = text.strip()
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out[:n]


def _gen_wiki_a(n: int) -> List[str]:
    return _gen_a(WIKI_SUBJECTS, WIKI_ATTRIBUTES, n, "{subj} {attr}.")


def _gen_news_a(n: int) -> List[str]:
    out = []
    combos = list(itertools.product(NEWS_DAYS, NEWS_EVENTS, (3, 4, 5)))
    random.shuffle(combos)
    seen = set()
    for day, event, rep in combos:
        if len(out) >= n:
            break
        text = (f"{day}, {event}. " * rep).strip()
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out[:n]


def _gen_book_a(n: int) -> List[str]:
    out = []
    combos = list(itertools.product(BOOK_CHARACTERS, BOOK_PHRASES, (3, 4, 5)))
    random.shuffle(combos)
    seen = set()
    for char, phrase, rep in combos:
        if len(out) >= n:
            break
        text = (f"Y entonces {char} dijo: \"{phrase}\". " * rep).strip()
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out[:n]


def _gen_wiki_b(n: int) -> List[str]:
    out = []
    for _ in range(n):
        order = WIKI_SUBJECTS.copy()
        random.shuffle(order)
        block = []
        for round_ in range(3):
            for subj in order:
                idx = (WIKI_SUBJECTS.index(subj) + round_) % len(WIKI_ATTRIBUTES)
                block.append(f"{subj.capitalize()} {WIKI_ATTRIBUTES[idx]}.")
        out.append(" ".join(block))
    return out


def _gen_news_b(n: int) -> List[str]:
    out = []
    for _ in range(n):
        order = NEWS_EVENTS.copy()
        random.shuffle(order)
        block = []
        for round_ in range(3):
            for i, event in enumerate(order):
                day = NEWS_DAYS[(i + round_) % len(NEWS_DAYS)]
                block.append(f"{day}, {event}.")
        out.append(" ".join(block))
    return out


def _gen_book_b(n: int) -> List[str]:
    out = []
    for _ in range(n):
        order = BOOK_CHARACTERS.copy()
        random.shuffle(order)
        block = []
        for round_ in range(3):
            for i, char in enumerate(order):
                phrase = BOOK_PHRASES[(i + round_) % len(BOOK_PHRASES)]
                block.append(f"Y entonces {char} dijo: \"{phrase}\".")
        out.append(" ".join(block))
    return out


GENERATORS_A = {"wiki": _gen_wiki_a, "news": _gen_news_a, "book": _gen_book_a}
GENERATORS_B = {"wiki": _gen_wiki_b, "news": _gen_news_b, "book": _gen_book_b}


def build_generalization_sets(n_a: int = 100, n_b: int = 10,
                               domains=("wiki", "news", "book")) -> Dict[str, Dict[str, List[str]]]:
    """
    Devuelve:
        {
          "wiki": {"A": [...100 textos...], "B": [...10 textos...]},
          "news": {...},
          "book": {...},
        }
    Cada texto se parte en prefijo/continuacion exactamente igual que cualquier otra
    muestra (ver prefixing.py) y se corre por el MISMO pipeline que los miembros reales
    -- la unica diferencia es que ya sabemos, por construccion, label=0 (no-miembro).
    Usamos un solo "domain=wiki" tanto para WikiMIA como para WikiMIA-24, ya que el
    estilo de texto es el mismo (enciclopedico); lo que cambia entre esos dos datasets es
    el corte temporal del propio benchmark, no el dominio de redaccion.
    """
    result = {}
    for domain in domains:
        result[domain] = {
            "A": GENERATORS_A[domain](n_a),
            "B": GENERATORS_B[domain](n_b),
        }
    return result


if __name__ == "__main__":
    sets = build_generalization_sets(n_a=100, n_b=10)
    for domain, d in sets.items():
        print(f"{domain}: Set A = {len(d['A'])} textos, Set B = {len(d['B'])} textos")
        print("  Ejemplo A:", d["A"][0][:120], "...")
        print("  Ejemplo B:", d["B"][0][:120], "...")
