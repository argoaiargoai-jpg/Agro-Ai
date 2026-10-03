"""Output safety helpers: remove invented pesticide dosages; redact secrets from text before logging."""
import re

_NUM = r"\d+(?:[.,]\d+)?"
_MASS_VOL = r"(?:ml|mL|cc|g|kg|mg|oz|lb|lbs|tsp|tbsp|l|L)"
_PER = r"(?:ml|mL|l|L|litre|liter|litres|liters|gal|gallon|gallons|ha|hectare|hectares|acre|acres|plant|plants|m2|m²|sq\.?\s?m)"
DOSAGE_PATTERNS = [
    re.compile(rf"{_NUM}\s*{_MASS_VOL}\s*(?:/|per)\s*(?:{_NUM})?\s*{_PER}\b", re.I),                 # 2 g/L, 500 ml per hectare
    re.compile(rf"{_NUM}\s*{_MASS_VOL}\s+(?:of\s+[\w\- ]{{1,30}}\s+)?(?:in|into|with|per)\s+{_NUM}\s*(?:l|litre|liter|litres|liters|gal|gallon|gallons)\b", re.I),  # 30 g in 10 L water
    re.compile(rf"{_NUM}\s*(?:%|percent)\s+(?:solution|concentration|dilution|w/v|v/v|wp|ec)\b", re.I),   # 2% solution
    re.compile(rf"\b{_NUM}\s*(?:ppm|mg/l|mg/kg)\b", re.I),
    re.compile(r"\b\d+\s*:\s*\d+\s+(?:dilution|ratio|mix)\b", re.I),                                      # 1:100 dilution
    re.compile(rf"{_NUM}\s*(?:ml|mL|g|kg)\s*(?:/|per)\s*(?:100\s*)?(?:l|L|litre|liter)\b", re.I),
]
DOSAGE_NOTE = "Specific application rates were left out on purpose: follow the product label, crop variety, local regulations and local agricultural guidance."


def strip_dosages(items: list[str]) -> tuple[list[str], int]:
    kept, removed = [], 0
    for it in items:
        if any(p.search(it) for p in DOSAGE_PATTERNS):
            removed += 1
        else:
            kept.append(it)
    return kept, removed


def redact(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s and len(s) >= 6:
            text = text.replace(s, "***")
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "***", text)       # Google API key shape, defence in depth
