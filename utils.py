import re


def clean_price(raw: str) -> int | None:
    """Convert PakWheels price strings to integer PKR."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.replace(",", "").strip()

    lac = re.search(r"PKR\s*([\d.]+)\s*lacs?", s, re.IGNORECASE)
    if lac:
        return round(float(lac.group(1)) * 100_000)

    crore = re.search(r"PKR\s*([\d.]+)\s*crore", s, re.IGNORECASE)
    if crore:
        return round(float(crore.group(1)) * 10_000_000)

    num = re.search(r"[\d.]+", s)
    if num:
        return round(float(num.group()))

    return None


def clean_mileage(raw: str) -> int | None:
    """Strip 'km' and commas, return integer km."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.replace(",", "").strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def parse_features(raw: str) -> list[str]:
    """Parse Python-list-style string to list."""
    if not raw or not isinstance(raw, str):
        return []
    cleaned = raw.strip().lstrip("[").rstrip("]")
    return [f.strip().strip("'\"") for f in cleaned.split(",") if f.strip().strip("'\"")]


def extract_make_model(nam: str) -> tuple[str, str, str]:
    KNOWN_MAKES = [
        "Toyota", "Honda", "Suzuki", "Daihatsu", "Hyundai", "Kia", "Nissan",
        "Mitsubishi", "Mercedes", "BMW", "Audi", "Ford", "Chevrolet", "Jeep",
        "Land Rover", "Range Rover", "Isuzu", "Fiat", "Subaru", "Mazda",
        "Volkswagen", "Changan", "Proton", "MG", "BAIC", "FAW", "Haval",
        "Prince", "United", "Regal", "Adam", "Renault", "Peugeot",
    ]
    if not nam:
        return "Unknown", "Unknown", ""

    for make in KNOWN_MAKES:
        if nam.lower().startswith(make.lower()):
            rest = nam[len(make):].strip()
            parts = rest.split()
            model = parts[0] if parts else "Unknown"
            variant = " ".join(parts[1:]).strip()
            variant = re.sub(r"\d{4}$", "", variant).strip()
            return make, model, variant

    parts = nam.split()
    make = parts[0] if parts else "Unknown"
    model = parts[1] if len(parts) > 1 else "Unknown"
    variant = " ".join(parts[2:])
    return make, model, variant
