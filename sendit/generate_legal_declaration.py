"""Generate and validate a legal train shipment declaration from local rules."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

try:
    from loguru import logger
except ImportError as error:
    raise SystemExit(
        "Brak zaleznosci 'loguru'. Zainstaluj ja poleceniem: pip install loguru"
    ) from error


STANDARD_TRAIN_CAPACITY_KG = 1000
EXTRA_WAGON_CAPACITY_KG = 500
EXTRA_WAGON_COST_PP = 55

BASE_FEE_PP = {
    "A": 0,
    "B": 0,
    "C": 2,
    "D": 5,
    "E": 10,
}

WEIGHT_BRACKETS = [
    (5, 0.5),
    (25, 1),
    (100, 2),
    (500, 3),
    (1000, 5),
]
ABOVE_1000_RATE = 7

ROUTE_DISTANCE_KM = {
    "GDANSK-ZARNOWIEC": 60,
}

REGIONAL_BOUNDARY_COUNT = {
    "GDANSK-ZARNOWIEC": 0,
}

ALLOWED_SPECIAL_DESTINATIONS = {
    "ZARNOWIEC": {"allowed_categories": {"A", "B"}},
}

STRATEGIC_CONTENT_KEYWORDS = {
    "ogniwa paliwowe",
    "moduly komunikacyjne",
    "czesci zamienne do automatow kontrolnych",
    "materialy do naprawy torow",
    "podzespoly elektroniczne",
}

MEDICAL_CONTENT_KEYWORDS = {
    "leki",
    "szczepionki",
    "sprzet medyczny",
    "probki laboratoryjne",
    "srodki dezynfekcyjne",
}

PROHIBITED_CONTENT_KEYWORDS = {
    "substancje radioaktywne",
}


@dataclass(slots=True)
class ShipmentInput:
    sender_id: str
    origin: str
    destination: str
    declared_mass_kg: int
    budget_pp: int
    contents: str
    special_notes: str
    date: str | None = None
    route_code: str | None = None
    wdp_override: int | None = None
    sender_authorized_for_category_a: bool = False
    sender_authorized_medical: bool = False
    confirmed_legal_basis: bool = False
    route_reopening_expected: bool = False
    generate_pending_draft: bool = False


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    status: str
    declaration_text: str | None
    errors: list[str]
    warnings: list[str]
    computed: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generuje deklaracje przewozowa tylko dla legalnie dopuszczalnych "
            "przypadkow, zgodnie z lokalna dokumentacja."
        )
    )
    parser.add_argument(
        "--shipment-file",
        type=Path,
        required=True,
        help="Plik JSON z danymi przesylki.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("legal_output"),
        help="Katalog wyjsciowy. Domyslnie: legal_output",
    )
    return parser.parse_args()


def resolve_path(path: Path | str, base_dir: Path) -> Path:
    normalized = path if isinstance(path, Path) else Path(path)
    return normalized if normalized.is_absolute() else (base_dir / normalized)


def normalize_text(value: str) -> str:
    replacements = {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ż": "z",
        "ź": "z",
    }
    normalized = value.strip().lower()
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())


def load_shipment(path: Path) -> ShipmentInput:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ShipmentInput(**data)


def infer_category(shipment: ShipmentInput) -> tuple[str | None, list[str]]:
    normalized_contents = normalize_text(shipment.contents)
    reasons: list[str] = []

    if any(keyword in normalized_contents for keyword in PROHIBITED_CONTENT_KEYWORDS):
        return None, ["Zawartosc wyglada na kategorie zakazana (X)."]

    if any(keyword in normalized_contents for keyword in STRATEGIC_CONTENT_KEYWORDS):
        reasons.append("Zawartosc pasuje do kategorii A (strategiczna).")
        return "A", reasons

    if any(keyword in normalized_contents for keyword in MEDICAL_CONTENT_KEYWORDS):
        reasons.append("Zawartosc pasuje do kategorii B (medyczna).")
        return "B", reasons

    reasons.append("Nie udalo sie jednoznacznie przypisac kategorii z tresci ladunku.")
    return None, reasons


def compute_weight_fee(mass_kg: int, category: str) -> float:
    if category in {"A", "B"}:
        return 0

    remaining = mass_kg
    total = 0.0
    lower_bound = 0
    for upper_bound, rate in WEIGHT_BRACKETS:
        if remaining <= 0:
            break
        bracket_width = upper_bound - lower_bound
        in_bracket = min(remaining, bracket_width)
        total += in_bracket * rate
        remaining -= in_bracket
        lower_bound = upper_bound

    if remaining > 0:
        total += remaining * ABOVE_1000_RATE

    return total


def compute_extra_wagons(mass_kg: int, category: str) -> tuple[int, int]:
    if mass_kg <= STANDARD_TRAIN_CAPACITY_KG:
        return 0, 0

    extra_mass = mass_kg - STANDARD_TRAIN_CAPACITY_KG
    wagons = math.ceil(extra_mass / EXTRA_WAGON_CAPACITY_KG)
    if category in {"A", "B"}:
        return wagons, 0
    return wagons, wagons * EXTRA_WAGON_COST_PP


def compute_route_fee(route_code: str, category: str) -> int:
    if category in {"A", "B"}:
        return 0

    distance = ROUTE_DISTANCE_KM[route_code]
    boundaries = REGIONAL_BOUNDARY_COUNT[route_code]
    if boundaries == 0:
        rate = 1
    elif boundaries == 1:
        rate = 2
    else:
        rate = 3
    return math.ceil(distance / 100 * rate)


def determine_route_code(shipment: ShipmentInput) -> str | None:
    if shipment.route_code:
        return shipment.route_code

    origin = normalize_text(shipment.origin).upper()
    destination = normalize_text(shipment.destination).upper()
    return f"{origin}-{destination}"


def validate_shipment(shipment: ShipmentInput) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    blocking_errors: list[str] = []

    if shipment.special_notes.strip():
        warnings.append("Pole 'special_notes' nie jest puste; deklaracja uzyje tej wartosci.")

    route_code = determine_route_code(shipment)
    if route_code not in ROUTE_DISTANCE_KM:
        errors.append("Brak znanego kodu lub dystansu trasy w lokalnej dokumentacji.")

    category, category_reasons = infer_category(shipment)
    warnings.extend(category_reasons)
    if not category:
        errors.append("Nie da sie legalnie i jednoznacznie ustalic kategorii przesylki.")
        category = "UNKNOWN"

    destination_key = normalize_text(shipment.destination).upper()
    special_destination = ALLOWED_SPECIAL_DESTINATIONS.get(destination_key)
    if special_destination and category not in special_destination["allowed_categories"]:
        blocking_errors.append(
            f"Trasy do {shipment.destination} sa dopuszczone w dokumentacji tylko dla kategorii A lub B."
        )

    if category == "A" and not shipment.sender_authorized_for_category_a:
        blocking_errors.append("Kategoria A wymaga jednostki autoryzowanej przez System.")
    if category == "B" and not shipment.sender_authorized_medical:
        blocking_errors.append("Kategoria B wymaga placowki medycznej z aktualna autoryzacja.")

    if not shipment.confirmed_legal_basis:
        blocking_errors.append("Brak potwierdzenia legalnej podstawy nadania w danych wejsciowych.")

    if len(shipment.contents) > 200:
        errors.append("Opis zawartosci przekracza limit 200 znakow.")

    if shipment.declared_mass_kg <= 0:
        errors.append("Masa musi byc dodatnia.")

    base_fee = BASE_FEE_PP.get(category, 0)
    weight_fee = compute_weight_fee(shipment.declared_mass_kg, category) if category in BASE_FEE_PP else 0
    computed_wdp, extra_wagon_fee = (
        compute_extra_wagons(shipment.declared_mass_kg, category)
        if category in BASE_FEE_PP
        else (0, 0)
    )
    route_fee = compute_route_fee(route_code, category) if route_code and category in BASE_FEE_PP else 0
    total_fee = int(base_fee + weight_fee + route_fee + extra_wagon_fee)

    if shipment.wdp_override is not None and shipment.wdp_override != computed_wdp:
        warnings.append(
            f"WDP z wejscia ({shipment.wdp_override}) rozni sie od wyliczenia ({computed_wdp}); uzyte zostanie wyliczenie."
        )

    if total_fee > shipment.budget_pp:
        blocking_errors.append(
            f"Budzet {shipment.budget_pp} PP nie pokrywa wyliczonej oplaty {total_fee} PP."
        )

    declaration_text: str | None = None
    status = "REJECTED"
    errors.extend(blocking_errors)

    if not blocking_errors and route_code:
        declaration_date = shipment.date or date.today().isoformat()
        special_notes = shipment.special_notes.strip()
        declaration_text = format_declaration(
            declaration_date=declaration_date,
            shipment=shipment,
            route_code=route_code,
            category=category,
            wdp=computed_wdp,
            total_fee=total_fee,
            special_notes=special_notes,
        )
        status = "READY"
    elif shipment.route_reopening_expected and shipment.generate_pending_draft and route_code:
        declaration_date = shipment.date or date.today().isoformat()
        special_notes = shipment.special_notes.strip()
        declaration_text = format_declaration(
            declaration_date=declaration_date,
            shipment=shipment,
            route_code=route_code,
            category=category,
            wdp=computed_wdp,
            total_fee=total_fee,
            special_notes=special_notes,
        )
        warnings.append(
            "Wygenerowano jedynie draft oczekujacy na ponowna walidacje po otwarciu trasy lub uzyskaniu brakujacych uprawnien."
        )
        status = "PENDING_ROUTE_REOPEN"

    computed = {
        "route_code": route_code,
        "category": category,
        "base_fee_pp": base_fee,
        "weight_fee_pp": weight_fee,
        "route_fee_pp": route_fee,
        "extra_wagon_fee_pp": extra_wagon_fee,
        "wdp": computed_wdp,
        "total_fee_pp": total_fee,
    }

    return ValidationResult(
        is_valid=not blocking_errors,
        status=status,
        declaration_text=declaration_text,
        errors=errors,
        warnings=warnings,
        computed=computed,
    )


def format_declaration(
    *,
    declaration_date: str,
    shipment: ShipmentInput,
    route_code: str,
    category: str,
    wdp: int,
    total_fee: int,
    special_notes: str,
) -> str:
    return (
        "SYSTEM PRZESYŁEK KONDUKTORSKICH - DEKLARACJA ZAWARTOŚCI\n"
        "======================================================\n"
        f"DATA: {declaration_date}\n"
        f"PUNKT NADAWCZY: {shipment.origin}\n"
        "------------------------------------------------------\n"
        f"NADAWCA: {shipment.sender_id}\n"
        f"PUNKT DOCELOWY: {shipment.destination}\n"
        f"TRASA: {route_code}\n"
        "------------------------------------------------------\n"
        f"KATEGORIA PRZESYŁKI: {category}\n"
        "------------------------------------------------------\n"
        f"OPIS ZAWARTOŚCI (max 200 znaków): {shipment.contents}\n"
        "------------------------------------------------------\n"
        f"DEKLAROWANA MASA (kg): {shipment.declared_mass_kg}\n"
        "------------------------------------------------------\n"
        f"WDP: {wdp}\n"
        "------------------------------------------------------\n"
        f"UWAGI SPECJALNE: {special_notes}\n"
        "------------------------------------------------------\n"
        f"KWOTA DO ZAPŁATY: {total_fee} PP\n"
        "------------------------------------------------------\n"
        "OŚWIADCZAM, ŻE PODANE INFORMACJE SĄ PRAWDZIWE.\n"
        "BIORĘ NA SIEBIE KONSEKWENCJĘ ZA FAŁSZYWE OŚWIADCZENIE.\n"
        "======================================================\n"
    )


def format_shipment_summary(shipment: ShipmentInput) -> str:
    weight_tons = shipment.declared_mass_kg / 1000
    weight_tons_text = f"{weight_tons:.1f}".replace(".", ",")
    special_notes = shipment.special_notes.strip() or "brak - nie dodawaj zadnych uwag"
    budget_text = (
        f"{shipment.budget_pp} PP"
        if shipment.budget_pp != 0
        else "0 PP (przesylka ma byc darmowa lub finansowana przez System)"
    )

    return (
        "| Pole | Wartosc |\n"
        "| --- | --- |\n"
        f"| Nadawca (identyfikator) | {shipment.sender_id} |\n"
        f"| Punkt nadawczy | {shipment.origin} |\n"
        f"| Punkt docelowy | {shipment.destination} |\n"
        f"| Waga | {weight_tons_text} tony ({shipment.declared_mass_kg} kg) |\n"
        f"| Budzet | {budget_text} |\n"
        f"| Zawartosc | {shipment.contents} |\n"
        f"| Uwagi specjalne | {special_notes} |\n"
    )


def write_outputs(output_dir: Path, shipment: ShipmentInput, result: ValidationResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shipment_summary.md").write_text(
        format_shipment_summary(shipment),
        encoding="utf-8",
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(
            {
                "shipment": asdict(shipment),
                "result": asdict(result),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if result.declaration_text:
        declaration_filename = (
            "declaration.txt" if result.status == "READY" else "declaration.pending.txt"
        )
        (output_dir / declaration_filename).write_text(
            result.declaration_text,
            encoding="utf-8",
        )


def main() -> int:
    args = parse_args()
    base_dir = Path.cwd()
    shipment_file = resolve_path(args.shipment_file, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)

    if not shipment_file.exists():
        logger.error("Brak pliku z danymi przesylki: {}", shipment_file)
        return 1

    shipment = load_shipment(shipment_file)
    result = validate_shipment(shipment)
    write_outputs(output_dir, shipment, result)

    if result.is_valid:
        logger.success("Deklaracja zostala wygenerowana w {}.", output_dir)
        return 0

    if result.status == "PENDING_ROUTE_REOPEN":
        logger.warning(
            "Zapisano draft oczekujacy w {}. Nie jest gotowy do wysylki bez ponownej walidacji.",
            output_dir,
        )
        for error in result.errors:
            logger.warning("- {}", error)
        return 0

    logger.warning("Nie wygenerowano deklaracji. Powody:")
    for error in result.errors:
        logger.warning("- {}", error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
