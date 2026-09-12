"""Descarga y prepara los accidentes de trafico de Madrid de 2019 a 2025."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tfm_accidentes.config import (  # noqa: E402
    CKAN_PACKAGE_URL,
    END_YEAR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    START_YEAR,
)
from tfm_accidentes.data import (  # noqa: E402
    build_analysis_table,
    build_quality_report,
    load_raw_files,
)


def fetch_json(url: str, timeout: int = 60) -> dict:
    """Obtiene un JSON mediante HTTPS usando solo la biblioteca estandar."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "TFM-Accidentes-Madrid/0.1 (uso academico)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def discover_csv_resources(metadata: dict, years: range) -> dict[int, dict]:
    """Relaciona cada anio con su recurso CSV usando la descripcion CKAN."""

    resources_by_year: dict[int, dict] = {}
    for resource in metadata["result"]["resources"]:
        if str(resource.get("format", "")).upper() != "CSV":
            continue

        searchable_text = " ".join(
            str(resource.get(field, ""))
            for field in ("description", "name", "url")
        )
        found_years = {int(value) for value in re.findall(r"20\d{2}", searchable_text)}
        for year in found_years.intersection(years):
            if year in resources_by_year:
                raise ValueError(f"Se han encontrado varios CSV para el anio {year}")
            resources_by_year[year] = resource

    missing_years = sorted(set(years).difference(resources_by_year))
    if missing_years:
        raise ValueError(f"No se han encontrado CSV para los anios: {missing_years}")

    return resources_by_year


def download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    """Descarga un fichero de forma atomica."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"[omitido] {destination.name} ya existe")
        return

    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "TFM-Accidentes-Madrid/0.1 (uso academico)"},
    )
    print(f"[descarga] {destination.name}")
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    temporary_path.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Descarga los CSV, pero no genera la tabla unificada.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year no puede ser posterior a end-year")

    years = range(args.start_year, args.end_year + 1)
    metadata = fetch_json(CKAN_PACKAGE_URL)
    if not metadata.get("success"):
        raise RuntimeError("La API CKAN no ha devuelto una respuesta valida")

    resources = discover_csv_resources(metadata, years)
    manifest: list[dict[str, object]] = []
    raw_paths: list[Path] = []

    for year in years:
        resource = resources[year]
        destination = RAW_DATA_DIR / f"accidentes_madrid_{year}.csv"
        download_file(resource["url"], destination, overwrite=args.overwrite)
        raw_paths.append(destination)
        manifest.append(
            {
                "year": year,
                "resource_id": resource.get("id"),
                "description": resource.get("description"),
                "url": resource.get("url"),
                "last_modified": resource.get("last_modified"),
                "local_file": str(destination.relative_to(PROJECT_ROOT)),
            }
        )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = PROCESSED_DATA_DIR / "metadata_descarga.json"
    metadata_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.skip_processed:
        return

    print("[proceso] Cargando y homogeneizando los ficheros anuales")
    raw_data = load_raw_files(raw_paths)
    analysis_data = build_analysis_table(raw_data)

    output_path = (
        PROCESSED_DATA_DIR
        / f"accidentes_madrid_{args.start_year}_{args.end_year}.csv.gz"
    )
    analysis_data.to_csv(output_path, index=False, compression="gzip")
    build_quality_report(analysis_data).to_csv(
        PROCESSED_DATA_DIR / "resumen_calidad.csv",
        index=False,
    )

    known_target = analysis_data["lesion_grave"].dropna()
    initial_summary = {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "rows": len(analysis_data),
        "unique_accidents": int(analysis_data["num_expediente"].nunique()),
        "fully_duplicated_rows": int(analysis_data.duplicated().sum()),
        "unknown_target_rows": int(analysis_data["lesion_grave"].isna().sum()),
        "serious_or_fatal_rows": int(known_target.sum()),
        "serious_or_fatal_prevalence": float(known_target.mean()),
        "rows_by_year": {
            str(int(year)): int(count)
            for year, count in analysis_data["anio"].value_counts().sort_index().items()
        },
    }
    (PROCESSED_DATA_DIR / "resumen_inicial.json").write_text(
        json.dumps(initial_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[ok] Filas totales: {len(analysis_data):,}")
    print(f"[ok] Expedientes: {analysis_data['num_expediente'].nunique():,}")
    print(f"[ok] Objetivo desconocido: {analysis_data['lesion_grave'].isna().sum():,}")
    print(f"[ok] Prevalencia de lesion grave: {known_target.mean():.2%}")
    print(f"[ok] Tabla preparada: {output_path}")


if __name__ == "__main__":
    main()
