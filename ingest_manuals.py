"""Script de ingestão e ativação de manuais técnicos no DefensiveRAG.

Processa os manuais em data/manuals/ (Doc1.pdf a Doc6.pdf), indexando os chunks no
ChromaDB e registrando/ativando as versões no SQLite (artifacts/app.db).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

from rag_engine import DefensiveRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_manuals")

DATABASE_URL = "sqlite:///./artifacts/app.db"
MANUALS_DIR = Path("./data/manuals")

DOC_MAPPING: dict[str, dict[str, str]] = {
    "Doc1.pdf": {
        "title": "Procedimento para Diagnóstico e Correção de Problemas em Rolamentos",
        "fault_code": "rolamento_outer",
    },
    "Doc2.pdf": {
        "title": "Procedimento para Correção de Desalinhamento em Motor Elétrico",
        "fault_code": "desalinhado",
    },
    "Doc3.pdf": {
        "title": "Procedimento para Correção de Desbalanceamento em Máquinas Rotativas",
        "fault_code": "desbalanceado_1parafuso",
    },
    "Doc4.pdf": {
        "title": "Procedimento para Diagnóstico e Correção de Problemas em Sistemas de Transmissão por Correias",
        "fault_code": "correia",
    },
    "Doc5.pdf": {
        "title": "Procedimento para Diagnóstico e Correção de Problemas em Polias de Sistemas Rotativos",
        "fault_code": "polia",
    },
    "Doc6.pdf": {
        "title": "Procedimento para Diagnóstico e Correção de Problemas de Rotor Inclinado (Cocked Rotor)",
        "fault_code": "cocked_rotor",
    },
}


def run_ingestion() -> None:
    Path("./artifacts").mkdir(parents=True, exist_ok=True)

    logger.info("Inicializando DefensiveRAG com banco: %s", DATABASE_URL)
    rag = DefensiveRAG(database_url=DATABASE_URL)

    success_count = 0
    total_docs = len(DOC_MAPPING)

    logger.info("Iniciando ingestão de %d manuais...", total_docs)
    print("=" * 80)

    for filename, meta in DOC_MAPPING.items():
        pdf_path = MANUALS_DIR / filename
        title = meta["title"]
        fault_code = meta["fault_code"]

        if not pdf_path.exists():
            logger.error("Arquivo não encontrado: %s", pdf_path)
            continue

        pdf_bytes = pdf_path.read_bytes()
        logger.info(
            "Processando [%s] | Fault: '%s' | Título: '%s' (%d bytes)...",
            filename,
            fault_code,
            title,
            len(pdf_bytes),
        )

        try:
            # 1. Ingestão do PDF
            ingest_result = rag.ingest_pdf(
                pdf_bytes,
                filename=filename,
                title=title,
                fault_code=fault_code,
                version="1.0",
            )
            doc_id = ingest_result["document_id"]
            chunk_count = ingest_result.get("chunk_count", 0)
            logger.info(
                "-> Ingestão concluída: doc_id=%s (status=%s, chunks=%d)",
                doc_id,
                ingest_result.get("status"),
                chunk_count,
            )

            # 2. Ativação imediata da versão como autoritativa
            activate_result = rag.activate_document(doc_id)
            logger.info(
                "-> Ativação concluída: status=%s (fault_code=%s)",
                activate_result.get("status"),
                activate_result.get("fault_code"),
            )
            success_count += 1
            print(
                f"[OK] {filename:10s} | doc_id: {doc_id} | status: {activate_result.get('status')} | chunks: {chunk_count}"
            )

        except Exception as exc:
            logger.exception("Erro ao processar %s: %s", filename, exc)
            print(f"[FALHA] {filename:10s} | Erro: {exc}")

        print("-" * 80)

    print("=" * 80)
    logger.info(
        "Ingestão finalizada: %d/%d manuais indexados e ativados com sucesso.",
        success_count,
        total_docs,
    )

    if success_count != total_docs:
        sys.exit(1)


if __name__ == "__main__":
    run_ingestion()
