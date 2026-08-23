"""Script de ingestão e ativação de manuais técnicos via API REST.

Envia os 6 manuais em data/manuals/ (Doc1.pdf a Doc6.pdf) para a API FastAPI
(http://localhost:8000), realizando o upload em multipart/form-data e em seguida
a ativação do documento para torná-lo autoritativo.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_via_api")

API_BASE_URL = "http://localhost:8000"
ADMIN_TOKEN = "change-me"
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
    logger.info("Conectando à API em: %s", API_BASE_URL)

    with httpx.Client(base_url=API_BASE_URL, timeout=120.0) as client:
        try:
            health_resp = client.get("/health")
            health_resp.raise_for_status()
            logger.info("API Online! Health check: %s", health_resp.json())
        except Exception as exc:
            logger.error("Falha ao conectar na API (%s): %s", API_BASE_URL, exc)
            sys.exit(1)

        success_count = 0
        total_docs = len(DOC_MAPPING)

        logger.info("Iniciando ingestão via API de %d manuais...", total_docs)
        print("=" * 80)

        for filename, meta in DOC_MAPPING.items():
            pdf_path = MANUALS_DIR / filename
            title = meta["title"]
            fault_code = meta["fault_code"]

            if not pdf_path.exists():
                logger.error("Arquivo local não encontrado: %s", pdf_path)
                print(f"[ERRO] Arquivo não encontrado: {pdf_path}")
                continue

            pdf_bytes = pdf_path.read_bytes()
            logger.info(
                "Enviando [%s] | Fault: '%s' | Título: '%s' (%d bytes)...",
                filename,
                fault_code,
                title,
                len(pdf_bytes),
            )

            # 1. Upload do documento (POST /documents/upload)
            files = {
                "file": (filename, pdf_bytes, "application/pdf"),
            }
            data = {
                "fault_code": fault_code,
                "title": title,
                "version": "1.0",
            }

            try:
                upload_resp = client.post("/documents/upload", files=files, data=data)
                upload_resp.raise_for_status()
                upload_json = upload_resp.json()
                doc_id = upload_json["document_id"]
                chunk_count = upload_json.get("chunk_count", 0)
                logger.info(
                    "-> Upload OK (HTTP %d): doc_id=%s, chunks=%d",
                    upload_resp.status_code,
                    doc_id,
                    chunk_count,
                )

                # 2. Ativação do documento (POST /documents/{document_id}/activate)
                headers = {
                    "X-Admin-Token": ADMIN_TOKEN,
                }
                activate_resp = client.post(
                    f"/documents/{doc_id}/activate",
                    headers=headers,
                )
                activate_resp.raise_for_status()
                activate_json = activate_resp.json()
                logger.info(
                    "-> Ativação OK (HTTP %d): status=%s, fault_code=%s",
                    activate_resp.status_code,
                    activate_json.get("status"),
                    activate_json.get("fault_code"),
                )

                success_count += 1
                print(
                    f"[OK 200] {filename:10s} | doc_id: {doc_id} | "
                    f"status: {activate_json.get('status')} | chunks: {chunk_count}"
                )

            except httpx.HTTPStatusError as exc:
                logger.error(
                    "HTTP Error ao processar %s: %s - %s",
                    filename,
                    exc.response.status_code,
                    exc.response.text,
                )
                print(
                    f"[FALHA HTTP {exc.response.status_code}] {filename:10s} | "
                    f"Erro: {exc.response.text}"
                )
            except Exception as exc:
                logger.exception("Erro inesperado ao processar %s: %s", filename, exc)
                print(f"[FALHA] {filename:10s} | Erro: {exc}")

            print("-" * 80)

        # 3. Listar documentos ativos na API
        try:
            docs_resp = client.get("/documents")
            if docs_resp.status_code == 200:
                active_docs = docs_resp.json()
                logger.info(
                    "Documentos registrados na API: %d", len(active_docs)
                )
        except Exception:
            pass

        print("=" * 80)
        logger.info(
            "Finalizado: %d/%d manuais indexados e ativados com sucesso via API.",
            success_count,
            total_docs,
        )

        if success_count != total_docs:
            sys.exit(1)


if __name__ == "__main__":
    run_ingestion()
