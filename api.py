from __future__ import annotations

import logging
import math
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

import httpx
from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.concurrency import run_in_threadpool

from rag_engine import DefensiveRAG, RAGError
from similarity_engine import (
    EngineUnavailableError,
    InputQualityError,
    SimilarityEngine,
    TemporalLeakageError,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maintenance-api")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    database_url: str = "sqlite:///./artifacts/app.db"
    csv_path: str = "./data/banner.csv"
    artifact_path: str = "./artifacts/similarity.joblib"
    kurtosis_definition: Literal["auto", "fisher", "pearson"] = "auto"

    allow_cloud: bool = False
    enable_demo_endpoints: bool = True
    demo_admin_token: str = "change-me"
    max_upload_mb: int = 20
    cors_origins: str = "http://localhost:8501"

    chroma_collection: str = "maintenance_manuals_v2"
    embedding_model: str = (
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    rag_min_similarity: float = 0.42
    ollama_base_url: str = "http://ollama:11434"


settings = Settings()


class SensorEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | None = Field(default=None, ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    machine_id: str = Field(default="machine_01", max_length=128)
    sequence_no: int | None = Field(default=None, ge=0)
    schema_version: str | None = Field(default=None, max_length=32)

    z_rms_velocity_in_s: float | None = Field(default=None, ge=0)
    z_rms_velocity_mm_s: float | None = Field(default=None, ge=0)
    temperature_f: float | None = None
    temperature_c: float | None = None
    x_rms_velocity_in_s: float | None = Field(default=None, ge=0)
    x_rms_velocity_mm_s: float | None = Field(default=None, ge=0)

    z_peak_acceleration_g: float = Field(ge=0)
    x_peak_acceleration_g: float = Field(ge=0)
    z_peak_vel_comp_freq_hz: float = Field(ge=0)
    x_peak_vel_comp_freq_hz: float = Field(ge=0)
    z_rms_acceleration_g: float = Field(ge=0)
    x_rms_acceleration_g: float = Field(ge=0)

    # Fisher pode ser negativo.
    z_kurtosis: float
    x_kurtosis: float

    z_crest_factor: float = Field(ge=0)
    x_crest_factor: float = Field(ge=0)
    z_peak_velocity_in_s: float | None = Field(default=None, ge=0)
    z_peak_velocity_mm_s: float | None = Field(default=None, ge=0)
    x_peak_velocity_in_s: float | None = Field(default=None, ge=0)
    x_peak_velocity_mm_s: float | None = Field(default=None, ge=0)
    z_high_freq_rms_accel_g: float = Field(ge=0)
    x_high_freq_rms_accel_g: float = Field(ge=0)
    rpm: float = Field(ge=0)

    # Somente anotação de replay; nunca entra nas features.
    fault: str | None = Field(default=None, max_length=128)

    @field_validator("created_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at deve possuir timezone.")
        return value

    @model_validator(mode="after")
    def validate_pairs(self) -> "SensorEvent":
        pairs = (
            ("temperature_c", "temperature_f"),
            ("x_rms_velocity_mm_s", "x_rms_velocity_in_s"),
            ("z_rms_velocity_mm_s", "z_rms_velocity_in_s"),
            ("x_peak_velocity_mm_s", "x_peak_velocity_in_s"),
            ("z_peak_velocity_mm_s", "z_peak_velocity_in_s"),
        )
        for first, second in pairs:
            if getattr(self, first) is None and getattr(self, second) is None:
                raise ValueError(
                    f"Informe pelo menos um entre {first} e {second}."
                )

        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field_name} deve ser finito.")
        return self


class DiagnoseRequest(SensorEvent):
    question: str = Field(
        default="Quais ações são sustentadas pelo manual ativo?",
        min_length=3,
        max_length=1_500,
    )
    include_prescription: bool = True
    llm_provider: Literal["none", "ollama", "cloud"] = "none"
    strict_temporal: bool = True
    asset_criticality: float = Field(default=0.5, ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.similarity = None
    app.state.rag = None
    app.state.similarity_error = None
    app.state.rag_error = None
    app.state.latest = {}

    try:
        engine = SimilarityEngine(
            database_url=settings.database_url,
            artifact_path=settings.artifact_path,
            kurtosis_definition=settings.kurtosis_definition,
        )
        summary = await run_in_threadpool(
            engine.load_or_fit,
            settings.csv_path,
        )
        app.state.similarity = engine
        logger.info("Similarity ready: %s", summary)
    except Exception as exc:
        app.state.similarity_error = str(exc)
        logger.exception("Similarity iniciou em modo degradado.")

    try:
        app.state.rag = await run_in_threadpool(
            lambda: DefensiveRAG(
                database_url=settings.database_url,
                collection_name=settings.chroma_collection,
                embedding_model=settings.embedding_model,
                min_similarity=settings.rag_min_similarity,
            )
        )
    except Exception as exc:
        app.state.rag_error = str(exc)
        logger.exception("RAG iniciou em modo degradado.")

    yield


app = FastAPI(
    title="Prescriptive Maintenance API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        item.strip()
        for item in settings.cors_origins.split(",")
        if item.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Token", "X-Trace-Id"],
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next: Any) -> Any:
    trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


def similarity_or_503(request: Request) -> SimilarityEngine:
    engine = request.app.state.similarity
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail={
                "component": "similarity",
                "error": request.app.state.similarity_error,
            },
        )
    return engine


def rag_or_503(request: Request) -> DefensiveRAG:
    rag = request.app.state.rag
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail={
                "component": "rag",
                "error": request.app.state.rag_error,
            },
        )
    return rag


async def ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.get(
                f"{settings.ollama_base_url}/api/tags"
            )
            return response.is_success
    except Exception:
        return False


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    rag_health = None
    if request.app.state.rag is not None:
        rag_health = await run_in_threadpool(
            request.app.state.rag.health
        )

    return {
        "status": (
            "ok"
            if request.app.state.similarity is not None
            else "degraded"
        ),
        "similarity": {
            "ready": request.app.state.similarity is not None,
            "error": request.app.state.similarity_error,
        },
        "rag": {
            "ready": request.app.state.rag is not None,
            "error": request.app.state.rag_error,
            "details": rag_health,
        },
        "providers": {
            "ollama": await ollama_available(),
            "cloud": settings.allow_cloud,
        },
    }


@app.post("/diagnose")
async def diagnose(
    payload: DiagnoseRequest,
    request: Request,
) -> dict[str, Any]:
    if payload.llm_provider == "cloud" and not settings.allow_cloud:
        raise HTTPException(
            status_code=403,
            detail="Nuvem desabilitada por política.",
        )

    engine = similarity_or_503(request)
    raw = payload.model_dump(exclude_none=True)

    try:
        diagnosis = await run_in_threadpool(
            lambda: engine.diagnose(
                raw,
                strict_temporal=payload.strict_temporal,
                asset_criticality=payload.asset_criticality,
                trace_id=request.state.trace_id,
            )
        )
    except TemporalLeakageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InputQualityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EngineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not payload.include_prescription or payload.llm_provider == "none":
        prescription = {
            "status": "not_requested",
            "llm_called": False,
            "answer": "Prescrição generativa não solicitada.",
            "citations": [],
        }
    elif diagnosis["status"] == "known_problem":
        rag = rag_or_503(request)
        prescription = await run_in_threadpool(
            lambda: rag.answer(
                fault_code=diagnosis["inferred_label"],
                question=payload.question,
                fault_summary=diagnosis,
                provider=payload.llm_provider,
            )
        )
    elif diagnosis["status"] == "non_problem_state":
        prescription = {
            "status": "not_applicable",
            "llm_called": False,
            "answer": (
                "Estado operacional não problemático. O RAG não foi "
                "acionado."
            ),
            "citations": [],
        }
    else:
        prescription = {
            "status": "diagnostic_abstention",
            "llm_called": False,
            "answer": (
                "O motor numérico se absteve; nenhuma prescrição foi "
                "gerada."
            ),
            "citations": [],
        }

    result = {
        "trace_id": request.state.trace_id,
        "diagnosis": diagnosis,
        "prescription": prescription,
    }
    request.app.state.latest[payload.machine_id] = result
    return result


@app.post("/documents/upload")
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    fault_code: Annotated[str, Form(...)],
    title: Annotated[str, Form(...)],
    version: Annotated[str, Form()] = "1.0",
) -> dict[str, Any]:
    rag = rag_or_503(request)
    maximum = settings.max_upload_mb * 1024 * 1024
    content = await file.read(maximum + 1)

    if len(content) > maximum:
        raise HTTPException(status_code=413, detail="PDF muito grande.")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail="PDF inválido.")

    try:
        return await run_in_threadpool(
            lambda: rag.ingest_pdf(
                content,
                filename=file.filename or "manual.pdf",
                title=title,
                fault_code=fault_code,
                version=version,
            )
        )
    except (RAGError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/documents/{document_id}/activate")
async def activate_document(
    document_id: str,
    request: Request,
    x_admin_token: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    if (
        x_admin_token is None
        or not secrets.compare_digest(
            x_admin_token,
            settings.demo_admin_token,
        )
    ):
        raise HTTPException(status_code=403, detail="Token inválido.")

    rag = rag_or_503(request)
    try:
        return await run_in_threadpool(
            rag.activate_document,
            document_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/documents")
async def documents(request: Request) -> list[dict[str, Any]]:
    rag = rag_or_503(request)
    return await run_in_threadpool(rag.list_documents)


@app.get("/telemetry/stats")
async def telemetry_stats(request: Request) -> dict[str, Any]:
    engine = similarity_or_503(request)
    return await run_in_threadpool(engine.telemetry_stats)


@app.get("/telemetry/sample")
async def telemetry_sample(
    request: Request,
    offset: int = Query(default=0, ge=0),
    label: str | None = Query(default=None),
    include_fault: bool = Query(default=False),
) -> dict[str, Any]:
    engine = similarity_or_503(request)
    try:
        return await run_in_threadpool(
            lambda: engine.get_test_sample(
                offset,
                label=label,
                include_fault=include_fault,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/telemetry/latest")
async def telemetry_latest(
    request: Request,
    machine_id: str = "machine_01",
) -> dict[str, Any]:
    result = request.app.state.latest.get(machine_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Sem leitura online.")
    return result


@app.post("/demo/anti-hallucination")
async def anti_hallucination_demo(
    request: Request,
) -> dict[str, Any]:
    if not settings.enable_demo_endpoints:
        raise HTTPException(status_code=404, detail="Desabilitado.")

    rag = rag_or_503(request)
    return await run_in_threadpool(
        lambda: rag.answer(
            fault_code="falha_sem_manual_live_test",
            question="Informe peças, torques e etapas completas.",
            provider="ollama",
        )
    )
