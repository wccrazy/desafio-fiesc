from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import pytest
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

from api import SensorEvent
from rag_engine import DefensiveRAG
from similarity_engine import (
    InputQualityError,
    SimilarityEngine,
    TemporalLeakageError,
)


class FakeEmbeddings:
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        length = float(len(text) % 7 + 1)
        return [length, 1.0, 0.5]


def make_row(
    *,
    source_id: int,
    created_at: datetime,
    fault: str,
    episode: int,
    sample: int,
) -> dict[str, Any]:
    del episode
    if fault == "fault_a":
        rpm = 1_000.0
        velocity = 5.0 + sample * 0.01
        acceleration = 1.2 + sample * 0.005
        kurtosis = 5.0
        frequency = 33.0
        temperature = 38.0
    elif fault == "motor_desligado":
        rpm = 0.0
        velocity = 0.01
        acceleration = 0.01
        kurtosis = -0.4
        frequency = 0.0
        temperature = 23.0
    else:
        rpm = 1_000.0
        velocity = 1.0 + sample * 0.002
        acceleration = 0.12 + sample * 0.001
        kurtosis = -0.2
        frequency = 16.7
        temperature = 25.0

    return {
        "id": source_id,
        "created_at": created_at.isoformat(),
        "fault": fault,
        "machine_id": "machine_01",
        "temperature_c": temperature,
        "temperature_f": temperature * 9.0 / 5.0 + 32.0,
        "x_rms_velocity_mm_s": velocity,
        "x_rms_velocity_in_s": velocity / 25.4,
        "z_rms_velocity_mm_s": velocity * 0.9,
        "z_rms_velocity_in_s": velocity * 0.9 / 25.4,
        "x_peak_velocity_mm_s": velocity * 1.5,
        "x_peak_velocity_in_s": velocity * 1.5 / 25.4,
        "z_peak_velocity_mm_s": velocity * 1.4,
        "z_peak_velocity_in_s": velocity * 1.4 / 25.4,
        "x_peak_acceleration_g": acceleration * 2.0,
        "z_peak_acceleration_g": acceleration * 1.8,
        "x_rms_acceleration_g": acceleration,
        "z_rms_acceleration_g": acceleration * 0.9,
        "x_high_freq_rms_accel_g": acceleration * 0.7,
        "z_high_freq_rms_accel_g": acceleration * 0.65,
        "x_peak_vel_comp_freq_hz": frequency,
        "z_peak_vel_comp_freq_hz": frequency,
        "x_kurtosis": kurtosis,
        "z_kurtosis": kurtosis + 0.1,
        "x_crest_factor": 3.0 if fault != "fault_a" else 5.0,
        "z_crest_factor": 2.8 if fault != "fault_a" else 4.8,
        "rpm": rpm,
    }


@pytest.fixture()
def synthetic_csv(tmp_path: Path) -> Path:
    rows: list[dict[str, Any]] = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    source_id = 1
    labels = ["normal", "fault_a", "motor_desligado"]

    for episode in range(18):
        label = labels[episode % len(labels)]
        episode_start = start + timedelta(minutes=episode * 10)
        for sample in range(5):
            rows.append(
                make_row(
                    source_id=source_id,
                    created_at=episode_start
                    + timedelta(seconds=sample * 2),
                    fault=label,
                    episode=episode,
                    sample=sample,
                )
            )
            source_id += 1

    path = tmp_path / "banner.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture()
def engine(
    tmp_path: Path,
    synthetic_csv: Path,
) -> SimilarityEngine:
    instance = SimilarityEngine(
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        artifact_path=tmp_path / "model.joblib",
        k_max=5,
        episode_gap_seconds=300,
        minimum_rpm_bin_samples=3,
    )
    instance.fit_from_csv(synthetic_csv)
    return instance


def test_target_leakage_absent(engine: SimilarityEngine) -> None:
    event = engine.get_test_sample(
        label="fault_a",
        include_fault=True,
    )
    first = engine.diagnose(event, strict_temporal=True)

    altered = dict(event)
    altered["fault"] = "normal"
    second = engine.diagnose(altered, strict_temporal=True)

    assert first["status"] == second["status"]
    assert first["inferred_label"] == second["inferred_label"]
    assert (
        first["neighborhood"]["neighbors"]
        == second["neighborhood"]["neighbors"]
    )
    assert first["target_leakage_control"]["fault_used_as_feature"] is False


def test_temporal_split_has_no_future_leakage(
    engine: SimilarityEngine,
) -> None:
    summary = engine.split_summary

    assert set(summary["reference_ids"]).isdisjoint(
        summary["test_ids"]
    )
    assert pd.Timestamp(
        summary["max_reference_timestamp"]
    ) < pd.Timestamp(summary["min_test_timestamp"])

    old_event = engine.reference.iloc[0].to_dict()
    with pytest.raises(TemporalLeakageError):
        engine.diagnose(old_event, strict_temporal=True)


def test_unit_consistency_and_negative_kurtosis(
    engine: SimilarityEngine,
) -> None:
    event = engine.get_test_sample(label="normal")
    event["x_kurtosis"] = -0.8
    SensorEvent.model_validate(event)

    inconsistent = dict(event)
    inconsistent["x_rms_velocity_in_s"] = 99.0

    with pytest.raises(InputQualityError):
        engine.canonicalize(inconsistent)


def test_motor_off_is_non_problem_state(
    engine: SimilarityEngine,
) -> None:
    event = engine.get_test_sample(label="motor_desligado")
    result = engine.diagnose(event, strict_temporal=True)

    assert result["status"] == "non_problem_state"
    assert result["inferred_label"] == "motor_desligado"
    assert result["is_problem"] is False


def test_llm_blocked_without_active_manual(
    tmp_path: Path,
) -> None:
    calls = {"count": 0}

    def fake_llm_factory(provider: str) -> Any:
        del provider
        calls["count"] += 1
        raise AssertionError("A LLM não deveria ser criada.")

    rag = DefensiveRAG(
        database_url=f"sqlite:///{tmp_path / 'rag.db'}",
        collection_name="test_collection",
        chroma_client=chromadb.EphemeralClient(),
        embeddings=FakeEmbeddings(),
        llm_factory=fake_llm_factory,
    )

    result = rag.answer(
        fault_code="fault_without_manual",
        question="Como reparar?",
        provider="ollama",
    )

    assert result["status"] == "document_absent"
    assert result["llm_called"] is False
    assert calls["count"] == 0
