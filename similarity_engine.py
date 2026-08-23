from __future__ import annotations

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


ARTIFACT_VERSION = 3

NON_PROBLEM_STATES = frozenset(
    {"normal", "baseline", "teste", "acelerando", "motor_desligado"}
)
ACTIVE_BASELINE_STATES = frozenset({"normal", "baseline"})

CANONICAL_FEATURES = (
    "temperature_c",
    "x_rms_velocity_mm_s",
    "z_rms_velocity_mm_s",
    "x_peak_velocity_mm_s",
    "z_peak_velocity_mm_s",
    "x_peak_acceleration_g",
    "z_peak_acceleration_g",
    "x_rms_acceleration_g",
    "z_rms_acceleration_g",
    "x_high_freq_rms_accel_g",
    "z_high_freq_rms_accel_g",
    "x_peak_vel_comp_freq_hz",
    "z_peak_vel_comp_freq_hz",
    "x_kurtosis",
    "z_kurtosis",
    "x_crest_factor",
    "z_crest_factor",
    "rpm",
)

MODEL_FEATURES = (
    "temperature_c",
    "x_rms_velocity_mm_s",
    "z_rms_velocity_mm_s",
    "x_peak_velocity_mm_s",
    "z_peak_velocity_mm_s",
    "x_peak_acceleration_g",
    "z_peak_acceleration_g",
    "x_rms_acceleration_g",
    "z_rms_acceleration_g",
    "x_high_freq_rms_accel_g",
    "z_high_freq_rms_accel_g",
    "x_kurtosis",
    "z_kurtosis",
    "x_crest_factor",
    "z_crest_factor",
    "rpm",
    "x_regularized_order_proxy",
    "z_regularized_order_proxy",
    "log_xz_rms_velocity_ratio",
    "log_xz_rms_acceleration_ratio",
    "log_xz_high_freq_ratio",
)

FEATURE_GROUPS: dict[str, tuple[float, tuple[str, ...]]] = {
    "velocity": (
        0.30,
        (
            "x_rms_velocity_mm_s",
            "z_rms_velocity_mm_s",
            "x_peak_velocity_mm_s",
            "z_peak_velocity_mm_s",
        ),
    ),
    "acceleration": (
        0.30,
        (
            "x_peak_acceleration_g",
            "z_peak_acceleration_g",
            "x_rms_acceleration_g",
            "z_rms_acceleration_g",
            "x_high_freq_rms_accel_g",
            "z_high_freq_rms_accel_g",
        ),
    ),
    "shape": (
        0.15,
        (
            "x_kurtosis",
            "z_kurtosis",
            "x_crest_factor",
            "z_crest_factor",
        ),
    ),
    "order": (
        0.12,
        (
            "x_regularized_order_proxy",
            "z_regularized_order_proxy",
        ),
    ),
    "thermal": (0.03, ("temperature_c",)),
    "speed": (0.03, ("rpm",)),
    "cross_axis": (
        0.07,
        (
            "log_xz_rms_velocity_ratio",
            "log_xz_rms_acceleration_ratio",
            "log_xz_high_freq_ratio",
        ),
    ),
}

SEVERITY_FEATURES = (
    "temperature_c",
    "x_rms_velocity_mm_s",
    "z_rms_velocity_mm_s",
    "x_peak_velocity_mm_s",
    "z_peak_velocity_mm_s",
    "x_peak_acceleration_g",
    "z_peak_acceleration_g",
    "x_rms_acceleration_g",
    "z_rms_acceleration_g",
    "x_high_freq_rms_accel_g",
    "z_high_freq_rms_accel_g",
    "x_kurtosis",
    "z_kurtosis",
    "x_crest_factor",
    "z_crest_factor",
)


class InputQualityError(ValueError):
    pass


class TemporalLeakageError(ValueError):
    pass


class EngineUnavailableError(RuntimeError):
    pass


class Base(DeclarativeBase):
    pass


class HistoricalEvent(Base):
    __tablename__ = "historical_events"
    __table_args__ = (
        UniqueConstraint(
            "dataset_hash",
            "machine_id",
            "source_id",
            name="uq_historical_snapshot_event",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), index=True)
    machine_id: Mapped[str] = mapped_column(String(128), index=True)
    source_id: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    fault: Mapped[str] = mapped_column(String(128), index=True)
    episode_id: Mapped[str] = mapped_column(String(255), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class DiagnosisLog(Base):
    __tablename__ = "diagnosis_log"
    __table_args__ = (
        UniqueConstraint(
            "machine_id",
            "sequence_no",
            name="uq_machine_sequence",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    machine_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sequence_no: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(64), index=True)
    inferred_label: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    result: Mapped[dict[str, Any]] = mapped_column(JSON)


def normalize_label(value: Any) -> str:
    label = str(value).strip().lower()
    if not label:
        raise InputQualityError("Rótulo vazio.")
    return label


def json_safe(value: Any) -> Any:
    def sanitize(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        if isinstance(obj, np.generic):
            val = obj.item()
            if isinstance(val, float) and (math.isinf(val) or math.isnan(val)):
                return None
            return val
        return obj

    sanitized = sanitize(value)
    return json.loads(
        json.dumps(
            sanitized,
            ensure_ascii=False,
            default=str,
        )
    )


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SafeRobustScaler:
    """RobustScaler com piso de IQR/MAD para features quase constantes."""

    def __init__(self, relative_floor: float = 1e-3) -> None:
        self.relative_floor = relative_floor
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "SafeRobustScaler":
        raw = RobustScaler(quantile_range=(25.0, 75.0))
        raw.fit(values)

        center = np.asarray(raw.center_, dtype=np.float64)
        iqr = np.asarray(raw.scale_, dtype=np.float64)
        mad = (
            np.median(np.abs(values - center), axis=0).astype(np.float64)
            * 1.4826
        )
        floor = self.relative_floor * np.maximum(np.abs(center), 1.0)

        self.center_ = center
        self.scale_ = np.maximum.reduce([iqr, mad, floor])
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise EngineUnavailableError("Scaler não ajustado.")
        return (values - self.center_) / self.scale_


class SimilarityEngine:
    def __init__(
        self,
        *,
        database_url: str,
        artifact_path: str | Path,
        k_max: int = 9,
        consensus_threshold: float = 0.65,
        episode_gap_seconds: int = 300,
        calibration_quantile: float = 0.95,
        minimum_rpm_bin_samples: int = 8,
        critical_faults: Sequence[str] = (),
        kurtosis_definition: str = "auto",
    ) -> None:
        if k_max < 3:
            raise ValueError("k_max deve ser pelo menos 3.")
        if kurtosis_definition not in {"auto", "fisher", "pearson"}:
            raise ValueError("Definição de curtose inválida.")

        connect_args = (
            {"check_same_thread": False}
            if database_url.startswith("sqlite")
            else {}
        )
        self.sql_engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self.Session = sessionmaker(
            bind=self.sql_engine,
            expire_on_commit=False,
        )
        Base.metadata.create_all(self.sql_engine)

        self.artifact_path = Path(artifact_path)
        self.k_max = k_max
        self.consensus_threshold = consensus_threshold
        self.episode_gap_seconds = episode_gap_seconds
        self.calibration_quantile = calibration_quantile
        self.minimum_rpm_bin_samples = minimum_rpm_bin_samples
        self.critical_faults = {
            normalize_label(item) for item in critical_faults if item
        }
        self.kurtosis_definition = kurtosis_definition

        self.scaler: SafeRobustScaler | None = None
        self.knn: NearestNeighbors | None = None
        self.reference: pd.DataFrame | None = None
        self.test_records: pd.DataFrame | None = None
        self.feature_weights = self._feature_weights()

        self.dataset_hash: str | None = None
        self.novelty_threshold: float | None = None
        self.kernel_scale: float | None = None
        self.knowledge_cutoff: pd.Timestamp | None = None
        self.rpm_baselines: dict[str, Any] | None = None
        self.history_by_label: dict[str, Any] = {}
        self.global_history: dict[str, Any] = {}
        self.split_summary: dict[str, Any] = {}
        self.quarantine: list[dict[str, Any]] = []
        self._lock = RLock()

    @staticmethod
    def _feature_weights() -> np.ndarray:
        weights: dict[str, float] = {}
        for _, (group_weight, features) in FEATURE_GROUPS.items():
            per_feature = group_weight / len(features)
            for feature in features:
                if feature in weights:
                    raise RuntimeError(f"Feature duplicada: {feature}")
                weights[feature] = per_feature

        missing = set(MODEL_FEATURES) - set(weights)
        if missing:
            raise RuntimeError(f"Features sem peso: {missing}")

        return np.sqrt(
            np.array(
                [weights[name] for name in MODEL_FEATURES],
                dtype=np.float64,
            )
        )

    @staticmethod
    def _number(
        data: Mapping[str, Any],
        name: str,
        *,
        required: bool = True,
    ) -> float | None:
        value = data.get(name)
        if value is None:
            if required:
                raise InputQualityError(f"Campo ausente: {name}")
            return None
        if isinstance(value, bool):
            raise InputQualityError(f"{name} não pode ser booleano.")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise InputQualityError(f"{name} não é numérico.") from exc
        if not math.isfinite(parsed):
            raise InputQualityError(f"{name} deve ser finito.")
        return parsed

    @classmethod
    def _velocity_pair(
        cls,
        data: Mapping[str, Any],
        metric_name: str,
        imperial_name: str,
    ) -> float:
        metric = cls._number(data, metric_name, required=False)
        imperial = cls._number(data, imperial_name, required=False)

        if metric is None and imperial is None:
            raise InputQualityError(
                f"Informe {metric_name} ou {imperial_name}."
            )

        converted = None if imperial is None else imperial * 25.4
        if metric is not None and converted is not None:
            tolerance = max(0.05, abs(metric) * 0.03)
            if abs(metric - converted) > tolerance:
                raise InputQualityError(
                    f"Inconsistência entre {metric_name} e {imperial_name}."
                )
        return float(metric if metric is not None else converted)

    def canonicalize(
        self,
        data: Mapping[str, Any],
    ) -> tuple[dict[str, float], list[str]]:
        alerts: list[str] = []

        temperature_c = self._number(
            data,
            "temperature_c",
            required=False,
        )
        temperature_f = self._number(
            data,
            "temperature_f",
            required=False,
        )
        if temperature_c is None and temperature_f is None:
            raise InputQualityError(
                "Informe temperature_c ou temperature_f."
            )

        converted_c = (
            None
            if temperature_f is None
            else (temperature_f - 32.0) * 5.0 / 9.0
        )
        if (
            temperature_c is not None
            and converted_c is not None
            and abs(temperature_c - converted_c) > 0.6
        ):
            raise InputQualityError("Temperaturas °C/°F inconsistentes.")

        canonical: dict[str, float] = {
            "temperature_c": float(
                temperature_c
                if temperature_c is not None
                else converted_c
            ),
            "x_rms_velocity_mm_s": self._velocity_pair(
                data,
                "x_rms_velocity_mm_s",
                "x_rms_velocity_in_s",
            ),
            "z_rms_velocity_mm_s": self._velocity_pair(
                data,
                "z_rms_velocity_mm_s",
                "z_rms_velocity_in_s",
            ),
            "x_peak_velocity_mm_s": self._velocity_pair(
                data,
                "x_peak_velocity_mm_s",
                "x_peak_velocity_in_s",
            ),
            "z_peak_velocity_mm_s": self._velocity_pair(
                data,
                "z_peak_velocity_mm_s",
                "z_peak_velocity_in_s",
            ),
        }

        direct = (
            "x_peak_acceleration_g",
            "z_peak_acceleration_g",
            "x_rms_acceleration_g",
            "z_rms_acceleration_g",
            "x_high_freq_rms_accel_g",
            "z_high_freq_rms_accel_g",
            "x_peak_vel_comp_freq_hz",
            "z_peak_vel_comp_freq_hz",
            "x_kurtosis",
            "z_kurtosis",
            "x_crest_factor",
            "z_crest_factor",
            "rpm",
        )
        for name in direct:
            value = self._number(data, name)
            assert value is not None
            canonical[name] = value

        nonnegative = (
            set(CANONICAL_FEATURES)
            - {"temperature_c", "x_kurtosis", "z_kurtosis"}
        )
        for name in nonnegative:
            if canonical[name] < 0:
                raise InputQualityError(f"{name} não pode ser negativo.")

        if canonical["temperature_c"] < -273.15:
            raise InputQualityError("Temperatura abaixo do zero absoluto.")

        for axis in ("x", "z"):
            rms = canonical[f"{axis}_rms_velocity_mm_s"]
            peak = canonical[f"{axis}_peak_velocity_mm_s"]
            if peak + max(0.05, 0.10 * rms) < rms:
                alerts.append(
                    f"Pico de velocidade {axis.upper()} inferior ao RMS."
                )

        x_channel = [
            canonical["x_rms_velocity_mm_s"],
            canonical["x_peak_velocity_mm_s"],
            canonical["x_rms_acceleration_g"],
            canonical["x_peak_acceleration_g"],
            canonical["x_high_freq_rms_accel_g"],
        ]
        z_channel = [
            canonical["z_rms_velocity_mm_s"],
            canonical["z_peak_velocity_mm_s"],
            canonical["z_rms_acceleration_g"],
            canonical["z_peak_acceleration_g"],
            canonical["z_high_freq_rms_accel_g"],
        ]

        if canonical["rpm"] >= 60:
            x_zero = max(x_channel) <= 1e-12
            z_zero = max(z_channel) <= 1e-12
            if x_zero and z_zero:
                alerts.append(
                    "Todos os canais vibracionais estão zerados com RPM alta."
                )
            elif x_zero != z_zero:
                alerts.append(
                    "Um eixo está zerado enquanto o outro permanece ativo."
                )

        return canonical, alerts

    @staticmethod
    def engineer(canonical: Mapping[str, float]) -> dict[str, float]:
        epsilon = 1e-9
        rotational_hz = canonical["rpm"] / 60.0
        frequency_floor_hz = 1.0

        result = dict(canonical)
        result["x_regularized_order_proxy"] = (
            canonical["x_peak_vel_comp_freq_hz"]
            / (rotational_hz + frequency_floor_hz)
        )
        result["z_regularized_order_proxy"] = (
            canonical["z_peak_vel_comp_freq_hz"]
            / (rotational_hz + frequency_floor_hz)
        )

        result["log_xz_rms_velocity_ratio"] = float(
            np.clip(
                math.log(
                    (canonical["x_rms_velocity_mm_s"] + epsilon)
                    / (canonical["z_rms_velocity_mm_s"] + epsilon)
                ),
                -8,
                8,
            )
        )
        result["log_xz_rms_acceleration_ratio"] = float(
            np.clip(
                math.log(
                    (canonical["x_rms_acceleration_g"] + epsilon)
                    / (canonical["z_rms_acceleration_g"] + epsilon)
                ),
                -8,
                8,
            )
        )
        result["log_xz_high_freq_ratio"] = float(
            np.clip(
                math.log(
                    (canonical["x_high_freq_rms_accel_g"] + epsilon)
                    / (canonical["z_high_freq_rms_accel_g"] + epsilon)
                ),
                -8,
                8,
            )
        )
        return result

    def load_or_fit(self, csv_path: str | Path | None) -> dict[str, Any]:
        csv = Path(csv_path) if csv_path else None
        artifact_exists = self.artifact_path.exists()

        if artifact_exists:
            try:
                artifact = joblib.load(self.artifact_path)
                if artifact.get("artifact_version") != ARTIFACT_VERSION:
                    raise ValueError("Versão de artefato incompatível.")

                if csv is None or not csv.exists():
                    self._restore(artifact)
                    self._persist_snapshot()
                    return self.summary(source="artifact_without_csv")

                current_hash = dataset_sha256(csv)
                if current_hash == artifact.get("dataset_hash"):
                    self._restore(artifact)
                    self._persist_snapshot()
                    return self.summary(source="artifact")
            except Exception:
                if csv is None or not csv.exists():
                    raise

        if csv is None or not csv.exists():
            raise FileNotFoundError(
                "Nem artefato válido nem banner.csv foram encontrados."
            )
        return self.fit_from_csv(csv)

    def fit_from_csv(self, csv_path: str | Path) -> dict[str, Any]:
        path = Path(csv_path)
        self.dataset_hash = dataset_sha256(path)
        raw = pd.read_csv(path)

        required = {"id", "created_at", "fault"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        valid_rows: list[dict[str, Any]] = []
        self.quarantine = []

        for row_number, row in raw.iterrows():
            try:
                source_id = int(row["id"])
                created_at = pd.to_datetime(
                    row["created_at"],
                    utc=True,
                    errors="raise",
                )
                fault = normalize_label(row["fault"])
                machine_id = str(
                    row.get("machine_id", "machine_01")
                    or "machine_01"
                )

                canonical, _ = self.canonicalize(row.to_dict())
                engineered = self.engineer(canonical)
                valid_rows.append(
                    {
                        "source_id": source_id,
                        "created_at": created_at,
                        "fault": fault,
                        "machine_id": machine_id,
                        "is_problem": fault not in NON_PROBLEM_STATES,
                        **engineered,
                    }
                )
            except Exception as exc:
                self.quarantine.append(
                    {
                        "row": int(row_number + 2),
                        "reason": str(exc),
                    }
                )

        frame = pd.DataFrame(valid_rows)
        if frame.empty:
            raise ValueError("Nenhum registro válido no CSV.")
        if frame.duplicated(["machine_id", "source_id"]).any():
            raise ValueError("IDs duplicados para a mesma máquina.")

        frame = self._sessionize(frame)
        train, calibration, test, purged = self._temporal_split(frame)

        self.scaler = SafeRobustScaler().fit(
            train[list(MODEL_FEATURES)].to_numpy(dtype=np.float64)
        )
        train_scaled = self._weighted_transform(train)

        provisional_knn = NearestNeighbors(metric="euclidean")
        provisional_knn.fit(train_scaled)

        (
            self.novelty_threshold,
            self.kernel_scale,
            calibration_source,
            calibration_episodes,
        ) = self._calibrate_novelty(
            train,
            calibration,
            provisional_knn,
            train_scaled,
        )

        self.reference = pd.concat(
            [train, calibration],
            ignore_index=True,
        ).sort_values("created_at")
        self.test_records = test.sort_values("created_at").reset_index(
            drop=True
        )

        self.knn = NearestNeighbors(metric="euclidean")
        self.knn.fit(self._weighted_transform(self.reference))

        self.knowledge_cutoff = pd.Timestamp(
            self.reference["created_at"].max()
        )
        self.rpm_baselines = self._build_rpm_baselines(self.reference)
        (
            self.history_by_label,
            self.global_history,
        ) = self._build_history_aggregates(self.reference)

        max_reference = pd.Timestamp(
            self.reference["created_at"].max()
        )
        min_test = pd.Timestamp(self.test_records["created_at"].min())
        if max_reference >= min_test:
            raise RuntimeError("Split temporal inválido após purga.")

        self.split_summary = {
            "train_records": len(train),
            "calibration_records": len(calibration),
            "reference_records": len(self.reference),
            "test_records": len(test),
            "purged_records": len(purged),
            "train_episodes": int(train["episode_id"].nunique()),
            "calibration_episodes": int(
                calibration["episode_id"].nunique()
            ),
            "test_episodes": int(test["episode_id"].nunique()),
            "calibration_source": calibration_source,
            "calibration_episodes_used": calibration_episodes,
            "max_reference_timestamp": max_reference.isoformat(),
            "min_test_timestamp": min_test.isoformat(),
            "reference_ids": [
                int(value) for value in self.reference["source_id"]
            ],
            "test_ids": [
                int(value) for value in self.test_records["source_id"]
            ],
        }

        self._save()
        self._persist_snapshot()
        self._write_quarantine_report()
        return self.summary(source="trained")

    def _sessionize(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values(
            ["machine_id", "created_at"]
        ).reset_index(drop=True)
        episode_ids: dict[int, str] = {}

        for machine_id, group in frame.groupby(
            "machine_id",
            sort=False,
        ):
            previous_fault: str | None = None
            previous_time: pd.Timestamp | None = None
            counter = 0

            for index, row in group.iterrows():
                current_time = pd.Timestamp(row["created_at"])
                gap = (
                    math.inf
                    if previous_time is None
                    else (current_time - previous_time).total_seconds()
                )
                if (
                    row["fault"] != previous_fault
                    or gap > self.episode_gap_seconds
                ):
                    counter += 1
                episode_ids[index] = f"{machine_id}:episode:{counter}"
                previous_fault = str(row["fault"])
                previous_time = current_time

        frame["episode_id"] = pd.Series(episode_ids)
        return frame

    @staticmethod
    def _temporal_split(
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        episodes = (
            frame.groupby("episode_id")
            .agg(
                start=("created_at", "min"),
                end=("created_at", "max"),
            )
            .sort_values("start")
        )

        if len(episodes) < 6:
            raise ValueError(
                "São necessários pelo menos seis episódios para split "
                "temporal treino/calibração/teste."
            )

        train_count = max(2, int(len(episodes) * 0.60))
        calibration_count = max(1, int(len(episodes) * 0.20))

        train_ids = set(episodes.index[:train_count])
        provisional_calibration = episodes.iloc[
            train_count : train_count + calibration_count
        ]
        provisional_test = episodes.iloc[
            train_count + calibration_count :
        ]

        train = frame[frame["episode_id"].isin(train_ids)].copy()
        train_end = pd.Timestamp(train["created_at"].max())

        calibration_ids = set(
            provisional_calibration[
                provisional_calibration["start"] > train_end
            ].index
        )
        calibration = frame[
            frame["episode_id"].isin(calibration_ids)
        ].copy()
        if calibration.empty:
            raise ValueError("Calibração vazia após purga temporal.")

        calibration_end = pd.Timestamp(
            calibration["created_at"].max()
        )
        test_ids = set(
            provisional_test[
                provisional_test["start"] > calibration_end
            ].index
        )
        test = frame[frame["episode_id"].isin(test_ids)].copy()
        if test.empty:
            raise ValueError("Teste vazio após purga temporal.")

        selected = train_ids | calibration_ids | test_ids
        purged = frame[~frame["episode_id"].isin(selected)].copy()
        return train, calibration, test, purged

    def _weighted_transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise EngineUnavailableError("Scaler indisponível.")
        values = frame[list(MODEL_FEATURES)].to_numpy(
            dtype=np.float64
        )
        return self.scaler.transform(values) * self.feature_weights

    @staticmethod
    def _radii(
        query: np.ndarray,
        knn: NearestNeighbors,
        reference_count: int,
    ) -> np.ndarray:
        count = min(3, reference_count)
        distances, _ = knn.kneighbors(query, n_neighbors=count)
        return np.median(distances, axis=1)

    def _episode_calibration_scores(
        self,
        query_frame: pd.DataFrame,
        reference_labels: set[str],
        knn: NearestNeighbors,
        reference_count: int,
    ) -> list[float]:
        scores: list[float] = []
        for _, episode in query_frame.groupby("episode_id"):
            label = str(episode["fault"].iloc[0])
            if label not in reference_labels:
                continue

            sampled = episode
            if len(sampled) > 200:
                positions = np.linspace(
                    0,
                    len(sampled) - 1,
                    200,
                    dtype=int,
                )
                sampled = sampled.iloc[positions]

            radii = self._radii(
                self._weighted_transform(sampled),
                knn,
                reference_count,
            )
            scores.append(float(np.quantile(radii, 0.95)))
        return scores

    def _leave_one_episode_out_scores(
        self,
        train: pd.DataFrame,
    ) -> list[float]:
        scores: list[float] = []
        transformed = self._weighted_transform(train)

        for episode_id, episode in train.groupby("episode_id"):
            holdout_mask = train["episode_id"] == episode_id
            reference_mask = ~holdout_mask
            holdout_label = str(episode["fault"].iloc[0])

            if holdout_label not in set(
                train.loc[reference_mask, "fault"].astype(str)
            ):
                continue

            local_reference = transformed[reference_mask.to_numpy()]
            local_query = transformed[holdout_mask.to_numpy()]
            if len(local_reference) < 3:
                continue

            local_knn = NearestNeighbors(metric="euclidean")
            local_knn.fit(local_reference)
            radii = self._radii(
                local_query,
                local_knn,
                len(local_reference),
            )
            scores.append(float(np.quantile(radii, 0.95)))
        return scores

    def _calibrate_novelty(
        self,
        train: pd.DataFrame,
        calibration: pd.DataFrame,
        provisional_knn: NearestNeighbors,
        train_scaled: np.ndarray,
    ) -> tuple[float, float, str, int]:
        del train_scaled
        scores = self._episode_calibration_scores(
            calibration,
            set(train["fault"].astype(str)),
            provisional_knn,
            len(train),
        )
        source = "temporal_holdout"

        if len(scores) < 3:
            scores = self._leave_one_episode_out_scores(train)
            source = "leave_one_episode_out_fallback"

        if not scores:
            raise ValueError(
                "Não foi possível calibrar novidade por episódios."
            )

        threshold = max(
            float(np.quantile(scores, self.calibration_quantile)),
            1e-6,
        )
        kernel_scale = max(float(np.median(scores)), threshold / 4, 1e-6)
        return threshold, kernel_scale, source, len(scores)

    def _build_rpm_baselines(
        self,
        reference: pd.DataFrame,
    ) -> dict[str, Any]:
        baseline = reference[
            reference["fault"].isin(ACTIVE_BASELINE_STATES)
        ].copy()
        if len(baseline) < self.minimum_rpm_bin_samples:
            baseline = reference[
                reference["fault"].isin(NON_PROBLEM_STATES)
            ].copy()
        if baseline.empty:
            raise ValueError("Sem baseline não problemático.")

        def center_scale(
            frame: pd.DataFrame,
        ) -> tuple[dict[str, float], dict[str, float]]:
            values = frame[list(SEVERITY_FEATURES)]
            center = values.median()
            iqr = values.quantile(0.75) - values.quantile(0.25)
            mad = (values - center).abs().median() * 1.4826
            floor = 1e-3 * np.maximum(np.abs(center), 1.0)
            scale = np.maximum.reduce(
                [
                    iqr.to_numpy(),
                    mad.to_numpy(),
                    floor.to_numpy(),
                ]
            )
            return (
                {name: float(center[name]) for name in SEVERITY_FEATURES},
                {
                    name: float(scale[index])
                    for index, name in enumerate(SEVERITY_FEATURES)
                },
            )

        global_center, global_scale = center_scale(baseline)
        unique_rpm = baseline["rpm"].nunique()

        if unique_rpm < 4:
            edges = [-math.inf, math.inf]
        else:
            quantiles = np.quantile(
                baseline["rpm"],
                [0.25, 0.50, 0.75],
            )
            interior = sorted(set(float(value) for value in quantiles))
            edges = [-math.inf, *interior, math.inf]

        bins: list[dict[str, Any]] = []
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            rows = baseline[
                (baseline["rpm"] >= lower)
                & (baseline["rpm"] < upper)
            ]
            fallback = len(rows) < self.minimum_rpm_bin_samples
            if fallback:
                center, scale = global_center, global_scale
            else:
                center, scale = center_scale(rows)

            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "sample_count": int(len(rows)),
                    "fallback_global": fallback,
                    "center": center,
                    "scale": scale,
                }
            )

        return {
            "bins": bins,
            "global_center": global_center,
            "global_scale": global_scale,
        }

    @staticmethod
    def _build_history_aggregates(
        reference: pd.DataFrame,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        by_label: dict[str, Any] = {}

        for label, rows in reference.groupby("fault"):
            episode_starts = (
                rows.groupby("episode_id")["created_at"].min().sort_values()
            )
            record_months = (
                rows.groupby(
                    rows["created_at"].dt.strftime("%Y-%m")
                )
                .size()
                .astype(int)
                .to_dict()
            )
            episode_months = (
                episode_starts.groupby(
                    episode_starts.dt.strftime("%Y-%m")
                )
                .size()
                .astype(int)
                .to_dict()
            )
            by_label[str(label)] = {
                "record_count": int(len(rows)),
                "episode_count": int(rows["episode_id"].nunique()),
                "records_by_month": record_months,
                "episodes_by_month": episode_months,
                "first_seen": pd.Timestamp(
                    rows["created_at"].min()
                ).isoformat(),
                "last_seen": pd.Timestamp(
                    rows["created_at"].max()
                ).isoformat(),
                "operational_context": {
                    "rpm_mean": round(float(rows["rpm"].mean()), 4),
                    "temperature_c_mean": round(
                        float(rows["temperature_c"].mean()),
                        4,
                    ),
                },
            }

        global_stats = {
            "record_count": int(len(reference)),
            "episode_count": int(reference["episode_id"].nunique()),
            "label_counts": {
                str(key): int(value)
                for key, value in reference["fault"]
                .value_counts()
                .items()
            },
            "problem_records": int(reference["is_problem"].sum()),
            "non_problem_records": int((~reference["is_problem"]).sum()),
            "warning": (
                "Episódios não equivalem a MTBF sem uptime e fechamento "
                "de ordens de manutenção."
            ),
        }
        return by_label, global_stats

    def _baseline_for_rpm(self, rpm: float) -> dict[str, Any]:
        if self.rpm_baselines is None:
            raise EngineUnavailableError("Baseline RPM indisponível.")
        for rpm_bin in self.rpm_baselines["bins"]:
            if rpm_bin["lower"] <= rpm < rpm_bin["upper"]:
                return rpm_bin
        return self.rpm_baselines["bins"][-1]

    def _condition_deviation(
        self,
        canonical: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        rpm_bin = self._baseline_for_rpm(canonical["rpm"])
        deviations = {
            name: abs(
                (canonical[name] - rpm_bin["center"][name])
                / rpm_bin["scale"][name]
            )
            for name in SEVERITY_FEATURES
        }
        lower_rpm = (
            None
            if math.isinf(rpm_bin["lower"])
            else float(rpm_bin["lower"])
        )
        upper_rpm = (
            None
            if math.isinf(rpm_bin["upper"])
            else float(rpm_bin["upper"])
        )
        return deviations, {
            "lower_rpm": lower_rpm,
            "upper_rpm": upper_rpm,
            "sample_count": rpm_bin["sample_count"],
            "fallback_global": rpm_bin["fallback_global"],
        }

    def diagnose(
        self,
        data: Mapping[str, Any],
        *,
        strict_temporal: bool = True,
        asset_criticality: float = 0.5,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        if (
            self.scaler is None
            or self.knn is None
            or self.reference is None
            or self.knowledge_cutoff is None
            or self.novelty_threshold is None
            or self.kernel_scale is None
        ):
            raise EngineUnavailableError("Motor não carregado.")

        event_time = pd.to_datetime(
            data.get("created_at", datetime.now(timezone.utc)),
            utc=True,
            errors="raise",
        )
        if strict_temporal and event_time <= self.knowledge_cutoff:
            raise TemporalLeakageError(
                "O evento não é posterior ao knowledge_cutoff. "
                "Use uma amostra do split de teste ou desabilite "
                "explicitamente o modo estrito."
            )

        canonical, alerts = self.canonicalize(data)
        engineered = self.engineer(canonical)
        query_frame = pd.DataFrame([engineered])
        query = self._weighted_transform(query_frame)

        neighbor_count = min(self.k_max + 1, len(self.reference))
        with self._lock:
            distances, indices = self.knn.kneighbors(
                query,
                n_neighbors=neighbor_count,
            )

        source_id = data.get("id")
        candidates: list[tuple[float, int]] = []
        for distance, index in zip(
            distances[0],
            indices[0],
            strict=True,
        ):
            row = self.reference.iloc[int(index)]
            if (
                source_id is not None
                and int(row["source_id"]) == int(source_id)
            ):
                continue
            candidates.append((float(distance), int(index)))
            if len(candidates) == self.k_max:
                break

        if not candidates:
            raise EngineUnavailableError("Sem vizinhos disponíveis.")

        first_distances = np.array(
            [item[0] for item in candidates[:3]],
            dtype=np.float64,
        )
        novelty_distance = float(np.median(first_distances))
        is_novel = novelty_distance > self.novelty_threshold

        close = [
            item
            for item in candidates
            if item[0] <= self.novelty_threshold
        ]
        voting_set = close[: self.k_max] if close else candidates[:3]

        weights = np.array(
            [
                math.exp(
                    -0.5 * (distance / self.kernel_scale) ** 2
                )
                for distance, _ in voting_set
            ],
            dtype=np.float64,
        )
        if float(weights.sum()) < 1e-12:
            weights = 1.0 / (
                np.array([item[0] for item in voting_set]) + 1e-6
            )

        label_weights: dict[str, float] = defaultdict(float)
        for weight, (_, index) in zip(
            weights,
            voting_set,
            strict=True,
        ):
            label = str(self.reference.iloc[index]["fault"])
            label_weights[label] += float(weight)

        best_label = max(label_weights, key=label_weights.get)
        consensus = (
            label_weights[best_label] / sum(label_weights.values())
        )
        accepted = (
            not is_novel
            and bool(close)
            and consensus >= self.consensus_threshold
        )

        severe_zero_alert = any("zerado" in item for item in alerts)
        if is_novel and severe_zero_alert:
            status = "data_quality_suspect"
            inferred_label = None
        elif is_novel:
            status = "unknown_pattern"
            inferred_label = None
        elif not accepted:
            status = "inconclusive_neighborhood"
            inferred_label = None
        elif best_label in NON_PROBLEM_STATES:
            status = "non_problem_state"
            inferred_label = best_label
        else:
            status = "known_problem"
            inferred_label = best_label

        deviations, rpm_regime = self._condition_deviation(canonical)
        severity = float(
            np.clip(
                (
                    np.quantile(list(deviations.values()), 0.90)
                    - 2.0
                )
                / 6.0,
                0.0,
                1.0,
            )
        )

        history = (
            self.history_by_label.get(inferred_label)
            if inferred_label is not None
            else None
        )

        if status == "known_problem" and inferred_label is not None:
            episode_count = int(history["episode_count"]) if history else 0
            recurrence = 1.0 - math.exp(-episode_count / 5.0)
            priority = 100.0 * consensus * (
                0.50 * severity
                + 0.25 * recurrence
                + 0.25 * asset_criticality
            )
            if (
                inferred_label in self.critical_faults
                and priority >= 75
            ):
                band = "Alta criticidade — revisão imediata"
            else:
                band = "Atenção"
        elif status == "non_problem_state":
            priority = 0.0
            band = "Normal"
        else:
            priority = None
            band = "Indeterminado"

        neighbor_rows: list[dict[str, Any]] = []
        for rank, (distance, index) in enumerate(candidates, start=1):
            row = self.reference.iloc[index]
            neighbor_rows.append(
                {
                    "rank": rank,
                    "id": int(row["source_id"]),
                    "machine_id": str(row["machine_id"]),
                    "created_at": pd.Timestamp(
                        row["created_at"]
                    ).isoformat(),
                    "fault": str(row["fault"]),
                    "distance": round(distance, 6),
                    "inside_radius": distance <= self.novelty_threshold,
                }
            )

        heatmap = [
            {
                "metric": "Velocidade RMS",
                "X": deviations["x_rms_velocity_mm_s"],
                "Z": deviations["z_rms_velocity_mm_s"],
            },
            {
                "metric": "Velocidade de pico",
                "X": deviations["x_peak_velocity_mm_s"],
                "Z": deviations["z_peak_velocity_mm_s"],
            },
            {
                "metric": "Aceleração RMS",
                "X": deviations["x_rms_acceleration_g"],
                "Z": deviations["z_rms_acceleration_g"],
            },
            {
                "metric": "Aceleração de pico",
                "X": deviations["x_peak_acceleration_g"],
                "Z": deviations["z_peak_acceleration_g"],
            },
            {
                "metric": "Alta frequência",
                "X": deviations["x_high_freq_rms_accel_g"],
                "Z": deviations["z_high_freq_rms_accel_g"],
            },
            {
                "metric": "Curtose",
                "X": deviations["x_kurtosis"],
                "Z": deviations["z_kurtosis"],
            },
            {
                "metric": "Crest factor",
                "X": deviations["x_crest_factor"],
                "Z": deviations["z_crest_factor"],
            },
        ]

        provided_fault = data.get("fault")
        result = {
            "status": status,
            "inferred_label": inferred_label,
            "is_problem": status == "known_problem",
            "temporal_guard": {
                "strict": strict_temporal,
                "event_time": event_time.isoformat(),
                "knowledge_cutoff": self.knowledge_cutoff.isoformat(),
                "valid": event_time > self.knowledge_cutoff,
            },
            "target_leakage_control": {
                "provided_fault": (
                    normalize_label(provided_fault)
                    if provided_fault is not None
                    else None
                ),
                "fault_used_as_feature": False,
            },
            "quality": {
                "status": "suspect" if alerts else "valid",
                "alerts": alerts,
                "kurtosis_definition": self.kurtosis_definition,
            },
            "novelty": {
                "is_novel": is_novel,
                "distance": round(novelty_distance, 6),
                "threshold": round(self.novelty_threshold, 6),
                "calibration": self.split_summary.get(
                    "calibration_source"
                ),
            },
            "neighborhood": {
                "adaptive_k": len(voting_set),
                "close_neighbors": len(close),
                "consensus": round(consensus, 6),
                "neighbors": neighbor_rows,
            },
            "rpm_regime": rpm_regime,
            "condition_deviation": {
                name: round(value, 4)
                for name, value in deviations.items()
            },
            "heatmap": heatmap,
            "history": history,
            "maintenance_priority": {
                "score": (
                    None if priority is None else round(priority, 2)
                ),
                "band": band,
                "heuristic_not_probability": True,
                "severity": round(severity, 4),
            },
        }
        self._persist_diagnosis(data, result, trace_id)
        return json_safe(result)

    def _persist_diagnosis(
        self,
        data: Mapping[str, Any],
        result: Mapping[str, Any],
        trace_id: str | None,
    ) -> None:
        machine_id = (
            str(data["machine_id"])
            if data.get("machine_id") is not None
            else None
        )
        sequence_no = (
            int(data["sequence_no"])
            if data.get("sequence_no") is not None
            else None
        )

        with self.Session() as session:
            if machine_id is not None and sequence_no is not None:
                duplicate = session.scalar(
                    select(DiagnosisLog.id).where(
                        DiagnosisLog.machine_id == machine_id,
                        DiagnosisLog.sequence_no == sequence_no,
                    )
                )
                if duplicate is not None:
                    return

            session.add(
                DiagnosisLog(
                    trace_id=trace_id,
                    machine_id=machine_id,
                    sequence_no=sequence_no,
                    status=str(result["status"]),
                    inferred_label=result.get("inferred_label"),
                    result=json_safe(result),
                )
            )
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                duplicate = session.scalar(
                    select(DiagnosisLog.id).where(
                        DiagnosisLog.machine_id == machine_id,
                        DiagnosisLog.sequence_no == sequence_no,
                    )
                )
                if duplicate is None:
                    raise exc

    def telemetry_stats(self) -> dict[str, Any]:
        with self.Session() as session:
            online = {
                str(status): int(count)
                for status, count in session.execute(
                    select(
                        DiagnosisLog.status,
                        func.count(DiagnosisLog.id),
                    ).group_by(DiagnosisLog.status)
                ).all()
            }

        return {
            **self.global_history,
            "by_label": self.history_by_label,
            "online_diagnoses_by_status": online,
            "split": {
                key: value
                for key, value in self.split_summary.items()
                if key not in {"reference_ids", "test_ids"}
            },
            "historical_aggregation_complexity": "O(1) lookup",
        }

    def get_test_sample(
        self,
        offset: int = 0,
        *,
        label: str | None = None,
        include_fault: bool = False,
    ) -> dict[str, Any]:
        if self.test_records is None or self.test_records.empty:
            raise EngineUnavailableError("Split de teste indisponível.")

        candidates = self.test_records
        if label:
            candidates = candidates[
                candidates["fault"] == normalize_label(label)
            ]
        if candidates.empty:
            raise KeyError("Nenhuma amostra de teste para o filtro.")

        row = candidates.iloc[offset % len(candidates)]
        payload: dict[str, Any] = {
            "id": int(row["source_id"]),
            "created_at": pd.Timestamp(row["created_at"]).isoformat(),
            "machine_id": str(row["machine_id"]),
            **{
                name: float(row[name])
                for name in CANONICAL_FEATURES
            },
        }
        if include_fault:
            payload["fault"] = str(row["fault"])
        return payload

    def summary(self, *, source: str) -> dict[str, Any]:
        return {
            "source": source,
            "artifact_version": ARTIFACT_VERSION,
            "dataset_hash": self.dataset_hash,
            "knowledge_cutoff": (
                self.knowledge_cutoff.isoformat()
                if self.knowledge_cutoff is not None
                else None
            ),
            "novelty_threshold": self.novelty_threshold,
            "reference_records": (
                len(self.reference)
                if self.reference is not None
                else 0
            ),
            "test_records": (
                len(self.test_records)
                if self.test_records is not None
                else 0
            ),
            "quarantined_rows": len(self.quarantine),
            "split": {
                key: value
                for key, value in self.split_summary.items()
                if key not in {"reference_ids", "test_ids"}
            },
        }

    def _save(self) -> None:
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "artifact_version": ARTIFACT_VERSION,
            "dataset_hash": self.dataset_hash,
            "scaler": self.scaler,
            "knn": self.knn,
            "reference": self.reference,
            "test_records": self.test_records,
            "feature_weights": self.feature_weights,
            "novelty_threshold": self.novelty_threshold,
            "kernel_scale": self.kernel_scale,
            "knowledge_cutoff": self.knowledge_cutoff,
            "rpm_baselines": self.rpm_baselines,
            "history_by_label": self.history_by_label,
            "global_history": self.global_history,
            "split_summary": self.split_summary,
            "quarantine": self.quarantine,
            "kurtosis_definition": self.kurtosis_definition,
        }
        joblib.dump(artifact, self.artifact_path)

    def _restore(self, artifact: Mapping[str, Any]) -> None:
        self.dataset_hash = artifact["dataset_hash"]
        self.scaler = artifact["scaler"]
        self.knn = artifact["knn"]
        self.reference = artifact["reference"]
        self.test_records = artifact["test_records"]
        self.feature_weights = artifact["feature_weights"]
        self.novelty_threshold = artifact["novelty_threshold"]
        self.kernel_scale = artifact["kernel_scale"]
        self.knowledge_cutoff = pd.Timestamp(
            artifact["knowledge_cutoff"]
        )
        self.rpm_baselines = artifact["rpm_baselines"]
        self.history_by_label = artifact["history_by_label"]
        self.global_history = artifact["global_history"]
        self.split_summary = artifact["split_summary"]
        self.quarantine = artifact["quarantine"]
        self.kurtosis_definition = artifact.get(
            "kurtosis_definition",
            "auto",
        )

    def _persist_snapshot(self) -> None:
        if self.reference is None or self.dataset_hash is None:
            return

        with self.Session.begin() as session:
            session.execute(delete(HistoricalEvent))
            session.add_all(
                [
                    HistoricalEvent(
                        dataset_hash=self.dataset_hash,
                        machine_id=str(row.machine_id),
                        source_id=int(row.source_id),
                        created_at=pd.Timestamp(
                            row.created_at
                        ).to_pydatetime(),
                        fault=str(row.fault),
                        episode_id=str(row.episode_id),
                        payload=json_safe(
                            {
                                name: float(getattr(row, name))
                                for name in CANONICAL_FEATURES
                            }
                        ),
                    )
                    for row in self.reference.itertuples(index=False)
                ]
            )

    def _write_quarantine_report(self) -> None:
        report_path = self.artifact_path.with_suffix(
            ".quarantine.json"
        )
        report_path.write_text(
            json.dumps(
                self.quarantine,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
