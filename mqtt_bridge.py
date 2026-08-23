from __future__ import annotations

import json
import os
import queue
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any

import httpx
import paho.mqtt.client as mqtt


@dataclass
class MachineState:
    labels: deque[str] = field(default_factory=lambda: deque(maxlen=5))
    stable_key: str | None = None
    last_prescription_at: dict[str, float] = field(
        default_factory=dict
    )


class LabelHysteresis:
    def __init__(
        self,
        *,
        window_size: int = 5,
        minimum_votes: int = 3,
        dominance: float = 0.60,
    ) -> None:
        self.window_size = window_size
        self.minimum_votes = minimum_votes
        self.dominance = dominance
        self.states: dict[str, MachineState] = defaultdict(
            lambda: MachineState(
                labels=deque(maxlen=self.window_size)
            )
        )

    def update(
        self,
        machine_id: str,
        key: str,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        state = self.states[machine_id]
        state.labels.append(key)
        counts = Counter(state.labels)
        winner, votes = counts.most_common(1)[0]
        ratio = votes / len(state.labels)

        stable = (
            votes >= self.minimum_votes
            and ratio >= self.dominance
        )
        transitioned = stable and winner != state.stable_key
        if transitioned:
            state.stable_key = winner

        return transitioned, state.stable_key, {
            "window": list(state.labels),
            "winner": winner,
            "votes": votes,
            "ratio": round(ratio, 4),
            "stable": stable,
            "transitioned": transitioned,
        }


class LatestValueBridge:
    def __init__(self) -> None:
        self.host = os.getenv("MQTT_HOST", "localhost")
        self.port = int(os.getenv("MQTT_PORT", "1883"))
        self.topic = os.getenv(
            "MQTT_TOPIC",
            "factory/telemetry/machine_01",
        )
        self.result_topic = os.getenv(
            "MQTT_RESULT_TOPIC",
            "factory/diagnosis/machine_01",
        )
        self.api_url = os.getenv("API_URL", "http://localhost:8000")
        self.llm_provider = os.getenv("LLM_PROVIDER", "none")
        self.cooldown = float(
            os.getenv("PRESCRIPTION_COOLDOWN_SECONDS", "300")
        )

        self.messages: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self.stop_event = Event()
        self.hysteresis = LabelHysteresis(
            window_size=int(os.getenv("HYSTERESIS_WINDOW", "5")),
            minimum_votes=int(
                os.getenv("HYSTERESIS_MIN_VOTES", "3")
            ),
            dominance=float(
                os.getenv("HYSTERESIS_DOMINANCE", "0.60")
            ),
        )

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="maintenance-latest-value-bridge",
        )
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any,
    ) -> None:
        del userdata, flags, reason_code, properties
        client.subscribe(self.topic, qos=1)

    def on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        message: mqtt.MQTTMessage,
    ) -> None:
        del client, userdata
        try:
            self.messages.put_nowait(message.payload)
        except queue.Full:
            try:
                self.messages.get_nowait()
                self.messages.task_done()
            except queue.Empty:
                pass
            self.messages.put_nowait(message.payload)

    @staticmethod
    def state_key(diagnosis: dict[str, Any]) -> str:
        label = diagnosis.get("inferred_label") or "-"
        return f"{diagnosis['status']}:{label}"

    def should_prescribe(
        self,
        machine_id: str,
        diagnosis: dict[str, Any],
        transitioned: bool,
    ) -> bool:
        if not transitioned or diagnosis["status"] != "known_problem":
            return False

        label = str(diagnosis["inferred_label"])
        state = self.hysteresis.states[machine_id]
        last = state.last_prescription_at.get(label, 0.0)
        if time.monotonic() - last < self.cooldown:
            return False

        state.last_prescription_at[label] = time.monotonic()
        return True

    def process(self, raw: bytes) -> None:
        payload = json.loads(raw.decode("utf-8"))
        machine_id = str(payload.get("machine_id", "machine_01"))

        numerical_payload = {
            **payload,
            "strict_temporal": True,
            "include_prescription": False,
            "llm_provider": "none",
        }

        with httpx.Client(timeout=30.0) as http:
            response = http.post(
                f"{self.api_url}/diagnose",
                json=numerical_payload,
            )
            response.raise_for_status()
            result = response.json()

            diagnosis = result["diagnosis"]
            transitioned, stable_key, hysteresis = (
                self.hysteresis.update(
                    machine_id,
                    self.state_key(diagnosis),
                )
            )

            if self.should_prescribe(
                machine_id,
                diagnosis,
                transitioned,
            ):
                prescribed_payload = {
                    **payload,
                    "strict_temporal": True,
                    "include_prescription": True,
                    "llm_provider": self.llm_provider,
                }
                prescribed = http.post(
                    f"{self.api_url}/diagnose",
                    json=prescribed_payload,
                    timeout=180.0,
                )
                prescribed.raise_for_status()
                result = prescribed.json()

        result["hysteresis"] = {
            **hysteresis,
            "stable_key": stable_key,
            "queue_policy": "latest_value_wins",
        }
        self.client.publish(
            self.result_topic,
            json.dumps(result, ensure_ascii=False),
            qos=1,
            retain=False,
        )

    def worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                raw = self.messages.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self.process(raw)
            except Exception as exc:
                self.client.publish(
                    self.result_topic,
                    json.dumps(
                        {
                            "status": "bridge_error",
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    qos=1,
                    retain=False,
                )
            finally:
                self.messages.task_done()

    def run(self) -> None:
        Thread(target=self.worker, daemon=True).start()
        self.client.connect(self.host, self.port, keepalive=60)
        try:
            self.client.loop_forever()
        finally:
            self.stop_event.set()


if __name__ == "__main__":
    LatestValueBridge().run()
