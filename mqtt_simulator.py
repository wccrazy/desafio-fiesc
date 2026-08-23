from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import paho.mqtt.client as mqtt


def main() -> None:
    api_url = os.getenv("API_URL", "http://localhost:8000")
    mqtt_host = os.getenv("MQTT_HOST", "localhost")
    mqtt_port = int(os.getenv("MQTT_PORT", "1883"))
    topic = os.getenv(
        "MQTT_TOPIC",
        "factory/telemetry/machine_01",
    )
    machine_id = os.getenv("MACHINE_ID", "machine_01")
    interval = float(os.getenv("INTERVAL_SECONDS", "2"))

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"test-split-replay-{machine_id}",
    )
    client.connect(mqtt_host, mqtt_port, keepalive=60)
    client.loop_start()

    sequence = 0
    try:
        while True:
            try:
                with httpx.Client(timeout=20.0) as http:
                    response = http.get(
                        f"{api_url}/telemetry/sample",
                        params={
                            "offset": sequence,
                            "include_fault": False,
                        },
                    )
                    response.raise_for_status()
                    payload: dict[str, Any] = response.json()

                payload["machine_id"] = machine_id
                payload["sequence_no"] = sequence
                payload["schema_version"] = "1.0"

                publish = client.publish(
                    topic,
                    json.dumps(payload, ensure_ascii=False),
                    qos=1,
                    retain=False,
                )
                publish.wait_for_publish()
                sequence += 1
            except Exception as exc:
                print(f"Simulador aguardando API/split de teste: {exc}")

            time.sleep(interval)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
