from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")


def api_get(path: str, **kwargs: Any) -> Any:
    with httpx.Client(timeout=60.0) as client:
        response = client.get(f"{API_URL}{path}", **kwargs)
        if not response.is_success:
            raise RuntimeError(response.text)
        return response.json()


def api_post(path: str, **kwargs: Any) -> Any:
    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{API_URL}{path}", **kwargs)
        if not response.is_success:
            raise RuntimeError(response.text)
        return response.json()


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        st.info("Nenhuma citação liberada.")
        return

    for citation in citations:
        with st.container(border=True):
            st.markdown(
                f"### [{citation['id']}] {citation['document']}"
            )
            columns = st.columns(4)
            columns[0].metric("Versão", citation["version"])
            columns[1].metric("Página", citation["page"])
            columns[2].metric(
                "Similaridade",
                f"{citation['similarity']:.3f}",
            )
            columns[3].metric(
                "Hash",
                citation["document_hash"][:10],
            )
            st.code(citation["excerpt"], language=None)


def render_diagnosis(result: dict[str, Any]) -> None:
    diagnosis = result["diagnosis"]
    prescription = result["prescription"]

    columns = st.columns(6)
    columns[0].metric("Status", diagnosis["status"])
    columns[1].metric(
        "Estado/Falha",
        diagnosis.get("inferred_label") or "Indeterminado",
    )
    columns[2].metric(
        "Consenso",
        f"{diagnosis['neighborhood']['consensus']:.1%}",
    )
    columns[3].metric(
        "Vizinhos no raio",
        diagnosis["neighborhood"]["close_neighbors"],
    )
    columns[4].metric(
        "Prioridade",
        diagnosis["maintenance_priority"]["band"],
    )
    columns[5].metric(
        "LLM chamada",
        "SIM" if prescription.get("llm_called") else "NÃO",
    )

    st.caption(
        f"Trace ID: {result['trace_id']} · "
        f"Knowledge cutoff: "
        f"{diagnosis['temporal_guard']['knowledge_cutoff']}"
    )

    history = diagnosis.get("history")
    if history:
        first, second = st.columns(2)
        first.metric("Registros históricos", history["record_count"])
        second.metric("Episódios históricos", history["episode_count"])

        months = sorted(
            set(history["records_by_month"])
            | set(history["episodes_by_month"])
        )
        timeline = pd.DataFrame(
            {
                "mês": months,
                "registros": [
                    history["records_by_month"].get(month, 0)
                    for month in months
                ],
                "episódios": [
                    history["episodes_by_month"].get(month, 0)
                    for month in months
                ],
            }
        )
        st.plotly_chart(
            px.bar(
                timeline,
                x="mês",
                y=["registros", "episódios"],
                barmode="group",
                title="Registros versus episódios",
            ),
            use_container_width=True,
        )

    left, right = st.columns(2)
    heatmap_frame = pd.DataFrame(diagnosis["heatmap"]).set_index(
        "metric"
    )
    left.plotly_chart(
        px.imshow(
            heatmap_frame,
            text_auto=".2f",
            color_continuous_scale="RdYlGn_r",
            title="Desvio robusto por eixo e regime de RPM",
            labels={"color": "|z| robusto"},
        ),
        use_container_width=True,
    )

    radar_values = [
        float(heatmap_frame[column].mean())
        for column in ("X", "Z")
    ]
    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=[
                heatmap_frame["X"].mean(),
                heatmap_frame["X"].max(),
                diagnosis["maintenance_priority"]["severity"] * 10,
            ],
            theta=["Média X", "Máximo X", "Severidade"],
            fill="toself",
            name="X",
        )
    )
    radar.add_trace(
        go.Scatterpolar(
            r=[
                heatmap_frame["Z"].mean(),
                heatmap_frame["Z"].max(),
                diagnosis["maintenance_priority"]["severity"] * 10,
            ],
            theta=["Média Z", "Máximo Z", "Severidade"],
            fill="toself",
            name="Z",
        )
    )
    radar.update_layout(title="Cockpit vibracional relativo")
    right.plotly_chart(radar, use_container_width=True)

    st.subheader("Vizinhos históricos")
    st.dataframe(
        pd.DataFrame(diagnosis["neighborhood"]["neighbors"]),
        use_container_width=True,
        hide_index=True,
    )

    if diagnosis["quality"]["alerts"]:
        st.warning("\n".join(diagnosis["quality"]["alerts"]))

    st.subheader("Resposta prescritiva")
    st.write(prescription["answer"])
    render_citations(prescription.get("citations", []))


st.set_page_config(
    page_title="Manutenção Prescritiva",
    layout="wide",
)
st.title("Cockpit Industrial de Manutenção Prescritiva")

try:
    health = api_get("/health")
except Exception as exc:
    st.error(f"API indisponível: {exc}")
    st.stop()

provider_options = ["none"]
if health["providers"]["ollama"]:
    provider_options.append("ollama")
if health["providers"]["cloud"]:
    provider_options.append("cloud")

telemetry_tab, assistant_tab, documents_tab = st.tabs(
    [
        "1. Telemetria & Diagnóstico",
        "2. Assistente Prescritivo",
        "3. Gestão de Manuais",
    ]
)

with telemetry_tab:
    controls = st.columns([1, 1, 1, 2])
    provider = controls[0].selectbox(
        "Motor",
        provider_options,
        format_func=lambda value: {
            "none": "Somente diagnóstico",
            "ollama": "Ollama local",
            "cloud": "Nuvem autorizada",
        }[value],
    )
    label_filter = controls[1].selectbox(
        "Caso de teste",
        [
            "qualquer",
            "normal",
            "motor_desligado",
            "acelerando",
        ],
    )

    if controls[2].button("Carregar teste temporal"):
        params = {"offset": 0}
        if label_filter != "qualquer":
            params["label"] = label_filter
        st.session_state["event"] = api_get(
            "/telemetry/sample",
            params=params,
        )

    if "event" not in st.session_state:
        st.session_state["event"] = api_get(
            "/telemetry/sample",
            params={"offset": 0},
        )

    raw_json = st.text_area(
        "JSON de entrada",
        json.dumps(
            st.session_state["event"],
            indent=2,
            ensure_ascii=False,
        ),
        height=430,
    )

    run_col, unknown_col = st.columns(2)
    if run_col.button("Diagnosticar", type="primary"):
        try:
            event = json.loads(raw_json)
            event.update(
                {
                    "strict_temporal": True,
                    "include_prescription": provider != "none",
                    "llm_provider": provider,
                }
            )
            result = api_post("/diagnose", json=event)
            st.session_state["event"] = event
            st.session_state["result"] = result
        except Exception as exc:
            st.error(str(exc))

    if unknown_col.button(
        "Injetar falha sem manual — anti-alucinação"
    ):
        try:
            refusal = api_post("/demo/anti-hallucination")
            st.success(
                f"LLM chamada: "
                f"{'SIM' if refusal['llm_called'] else 'NÃO'}"
            )
            st.write(refusal["answer"])
            st.json(refusal)
        except Exception as exc:
            st.error(str(exc))

    if "result" in st.session_state:
        render_diagnosis(st.session_state["result"])

    st.divider()
    st.subheader("Telemetria MQTT ao vivo")

    @st.fragment(run_every=2.0)
    def live_panel() -> None:
        try:
            latest = api_get(
                "/telemetry/latest",
                params={"machine_id": "machine_01"},
            )
            diagnosis = latest["diagnosis"]
            cols = st.columns(4)
            cols[0].metric("Status online", diagnosis["status"])
            cols[1].metric(
                "Rótulo estável",
                diagnosis.get("inferred_label") or "-",
            )
            cols[2].metric(
                "Distância de novidade",
                diagnosis["novelty"]["distance"],
            )
            cols[3].metric(
                "Threshold",
                diagnosis["novelty"]["threshold"],
            )

            history = st.session_state.setdefault(
                "live_history",
                [],
            )
            history.append(
                {
                    "status": diagnosis["status"],
                    "consenso": diagnosis["neighborhood"]["consensus"],
                }
            )
            del history[:-30]
            chart = pd.DataFrame(history)
            st.line_chart(chart[["consenso"]])
        except Exception:
            st.info("Aguardando telemetria MQTT.")

    live_panel()

with assistant_tab:
    st.markdown(
        "O chat reutiliza o último evento e mantém o diagnóstico "
        "numérico independente da LLM."
    )
    if "event" not in st.session_state:
        st.info("Execute um diagnóstico primeiro.")
    else:
        question = st.chat_input(
            "Pergunte sobre o procedimento documentado"
        )
        if question:
            event = dict(st.session_state["event"])
            event.update(
                {
                    "question": question,
                    "include_prescription": provider != "none",
                    "llm_provider": provider,
                    "strict_temporal": True,
                }
            )
            try:
                answer = api_post("/diagnose", json=event)
                st.chat_message("assistant").write(
                    answer["prescription"]["answer"]
                )
                render_citations(
                    answer["prescription"].get("citations", [])
                )
            except Exception as exc:
                st.error(str(exc))

with documents_tab:
    st.warning(
        "Upload sempre cria um draft. A ativação usa um token "
        "administrativo de demonstração."
    )
    fault_code = st.text_input("Código da falha")
    title = st.text_input("Título")
    version = st.text_input("Versão", value="1.0")
    pdf = st.file_uploader("Manual PDF", type=["pdf"])

    if st.button("Enviar como draft") and pdf:
        try:
            uploaded = api_post(
                "/documents/upload",
                files={
                    "file": (
                        pdf.name,
                        pdf.getvalue(),
                        "application/pdf",
                    )
                },
                data={
                    "fault_code": fault_code,
                    "title": title,
                    "version": version,
                },
            )
            st.success("Documento indexado como draft")
            st.json(uploaded)
        except Exception as exc:
            st.error(str(exc))

    admin_token = st.text_input(
        "Token administrativo de demonstração",
        type="password",
    )

    try:
        documents = api_get("/documents")
        for document in documents:
            with st.container(border=True):
                columns = st.columns([3, 1, 1, 1])
                columns[0].write(
                    f"**{document['title']}**  \n"
                    f"`{document['fault_code']}` · "
                    f"v{document['version']}"
                )
                columns[1].metric(
                    "Status",
                    document["document_status"],
                )
                columns[2].metric(
                    "Chunks",
                    document["chunk_count"],
                )
                if columns[3].button(
                    "Ativar",
                    key=document["document_id"],
                    disabled=document["document_status"] == "active",
                ):
                    activated = api_post(
                        f"/documents/"
                        f"{document['document_id']}/activate",
                        headers={"X-Admin-Token": admin_token},
                    )
                    st.success(str(activated))
                    st.rerun()
    except Exception as exc:
        st.error(str(exc))
