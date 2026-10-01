"""Offline HTTP protocol and nonauthorization boundaries, not live JEV evidence."""
from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from self_directing_mcp import workflow_jev as jev


@pytest.fixture
def transport(monkeypatch):
    client_type = httpx.Client

    def install(handler):
        monkeypatch.setattr(jev.httpx, "Client", lambda **kwargs: client_type(
            transport=httpx.MockTransport(handler), **kwargs))

    return install


def envelope(choice="ReadyForCompletion", *, confidence=.88):
    probabilities = {label: .03 for label in jev.CHOICES.values()}
    probabilities[choice] = .91
    return {
        "model": "jev-1.13.0",
        "answers": {"next_stage": {
            "type": "choice", "choice": choice,
            "probabilities": probabilities, "confidence": confidence,
        }},
        "usage": {"input_tokens": 350, "output_tokens": 40},
    }


@pytest.mark.parametrize("choice_id,choice", list(jev.CHOICES.items()))
def test_choice_mapping_uses_labels_not_rounded_scores(transport, choice_id, choice):
    payload = envelope(choice)
    transport(lambda request: httpx.Response(200, json=payload))
    result = jev.classify({"approved_requirement": "Synthetic fixture"}, api_key="fixture-key")
    assert result.status == "ok"
    assert (result.choice_id, result.choice) == (choice_id, choice)
    assert result.probabilities == payload["answers"]["next_stage"]["probabilities"]
    assert result.confidence == .88


def test_actual_request_hash_binds_assessed_context_and_schema(transport, monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=envelope())

    transport(respond)
    first = jev.classify({"requirement": "A"}, api_key="fixture-key")
    second = jev.classify({"requirement": "B"}, api_key="fixture-key")
    monkeypatch.setattr(jev, "_INSTRUCTIONS", jev._INSTRUCTIONS + " Additional approved instruction.")
    third = jev.classify({"requirement": "A"}, api_key="fixture-key")
    assert len({first.input_hash, second.input_hash, third.input_hash}) == 3
    for request, result in zip(requests, (first, second, third)):
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.method == "POST"
        assert result.input_hash == hashlib.sha256(request.content).hexdigest()
        body = json.loads(request.content)
        question = body["questions"]["next_stage"]
        assert body["model"] == result.model == "jev-1.13.0"
        assert question["type"] == "choice"
        assert set(question["criteria"]) == set(jev.CHOICES.values())
        assert question["instructions"]["mapping_hash"] == jev.MAPPING_HASH
        assert question["instructions"]["class_mapping"] == {str(k): v for k, v in jev.CHOICES.items()}
        assert b"fixture-key" not in request.content
        assert "fixture-key" not in repr(result.as_dict())


def test_low_confidence_cannot_authorize_but_retains_observation(transport):
    payload = envelope(confidence=.699999999999)
    transport(lambda request: httpx.Response(200, json=payload))
    result = jev.classify({}, api_key="fixture-key", threshold=.7)
    assert result.status == "low_confidence"
    assert result.choice_id == 0
    assert result.confidence == .699999999999
    assert result.probabilities == payload["answers"]["next_stage"]["probabilities"]
    assert "ReadyForCompletion" in result.reason


def test_exact_threshold_is_accepted_without_rounding(transport):
    transport(lambda request: httpx.Response(200, json=envelope(confidence=.7)))
    assert jev.classify({}, api_key="fixture-key", threshold=.7).status == "ok"


@pytest.mark.parametrize("field,value", [
    ("type", "score"), ("choice", 3), ("choice", "unknown"), ("choice", []),
    ("confidence", True), ("confidence", "0.9"), ("confidence", None),
    ("confidence", float("nan")), ("confidence", float("inf")),
    ("confidence", -0.1), ("confidence", 1.1), ("confidence", 10 ** 500),
    ("probabilities", {}),
    ("probabilities", {"ReadyForCompletion": 1.0}),
    ("probabilities", dict(zip(jev.CHOICES.values(), [.03, .03, .03, True]))),
    ("probabilities", dict(zip(jev.CHOICES.values(), [.03, .03, .03, "0.91"]))),
    ("probabilities", dict(zip(jev.CHOICES.values(), [.03, .03, .03, float("nan")]))),
    ("probabilities", dict(zip(jev.CHOICES.values(), [.03, .03, .03, -.09]))),
    ("probabilities", dict(zip(jev.CHOICES.values(), [.03, .03, .03, .5]))),
    ("probabilities", dict(zip(jev.CHOICES.values(), [.91, .03, .03, .03]))),
])
def test_malformed_choice_never_authorizes(transport, field, value):
    payload = envelope()
    payload["answers"]["next_stage"][field] = value
    # Raw JSON deliberately includes NaN/Infinity to exercise invalid upstream data.
    transport(lambda request: httpx.Response(200, content=json.dumps(payload)))
    result = jev.classify({}, api_key="fixture-key")
    assert result.status == "invalid_response"
    assert result.choice_id == 0
    assert result.probabilities == {}
    assert result.confidence == 0


@pytest.mark.parametrize("payload", [None, [], {}, {"model": "jev-latest"},
    {"model": "jev-1.13.0", "answers": {}},
    {"model": "jev-1.13.0", "answers": {"next_stage": None}},
])
def test_incomplete_or_unidentified_response_is_rejected(transport, payload):
    transport(lambda request: httpx.Response(200, content=json.dumps(payload)))
    result = jev.classify({}, api_key="fixture-key")
    assert result.status == "invalid_response"
    assert result.choice_id == 0


@pytest.mark.parametrize("body", [b"not json", b'{"model":"jev-1.13.0","model":"jev-1.13.0"}'])
def test_bad_json_and_duplicate_keys_are_rejected(transport, body):
    transport(lambda request: httpx.Response(200, content=body))
    assert jev.classify({}, api_key="fixture-key").status == "invalid_response"


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_http_errors_and_redirects_do_not_leak_response_body(transport, status):
    transport(lambda request: httpx.Response(status, text="fixture-private-material",
                                             headers={"Location": "https://untrusted.invalid/"}))
    result = jev.classify({}, api_key="fixture-key")
    assert result.status == "http_error"
    assert result.choice_id == 0
    assert "fixture-private-material" not in repr(result)


@pytest.mark.parametrize("error,status", [(httpx.ReadTimeout, "timeout"),
                                          (httpx.ConnectError, "transport_error")])
def test_network_errors_are_explicit_and_sanitized(transport, error, status):
    def fail(request):
        raise error("fixture-private-material", request=request)

    transport(fail)
    result = jev.classify({}, api_key="fixture-key")
    assert result.status == status
    assert result.choice_id == 0
    assert "fixture-private-material" not in repr(result)


def test_key_file_is_authoritative_and_never_reuses_other_providers(tmp_path, monkeypatch):
    key_file = tmp_path / "keys.env"
    key_file.write_text("OPENROUTER_API_KEY=unrelated-fixture\n", encoding="utf-8")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ambient-fixture")
    result = jev.classify({}, key_file=key_file, api_key="explicit-fixture")
    assert result.status == "missing_key"
    assert result.choice_id == 0
    assert "fixture" not in repr(result)


def test_dotenv_key_is_not_interpolated(transport, tmp_path, monkeypatch):
    key_file = tmp_path / "keys.env"
    key_file.write_text("TYPESAFE_API_KEY='${UNRELATED_SECRET}'\n", encoding="utf-8")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-expanded")

    def respond(request):
        assert request.headers["Authorization"] == "Bearer ${UNRELATED_SECRET}"
        return httpx.Response(200, json=envelope())

    transport(respond)
    assert jev.classify({}, key_file=key_file).status == "ok"


def test_missing_key_never_substitutes_provider_environment(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "other-provider-fixture")
    result = jev.classify({})
    assert result.status == "missing_key"
    assert result.choice_id == 0


@pytest.mark.parametrize("context", [
    {"api_key": "do-not-send"}, {"nested": [{"password": "do-not-send"}]},
    {"text": "Bearer secret-fixture"}, {"text": "fixture-key"},
])
def test_recognized_secrets_never_reach_transport(transport, context):
    def unexpected(request):
        pytest.fail("Sensitive context reached HTTP transport")

    transport(unexpected)
    result = jev.classify(context, api_key="fixture-key")
    assert result.status == "unsafe_context"
    assert result.choice_id == 0
    assert "do-not-send" not in repr(result)


@pytest.mark.parametrize("kwargs", [
    {"model": "jev-latest"}, {"threshold": True}, {"threshold": float("nan")},
    {"threshold": -1}, {"timeout_sec": False}, {"timeout_sec": 0},
    {"timeout_sec": float("inf")}, {"timeout_sec": 10 ** 500},
])
def test_invalid_configuration_never_authorizes(kwargs):
    result = jev.classify({}, api_key="fixture-key", **kwargs)
    assert result.status == "invalid_configuration"
    assert result.choice_id == 0


def test_model_mismatch_records_actual_version_without_authorizing(transport):
    payload = envelope()
    payload["model"] = "jev-1.14.0"
    transport(lambda request: httpx.Response(200, json=payload))
    result = jev.classify({}, api_key="fixture-key")
    assert result.status == "invalid_response"
    assert result.model == "jev-1.14.0"
    assert result.choice_id == 0


@pytest.mark.parametrize("context", [{"value": float("nan")}, {"value": object()}])
def test_invalid_json_context_never_reaches_transport(transport, context):
    def unexpected(request):
        pytest.fail("Invalid JSON context reached transport")

    transport(unexpected)
    result = jev.classify(context, api_key="fixture-key")
    assert result.status in {"invalid_context", "unsafe_context"}
    assert result.choice_id == 0
