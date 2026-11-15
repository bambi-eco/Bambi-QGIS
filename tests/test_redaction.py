# -*- coding: utf-8 -*-
"""Credentials stay out of logs and error messages (core.redaction).

Roboflow takes its API key as a query parameter, and the HTTP library quotes
the failing URL in every error - so a wrong key, a timeout or a dropped
connection wrote the key into the log, which is what gets copied into bug
reports.
"""
import pytest

from bambi_wildlife_detection.core import redaction, segmentation as seg
from bambi_wildlife_detection.core.redaction import MASK, redact

KEY = "rf_AbC123xYz987SecretKey"


def test_a_known_secret_is_masked_wherever_it_stands():
    text = f"could not post to https://x.example/run?api_key={KEY} ({KEY})"
    cleaned = redact(text, [KEY])
    assert KEY not in cleaned
    assert cleaned.count(MASK) == 2


@pytest.mark.parametrize("text", [
    "401 Client Error: Unauthorized for url: https://serverless.roboflow.com"
    "/sam3/concept_segment?api_key=rf_AbC123xYz987SecretKey",
    "Max retries exceeded with url: /sam3/concept_segment?api_key="
    "rf_AbC123xYz987SecretKey (Caused by NewConnectionError(...))",
    "GET /models?token=rf_AbC123xYz987SecretKey&page=2",
    "headers={'Authorization': 'Bearer rf_AbC123xYz987SecretKey'}",
    "access_token=rf_AbC123xYz987SecretKey",
])
def test_the_usual_ways_of_writing_a_key_are_masked_unprompted(text):
    """Also when nobody said what the key is."""
    cleaned = redact(text)
    assert KEY not in cleaned and MASK in cleaned


def test_a_hugging_face_token_is_masked():
    token = "hf_" + "a1B2c3D4" * 4
    assert token not in redact(f"token {token} was refused")


def test_the_rest_of_a_url_survives():
    cleaned = redact("for url: https://x.example/run?api_key=abcdef&page=2")
    assert cleaned == f"for url: https://x.example/run?api_key={MASK}&page=2"


def test_ordinary_text_is_left_alone():
    text = "frame 12: key frame missing, token count 3, 4 boxes"
    assert redact(text, [""]) == text
    assert redact(text, ["ab"]) == text          # too short to be a secret
    assert redact(None) == ""


def test_an_exception_is_read_as_its_message():
    assert redact(ValueError(f"bad key {KEY}"), [KEY]) == f"bad key {MASK}"


def test_the_secrets_of_a_configuration():
    config = {"sam3_api_key": KEY, "hf_token": "", "target_folder": "/x"}
    assert redaction.secrets_of(config) == [KEY]
    assert redaction.secrets_of({}) == []
    assert redaction.secrets_of(None) == []


# ---------------------------------------------------------------------------
# The Roboflow backend
# ---------------------------------------------------------------------------

class _Response:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


class _Session:
    """Answers, or fails the way ``requests`` does - quoting the URL."""

    def __init__(self, response=None, error=None):
        self.response, self.error = response, error

    def post(self, url, params=None, **_kwargs):
        if self.error is not None:
            import requests
            raise self.error(
                f"HTTPSConnectionPool(host='serverless.roboflow.com', "
                f"port=443): Max retries exceeded with url: "
                f"/sam3/concept_segment?api_key={params['api_key']}")
        return self.response

    def close(self):
        pass


def _backend(tmp_path, **session):
    backend = seg.RoboflowBackend(KEY)
    backend._session = _Session(**session)
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"not really a picture")
    return backend, str(image)


def _failure(tmp_path, **session):
    backend, image = _backend(tmp_path, **session)
    with pytest.raises(seg.SegmentationError) as caught:
        backend.segment_image(image, ["deer"], [], {}, 0.5)
    return caught


def test_a_connection_error_does_not_quote_the_key(tmp_path):
    import requests

    caught = _failure(tmp_path, error=requests.ConnectionError)
    assert KEY not in str(caught.value)
    assert "could not be reached" in str(caught.value)
    assert f"api_key={MASK}" in str(caught.value)
    # ... and the original error, which does, is not chained on
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_key_is_said_in_plain_words(tmp_path, status):
    caught = _failure(tmp_path, response=_Response(status))
    assert "did not accept the API key" in str(caught.value)
    assert KEY not in str(caught.value)


def test_a_server_error_quotes_the_answer_without_the_key(tmp_path):
    caught = _failure(tmp_path, response=_Response(
        500, text=f"internal error while handling api_key={KEY}"))
    assert "HTTP 500" in str(caught.value)
    assert KEY not in str(caught.value)


def test_an_answer_that_is_not_json_is_an_error_of_ours(tmp_path):
    caught = _failure(tmp_path, response=_Response(200, body=None))
    assert "not JSON" in str(caught.value)


def test_a_good_answer_is_read_as_before(tmp_path):
    backend, image = _backend(tmp_path, response=_Response(200, body={
        "prompt_results": [{"echo": {"text": "deer"}, "predictions": [
            {"confidence": 0.9, "masks": [[[0, 0], [1, 0], [1, 1]]]}]}]}))
    entries = backend.segment_image(image, ["deer"], [], {}, 0.5)
    assert len(entries) == 1


def test_the_request_does_not_show_its_credentials():
    request = seg.SegmentationRequest(
        target_folder="/x", modality="t", api_key=KEY, hf_token="hf_secret")
    assert KEY not in repr(request) and "hf_secret" not in repr(request)
    assert request.api_key == KEY


def test_the_worker_reports_a_failure_without_the_key():
    """Whatever step fails, and however the key got into the message - here
    the way an HTTP error quotes its URL, in the text and the traceback."""
    from bambi_wildlife_detection.bambi_processing import ProcessingWorker
    from tests.fakes import SignalRecorder

    class Failing:
        def __getattr__(self, _name):
            def method(config, progress_fn, log_fn, cancel_check):
                raise RuntimeError(
                    "401 Client Error: Unauthorized for url: https://"
                    f"serverless.roboflow.com/x?api_key={config['sam3_api_key']}")
            return method

    worker = ProcessingWorker(Failing(), "sam3_segmentation",
                              {"sam3_api_key": KEY, "hf_token": "hf_secret1"})
    for signal in ("finished", "error", "progress", "log"):
        setattr(worker, signal, SignalRecorder())
    worker.run()

    (step, text), = worker.error.calls
    assert step == "sam3_segmentation"
    assert "401 Client Error" in text and "Traceback" in text
    assert KEY not in text
    assert text.count(MASK) >= 2            # the message and the traceback
