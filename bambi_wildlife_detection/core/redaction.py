# -*- coding: utf-8 -*-
"""Keeping credentials out of messages.

A log is copied into bug reports and e-mails, and an error dialog is
photographed. Neither may show an API key - but an HTTP library's error
message quotes the URL that failed, and a service that takes its key as a
query parameter (Roboflow does) has the key in that URL::

    401 Client Error: Unauthorized for url:
    https://serverless.roboflow.com/sam3/concept_segment?api_key=rf_AbC123...

:func:`redact` takes the known secrets and the usual ways of writing one out
of a text before it is shown or logged.
"""

import re
from typing import Iterable, List, Optional

#: What stands where a credential stood.
MASK = "***"

#: Settings whose value is a credential, by their key in the processing
#: configuration.
SECRET_KEYS = ("sam3_api_key", "hf_token")

#: A secret shorter than this is not searched for: masking every "a" of a
#: message helps nobody, and no real key is that short.
_MIN_LENGTH = 4

_QUERY = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|hf[_-]?token|"
    r"token)=([^&\s'\"<>]+)")
_BEARER = re.compile(r"(?i)\b(Bearer)\s+[A-Za-z0-9._~+/=-]{8,}")
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")


def redact(text, secrets: Optional[Iterable[str]] = ()) -> str:
    """*text* with every credential replaced by ``***``.

    :param secrets: the credentials in use - each occurrence is masked,
        however it got into the text. Independently of them, ``api_key=...``
        and ``token=...`` query values, ``Bearer ...`` headers and Hugging
        Face tokens are masked wherever they appear.
    """
    text = "" if text is None else str(text)
    for secret in sorted({str(s) for s in (secrets or ()) if s},
                         key=len, reverse=True):
        if len(secret) >= _MIN_LENGTH:
            text = text.replace(secret, MASK)
    text = _QUERY.sub(lambda match: f"{match.group(1)}={MASK}", text)
    text = _BEARER.sub(lambda match: f"{match.group(1)} {MASK}", text)
    return _HF_TOKEN.sub(MASK, text)


def secrets_of(config) -> List[str]:
    """The credentials a processing configuration carries."""
    if not isinstance(config, dict):
        return []
    return [str(config[key]) for key in SECRET_KEYS if config.get(key)]
