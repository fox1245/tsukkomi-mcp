"""Secret and blob masking patterns (synthetic credentials only)."""
from __future__ import annotations

import hashlib
import time

import pytest

from self_directing_mcp.security import mask
from self_directing_mcp.security.mask import mask_secrets, mask_value, register_secret

RANDOM_B64 = ("Zm9vYmFyQmF6" * 30) + "QUJDREVGZ2hpSktMTW5vUFFSU1RVVldYWVowMTIzNDU2Nzg5"

# (text, raw fragment that must disappear)
LEAKS = [
    ("key sk" "-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", "0123456789abcdef"),
    ("OPENAI_KEY sk" "-proj-AbCdEf_GhIjKl-MnOpQrStUvWxYz0123456789", "AbCdEf_GhIjKl"),
    ("x-ant sk" "-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz-0123456789_abcdefgh", "AbCdEfGhIjKl"),
    ("t gh" "p_0123456789abcdefghijklmnopqrstuvwxyzAB", "0123456789abcdefghij"),
    ("t gh" "o_0123456789abcdefghijklmnopqrstuvwxyzAB", "0123456789abcdefghij"),
    ("t gh" "u_0123456789abcdefghijklmnopqrstuvwxyzAB", "0123456789abcdefghij"),
    ("t gh" "s_0123456789abcdefghijklmnopqrstuvwxyzAB", "0123456789abcdefghij"),
    ("t gh" "r_0123456789abcdefghijklmnopqrstuvwxyzAB", "0123456789abcdefghij"),
    ("t gi" "thub_pat_11ABCDEFG0abcdefghijkl_mnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyzABCDEFG", "mnopqrstuvwxyz0123"),
    ("t gl" "pat-abcdefghij0123456789", "abcdefghij0123456789"),
    ("t xo" "xb-123456789012-abcdefghijklmnop", "abcdefghijklmnop"),
    ("t xo" "xa-2-123456789012-abcdefghijklmnop", "abcdefghijklmnop"),
    ("t xo" "xp-1-abcdefghijklm", "abcdefghijklm"),
    ("t xo" "xr-1-abcdefghijklm", "abcdefghijklm"),
    ("id AK" "IAIOSFODNN7EXAMPLE", "IOSFODNN7EXAMPLE"),
    ("id AS" "IAIOSFODNN7EXAMPLE", "IOSFODNN7EXAMPLE"),
    ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI"),
    ('{"SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "SessionToken": "IQoJb3JpZ2luX2VjEBA"}', "IQoJb3JpZ2luX2VjEBA"),
    ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI"),
    ("k AI" "zaSyA-0123456789abcdefghijklmnopqrstuv", "0123456789abcdefghij"),
    ("jwt eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c", "SflKxwRJSMeKKF2QT4fw"),
    ("jwt eyJhbGciOiJub25lIn0.eyJzdWIiOiIxMjM0NTY3ODkwIn0.", "eyJzdWIiOiIxMjM0NTY3ODkwIn0"),
    ("Authorization: Bearer abc123def456ghi", "abc123def456ghi"),
    ('{"Authorization": "Bearer abc123def456ghi"}', "abc123def456ghi"),
    ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpwYXNzd29yZA"),
    ("curl -H 'Authorization: Token abcdef123456' https://x", "abcdef123456"),
    ("Proxy-Authorization: Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpwYXNzd29yZA"),
    ("headers: Bearer abc.def.ghi", "abc.def.ghi"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAabc\nxyz123\n-----END RSA PRIVATE KEY-----", "MIIEowIBAAKCAQEA"),
    ("-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\nAAAA", "b3BlbnNzaC1rZXktdjEAAAAA"),
    ('{"k":"-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0B\\n-----END PRIVATE KEY-----\\n"}', "MIIEvQIBADANBgkq"),
    ("-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFHDBOBgkqhkiG9w0BBQ0w\n-----END ENCRYPTED PRIVATE KEY-----", "MIIFHDBOBgkq"),
    ("password=hunter2zzz", "hunter2zzz"),
    ("passwd: hunter2zzz", "hunter2zzz"),
    ("token=abc123xyz789", "abc123xyz789"),
    ("api_key=abcd1234efgh", "abcd1234efgh"),
    ("secret: s3cr3tvalue", "s3cr3tvalue"),
    ('{"password": "hunter 2zzz", "x": 1}', "hunter 2zzz"),
    ("{'api_key': 'abcd1234efgh'}", "abcd1234efgh"),
    ('{\\"password\\": \\"hunter2zzz\\"}', "hunter2zzz"),
    ('{\\"access_token\\":\\"abc123xyz789\\"}', "abc123xyz789"),
    ("GET /cb?code=1&access_token=abc123xyz789&state=ok", "abc123xyz789"),
    ("https://example.com/hook?api_key=abcd1234efgh&x=1", "abcd1234efgh"),
    ("OPENROUTER_API_KEY=abcd1234efgh", "abcd1234efgh"),
    ("export DB_PASSWORD='my pass phrase'", "my pass phrase"),
    ("CLIENT_SECRET=zzz123yyy", "zzz123yyy"),
    ("GITHUB_TOKEN=thisisnotaprefixedtoken", "thisisnotaprefixedtoken"),
    ("STRIPE_SECRET_KEY=whateverformat123", "whateverformat123"),
    ("psql postgres://admin:s3cretpw@db.example.com:5432/app", "s3cretpw"),
    ("redis://:p4ssw0rdzz@cache:6379/0", "p4ssw0rdzz"),
    ("git clone https://x-access-token:abcdefTOKEN123@github.com/o/r.git", "abcdefTOKEN123"),
    ("hf" "_abcdefghijklmnopqrstuvwxyzABCDEFGH", "abcdefghijklmnopqrstuvwxyz"),
    ("sk" "_live_4eC39HqLyjWDarjtT1zdp7dc", "4eC39HqLyjWDarjtT1zdp7dc"),
    ("rk" "_live_4eC39HqLyjWDarjtT1zdp7dc", "4eC39HqLyjWDarjtT1zdp7dc"),
    ("wh" "sec_abcdefghijklmnopqrstuvwxyz0123", "abcdefghijklmnopqrstuvwxyz"),
    ("np" "m_abcdefghijklmnopqrstuvwxyz0123456789", "abcdefghijklmnopqrstuvwxyz"),
    ("Cookie: session=abc123xyz789; theme=dark", "abc123xyz789"),
    ("Set-Cookie: id=abc123xyz789; HttpOnly", "abc123xyz789"),
    ("mysql --password hunter2zzz --host x", "hunter2zzz"),
    ("--header 'X-Api-Key: abcd1234efgh'", "abcd1234efgh"),
    ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"),
    ('{"type":"compaction","encrypted_content":"gAAAAABo_abcDEF123xyz"}', "gAAAAABo_abcDEF123xyz"),
    ('{"redacted_thinking": "opaqueBlobValue123"}', "opaqueBlobValue123"),
    ("blob " + RANDOM_B64, RANDOM_B64[:80]),
]


@pytest.mark.parametrize("text,raw", LEAKS)
def test_known_formats_are_masked_and_idempotent(text, raw):
    masked = mask_secrets(text)
    assert raw not in masked, masked
    assert mask_secrets(masked) == masked
    assert masked != text


def test_documented_replacements_are_stable():
    assert mask_secrets("key sk-ABCDEFGHIJK and Bearer abc.def.ghi and api_key=supersecret") == \
        "key sk-*** and Bearer *** and api_key=***"
    assert mask_secrets("OPENROUTER_API_KEY=abcdef") == "OPENROUTER_API_KEY=***"
    assert mask_secrets('{"api_key": "abcdef"}') == '{"api_key": "***"}'
    assert mask_secrets("AK" "IAIOSFODNN7EXAMPLE") == "AK" "IA***"
    assert mask_secrets("gh" "p_0123456789abcdefghijklmnopqrstuvwxyzAB") == "ghp_***"
    assert mask_secrets("postgres://admin:pw@h/db") == "postgres://admin:***@h/db"
    assert mask_secrets("Authorization: Bearer abcdef123456") == "Authorization: Bearer ***"
    assert mask_secrets("?token=abc&state=ok&password=pw#frag") == "?token=***&state=ok&password=***#frag"
    assert mask_secrets("--header 'X-Api-Key: abcd1234efgh'") == "--header 'X-Api-Key: ***'"


def test_ordinary_text_hashes_and_ids_are_untouched():
    plain = [
        "commit 0123456789abcdef0123456789abcdef01234567 fixes #12",
        "sha256 " + hashlib.sha256(b"x").hexdigest(),
        "uuid 123e4567-e89b-12d3-a456-426614174000",
        "chunk codex:123e4567-e89b-12d3-a456-426614174000:1024:0:0123456789abcdef",
        "max_tokens: 4096, token_type: bearer, tokens used = 10, total_token_usage {}",
        "AAAA" * 100,  # padding, not a mixed-case blob
        hashlib.sha512(b"x").hexdigest() * 3,  # lowercase hex dump
        "https://example.com/path/to/page?x=1&y=2",
        "ssh://git@github.com:org/repo.git",
        "PWD=/home/user/project",
        "risk-assessment-report task-list-items desk-lamp",
        "Basic usage of the Bearer scheme is described below.".replace("Bearer scheme", "scheme"),
        "src/self_directing_mcp/security/mask.py:42",
    ]
    for text in plain:
        assert mask_secrets(text) == text, text


def test_blob_placeholders_never_carry_payload():
    masked = mask_secrets("image: " + RANDOM_B64 + " end")
    assert RANDOM_B64[:40] not in masked and "base64 blob omitted" in masked and masked.endswith(" end")
    uri = mask_secrets("{\"image_url\": \"data:image/jpeg;base64," + RANDOM_B64 + "\"}")
    assert "Zm9vYmFy" not in uri and "base64 data omitted" in uri
    enc = mask_secrets('{"content": [{"type": "encrypted_content", "encrypted_content": "' + RANDOM_B64 + '"}]}')
    assert "Zm9vYmFy" not in enc and "encrypted content omitted" in enc


def test_mask_value_walks_json_and_honors_skip_keys():
    value = {"grant": "grant-token-abc123", "regex": "password=\\S+", "snippet": "api_key=abcd1234efgh",
             "nested": [{"Authorization: Bearer abcdef123456": ("sk-ABCDEFGHIJKL", 3)}], "n": 4, "ok": True}
    masked = mask_value(value, skip_keys={"grant", "regex"})
    assert masked["grant"] == "grant-token-abc123" and masked["regex"] == "password=\\S+"
    assert masked["snippet"] == "api_key=***"
    assert masked["nested"] == [{"Authorization: Bearer ***": ("sk-***", 3)}]
    assert masked["n"] == 4 and masked["ok"] is True
    assert mask_value(value)["regex"] == "password=***"
    assert "abcd1234efgh" in str(value)  # input is not mutated


def test_registered_exact_secret_is_masked_in_any_format(monkeypatch):
    monkeypatch.setattr(mask, "_KNOWN", ())
    register_secret("plain-format-key-0042")
    register_secret("short")  # too short to register safely
    assert mask_secrets("echo plain-format-key-0042 short") == "echo *** short"


def test_pathological_inputs_stay_bounded():
    inputs = ["a-" * 100_000, 'password: "' * 20_000, "-----BEGIN PRIVATE KEY-----\n" * 3_000,
              "A" * 400_000, "token" * 60_000, "token=" * 30_000, "://" * 60_000, "eyJ." * 30_000,
              "Bearer " * 30_000, "Authorization:" * 20_000, "?token=" * 30_000, "sk-" * 60_000,
              "\\" * 200_000, "\"" * 200_000, "--" * 100_000, "a" * 300_000 + "=" * 5,
              "encrypted_content: \"" + "x" * 300_000, "-----BEGIN PRIVATE KEY-----" + "A" * 300_000]
    started = time.perf_counter()
    for text in inputs:
        mask_secrets(text)
    assert time.perf_counter() - started < 12, "masking is not linear on adversarial input"


def test_classifier_context_guard_accepts_hashes_and_ids_but_refuses_credentials():
    from self_directing_mcp.workflow_jev import _safe_context
    digest = hashlib.sha256(b"approved").hexdigest()
    ok = {"versions": {"kernel": digest, "mapping": digest[:32], "git": "0123456789abcdef0123456789abcdef01234567"},
          "requirements": [{"id": "req-1", "text": "Reject expired sessions; max_tokens stays below 4096"}],
          "changes": [{"path": "src/self_directing_mcp/security/mask.py", "sha256": digest}], "stage": "verify"}
    assert _safe_context(ok)
    for text in ("deploy with password=hunter2", "key AK" "IAIOSFODNN7EXAMPLE", "Authorization: Basic dXNlcjpwYXNz",
                 "postgres://admin:s3cretpw@db/app", "gh" "p_0123456789abcdefghijklmnopqrstuvwxyzAB"):
        assert not _safe_context({**ok, "requirements": [{"id": "req-1", "text": text}]}), text
