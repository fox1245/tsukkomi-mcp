from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
import unicodedata
import uuid
from typing import Any, Iterator

from pydantic import BaseModel, Field

from self_directing_mcp.audit.patterns import ContractRegexTimeout, SEARCH_TIMEOUT_SEC, _remaining, compile_contract, contract_budget
from self_directing_mcp.audit.runner import _eligible, evidence
from self_directing_mcp.schemas import AuditEvent, ContractRule, Finding
from self_directing_mcp.security.mask import mask_secrets


_PROCESS_GRANTS_LOCK = threading.RLock()

MAX_GRANT_TTL_SEC = 3600.0

_DURATION_MULTIPLIERS = {
    "": 1.0,
    "ms": 0.001,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "d": 86400.0,
}

_BENIGN_PROBE_TEXT = "benign_unmatched_probe_token_9f8a7b6c5d4e"
_NON_NEWLINE_LINEBREAK_RE = re.compile(r"[\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]")
_SHELL_CHAIN_RE = re.compile(r"(?:&&|\|\||[;&|<>`\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]|\$\(|\$\{)")
_CANONICAL_APPROVAL_RE = re.compile(
    r"\b(?:"
    r"you\s+(?:may|can)(?!\s+(?:not|never|have|had|be|been|being|see|saw|seen|find|found|notice|observe|recall|remember|know|think|thought|wonder|ask|want|need|check|look|read|hear|imagine|expect|guess|say|note|also\s+see|also\s+find|still\s+see)\b)|"
    r"have\s+my\s+(?:explicit\s+)?permission|permission\s+(?:is\s+)?granted|"
    r"i\s+(?:hereby\s+)?(?:allow|approve|authorize|permit|grant)(?!\s+(?:no|nothing|none)\b)|"
    r"(?:is|are)(?!\s+not\b)\s+(?:allowed|approved|permitted|authorized)|"
    r"(?<!not\s)go\s+ahead(?:\s+and)?|(?<!not\s)ok(?:ay)?\s+to|(?<!not\s)feel\s+free\s+to"
    r")\b",
    re.IGNORECASE,
)
_ENG_APPROVAL_PREFIX_RE = re.compile(
    r"^\s*(?:please\s+|yes\s+)?"
    r"(?:"
    r"you\s+(?:may|can)|"
    r"(?:you\s+)?have\s+my\s+(?:explicit\s+)?permission\s+to|"
    r"permission\s+(?:is\s+)?granted\s+to|"
    r"i\s+(?:hereby\s+)?(?:allow|approve|authorize|permit|grant)(?:\s+you)?(?:\s+to)?|"
    r"go\s+ahead(?:\s+and)?|"
    r"ok(?:ay)?\s+to|"
    r"feel\s+free\s+to"
    r")"
    r"(?:\s+(?:now|once|explicitly|hereby))*"
    r"(?:\s+(?:run|execute|perform|invoke|call|use|do|proceed\s+with))?"
    r"\s*$",
    re.IGNORECASE,
)
_ENG_APPROVAL_SUFFIX_RE = re.compile(
    r"^\s*(?:(?:once|one\s+time|now|right\s+now|in\s+this\s+session|for\s+this\s+session)\s*)*$",
    re.IGNORECASE,
)
_ENG_PASSIVE_PREFIX_RE = re.compile(
    r"^\s*(?:running|executing|invoking|calling|using)?\s*$",
    re.IGNORECASE,
)
_ENG_PASSIVE_SUFFIX_RE = re.compile(
    r"^\s+(?:is|are)\s+(?:explicitly\s+|now\s+)*(?:allowed|approved|permitted|authorized)"
    r"(?:\s+(?:once|one\s+time|now|right\s+now|in\s+this\s+session|for\s+this\s+session))*\s*$",
    re.IGNORECASE,
)
_NON_ASCII_APPROVAL_RE = re.compile(
    r"(?<![不非無未勿莫안불못가-힣A-Za-z0-9_])(?:"
    r"허용합니다|허용할게|허용할게요|허용함|허락합니다|허락할게|승인합니다|승인할게|승인함|"
    r"진행하세요|진행해\s*줘|진행해라|실행하세요|실행해\s*줘|실행해도\s*좋아|실행해도\s*됩니다|"
    r"실행해도\s*돼|써도\s*됩니다|써도\s*돼|"
    r"許可します|承認します|構いません|允许执行|批准执行)(?![가-힣a-zA-Z0-9_?？])"
)
_CANONICAL_NEGATION_RE = re.compile(
    r"(?:\b(?:not|no|never|nothing|none|neither|nor|or|nope|nah|nvm|negative|do\s+not|don't|can\s+not|cannot|can't|"
    r"will\s+not|won't|should\s+not|shouldn't|must\s+not|mustn't|without|stop|avoid|"
    r"forbid|forbidden|prohibit|prohibited|refuse|deny|denied|disallow|disallowed|"
    r"reject|rejected|cancel|wait|hold\s+on|hold\s+off|unless|except|if|"
    r"when|whenever|after|before|until|as\s+long\s+as|provided|providing|in\s+case|on\s+condition|"
    r"once\s+(?!in\s+this\s+session\b)[A-Za-z]+|"
    r"only\s+if|only\s+when|only\s+after|only\s+unless|whether|disregard|ignore|"
    r"just\s+kidding|kidding|jk|abort|skip|pretend|hypothetical|hypothetically|suppose|assuming|"
    r"but|however|although|though|instead|nevermind|never\s+mind|scratch\s+that|take\s+that\s+back|take\s+it\s+back|"
    r"belay\s+that|stand\s+down|call\s+it\s+off|forget\s+it|forget\s+that|not\s+now|not\s+yet|"
    r"retract|withdraw|postpone|delay|defer|"
    r"example|manual|docs|documentation|guide|quote|quoted|says|states|written|logged)\b"
    r"|지\s*(?:마(?:세요|십시오|라|요)?|말(?:아|아\s*줘|아줘|아요|자|고|며|것)?)|하지\s*마|마세요|마라|(?<![가-힣])안(?=\s|허용|허락|승인|진행|실행|써도|돼|됨|함|해|한다|해요|합니다|할|하|쓰|써|지우|삭제|제거|건드|바꾸|바꿔|고치|고쳐)|안\s*돼|안돼|안\s*됩니다|하지\s*않|지\s*않|금지|절대로|절대|불가|불허|불승인|죽어도|말아\s*줘|말아줘|말고"
    r"|빼고|제외|이외|외에|또는|혹은|아니면|거나"
    r"|싫어|싫다|싫습니다|그만|아니야|아니요|아뇨|아닙니다|아니다|아니에요|아닌|아닐|아니라|됐어|됐습니다|관둬|집어치워|멈춰|스톱"
    r"|않|못|거부|취소|보류|잠깐|괜찮나|괜찮을까|될까|할까|인가요|나요|까요|습니까|어떻게|어째서|왜|무엇을|무슨|어디서|언제|할지|말지"
    r"|만약|만일|경우에만|가정|무시|농담|장난|중단|건너뛰|연기|미루|대신|지만|그러나|하지만|그렇지만|다만|철회"
    r"|문서|예시|매뉴얼|가이드|로그|에러|오류|주석|설명|적혀|쓰여|나와|라고|라는|이라며|인용|예제|샘플"
    r"|マニュアル|ドキュメント|例|引用|書いて|記載|文档|示例|手册|写着|または|あるいは|除く|以外|或者|或是|除外"
    r"|不|非|無|未|勿|莫|禁止|ダメ|しないで|やめて|やめろ|やだ|いやだ|嫌だ|ない|ません|ですか|算了|不要|不行|别|不可以|不能|不可|吗|呢"
    r"|\?|？)",
    re.IGNORECASE,
)
_CANDIDATE_TEMPORARY_SCOPE_RE = re.compile(
    r"(?:\b(?:once|one\s+time|this\s+time|for\s+this\s+session|in\s+this\s+session|this\s+session|temporarily|for\s+now|right\s+now|and|or)\b|"
    r"한\s*번|한번|이번만|이번에|이번\s*세션|현\s*세션|현재\s*세션|일회성|임시로|일단|지금만|및|그리고|또는|혹은|아니면|거나|빼고|제외|이외|외에|"
    r"または|あるいは|および|並びに|除く|以外|一回|一度|今回|このセッション|一時的|"
    r"或者|或是|以及|除外|一次|这次|本次会话|暂时)",
    re.IGNORECASE,
)
_CLI_FLAG_RE = re.compile(r"^--?[A-Za-z0-9][A-Za-z0-9_-]*$")
_EN_EXEC_USE_VERBS_RE = re.compile(
    r"\b(?:run|running|execute|executing|call|calling|invoke|invoking|use|using|perform|performing)\b",
    re.IGNORECASE,
)
_NON_ASCII_ALLOWED_CONNECTOR_RE = re.compile(
    r"^[\s`'\"()을를이가은는도에에서으로로만번한일회성실행명령어작업이번세션지금해줘주세요도됨됩니다좋습니다좋아돼요하자할게할게요합니다함のを一回だけ今このセッションで実行コマンド允许一次现在在本次会话中执行命令]*$"
)
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:\.+(?=\s|$)|[!?;,\n\r。！？；，]+|\b(?:but|however|although|though|except|unless)\b|하지만|그러나|그렇지만|다만)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?:\.+(?=\s|$)|[\n;!。！；]+)")
_SHELL_METACHAR_SEQ_RE = re.compile(r"(?:&&|\|\||[|;&<>()*$`\\\"'])")
_TECH_SPAN_FOR_CUES_RE = re.compile(
    r"(?<![A-Za-z0-9_./:=@+~%*-])(?=[A-Za-z0-9_./:=@+~%*-]*[0-9_./:=@+~%*-])[A-Za-z0-9_./:=@+~%*-]+(?![A-Za-z0-9_./:=@+~%*-])"
)
_VERB_SUBCATEGORY_PAIRS: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...] = (
    (
        re.compile(r"(?:삭제|제거|지워|지우|없애|削除|消去|删除|清除|\b(?:delete|deleting|remove|removing|erase|erasing|un[l]ink|r[m])\b)", re.IGNORECASE),
        re.compile(r"\b(?:delete|deleting|remove|removing|erase|erasing|un[l]ink|r[m])\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:드롭|\b(?:drop|dropping)\b)", re.IGNORECASE),
        re.compile(r"\b(?:drop|dropping)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:초기화|비우|비워|클리어|破棄|销毁|\b(?:wipe|wiping|purge|purging)\b)", re.IGNORECASE),
        re.compile(r"\b(?:wipe|wiping|purge|purging)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:포맷|フォーマット|格式化|\b(?:format|formatting|mkfs)\b)", re.IGNORECASE),
        re.compile(r"\b(?:format|formatting|mkfs)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:강제종료|종료|죽이|죽여|킬|\b(?:kill|killing|pkill)\b)", re.IGNORECASE),
        re.compile(r"\b(?:kill|killing|pkill)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:절단|트렁케이트|\b(?:truncate|truncating)\b)", re.IGNORECASE),
        re.compile(r"\b(?:truncate|truncating)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:파괴|폭파|파기|\b(?:destroy|destroying|sh[r]ed|sh[r]edding)\b)", re.IGNORECASE),
        re.compile(r"\b(?:destroy|destroying|sh[r]ed|sh[r]edding)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:수정|변경|바꾸|바꿔|고치|고쳐|편집|건드리|건드려|업데이트|変更|修正|編集|修改|更改|编辑|\b(?:modify|modifying|change|changing|edit|editing)\b)", re.IGNORECASE),
        re.compile(r"\b(?:modify|modifying|change|changing|edit|editing)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:작성|덮어쓰|덮어써|쓰기|기록|書き込み|写入|\b(?:write|writing|overwrite)\b)", re.IGNORECASE),
        re.compile(r"\b(?:write|writing|overwrite)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:백업|バックアップ|备份|\b(?:backup|back\s+up)\b)", re.IGNORECASE),
        re.compile(r"\b(?:backup|back\s+up)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:유지|보존|보관|보호|저장|維持|保存|保護|保留|保护|\b(?:preserve|preserving|keep|keeping|save|saving|protect|protecting|retain|retaining|maintain|maintaining)\b)", re.IGNORECASE),
        re.compile(r"\b(?:preserve|preserving|keep|keeping|save|saving|protect|protecting|retain|retaining|maintain|maintaining)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:복구|복원|復元|恢复|\b(?:restore|restoring)\b)", re.IGNORECASE),
        re.compile(r"\b(?:restore|restoring)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:읽기\s*전용|읽기만|\b(?:read|reading|read-only|readonly)\b)", re.IGNORECASE),
        re.compile(r"\b(?:read|reading|read-only|readonly)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:격리|이동|옮기|옮겨|\b(?:quarantine|isolate|move|moving)\b)", re.IGNORECASE),
        re.compile(r"\b(?:quarantine|isolate|move|moving)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:(?<![가-힣])(?:실행|수행|호출|구동|돌리|돌려)|実行|执行|调用|\b(?:run|running|execute|executing|call|calling|invoke|invoking|perform|performing)\b)", re.IGNORECASE),
        re.compile(r"\b(?:run|running|execute|executing|call|calling|invoke|invoking|perform|performing)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"(?:(?<![가-힣])(?:사용(?:하|해|합|할|한|함|된|되|시|치|금|만|은|는|을|를|이|가)?(?![가-힣])|쓰(?:고|며|지|면|는|니|세|십|자|라|요|다)|써(?:요|라|도|야|서|주|보))|使用|\b(?:use|using)\b)", re.IGNORECASE),
        re.compile(r"\b(?:use|using)\b", re.IGNORECASE),
    ),
)
_CANONICAL_MUST_NOT_RE = re.compile(
    r"\b(?:never|do\s+not|don't|must\s+not|forbid|prohibit|not\s+allowed\s+to)\b",
    re.IGNORECASE,
)
_CANONICAL_MUST_RE = re.compile(
    r"\b(?:always|must|required\s+to|make\s+sure\s+to|be\s+sure\s+to)\b",
    re.IGNORECASE,
)
_RAW_MUST_NOT_CUE_RE = re.compile(
    r"(?:\b(?:never|do\s+not|don't|must\s+not|forbid|prohibit|not\s+allowed\s+to)\b|"
    r"절대로|절대|지\s*(?:마(?:세요|십시오|라|요)?|말(?:아|아\s*줘|아줘|아요|자|고|며|것)?)|하지\s*마|마세요|마라|금지|불가|불허|(?<![가-힣])안\s*(?:돼|됨|됩니다|함|해|한다|해요|합니다|할|하|쓰|써|지우|삭제|제거|실행|진행|허용|허락|승인|건드|바꾸|바꿔|고치|고쳐)|하지\s*않|지\s*않|아닙니다|아니다|아니에요|아닌|아닐|아니라|말아\s*줘|말아줘|"
    r"禁止|しないで|やめて|やめろ|ダメ|不要|别|不可以|不能)",
    re.IGNORECASE,
)
_RAW_NEGATIVE_PREDICATE_CUE_RE = re.compile(
    r"(?:\b(?:never|do\s+not|don't|must\s+not|forbid|prohibit|not\s+allowed\s+to)\b|"
    r"지\s*(?:마(?:세요|십시오|라|요)?|말(?:아|아\s*줘|아줘|아요|자|고|며|것)?)|하지\s*마|마세요|마라|금지|불가|불허|(?<![가-힣])안\s*(?:돼|됨|됩니다|함|해|한다|해요|합니다|할|하|쓰|써|지우|삭제|제거|실행|진행|허용|허락|승인|건드|바꾸|바꿔|고치|고쳐)|하지\s*않|지\s*않|아닙니다|아니다|아니에요|아닌|아닐|아니라|말아\s*줘|말아줘|"
    r"禁止|しないで|やめて|やめろ|ダメ|不要|别|不可以|不能)",
    re.IGNORECASE,
)
_RAW_AFFIRMATIVE_REQUEST_RE = re.compile(
    r"(?:해\s*줘|해줘|해주세요|해\s*주세요|하세요|십시오|할\s*것)\s*[.!。！]*$",
    re.IGNORECASE,
)
_RAW_MUST_CUE_RE = re.compile(
    r"(?:\b(?:always|must|required\s+to|make\s+sure\s+to|be\s+sure\s+to)\b|"
    r"항상|반드시|꼭|필수|해야|만\s*사용|만\s*써|"
    r"必ず|常に|必須|必须|务必|一定)",
    re.IGNORECASE,
)
_RAW_POSITIVE_ONLY_CUE_RE = re.compile(
    r"(?:\b(?:always|must|required\s+to|make\s+sure\s+to|be\s+sure\s+to)\b|"
    r"필수|해야|만\s*사용|만\s*써|"
    r"必須|必须)",
    re.IGNORECASE,
)
_EN_DESTRUCTIVE_VERBS_RE = re.compile(
    r"\b(?:delete|deleting|remove|removing|drop|dropping|destroy|destroying|wipe|wiping|purge|purging|erase|erasing|kill|killing|truncate|truncating|format|formatting|sh[r]ed|sh[r]edding)\b",
    re.IGNORECASE,
)
_RAW_DESTRUCTIVE_CUE_RE = re.compile(
    r"(?:삭제|제거|지워|지우|드롭|초기화|포맷|강제종료|파괴|폭파|削除|消去|破棄|删除|清除|销毁|"
    r"\b(?:delete|deleting|remove|removing|drop|dropping|destroy|destroying|wipe|wiping|purge|purging|erase|erasing|kill|killing|truncate|truncating|format|formatting|sh[r]ed|sh[r]edding)\b)",
    re.IGNORECASE,
)
_EN_MUTATION_VERBS_RE = re.compile(
    r"\b(?:modify|modifying|change|changing|edit|editing|write|writing)\b",
    re.IGNORECASE,
)
_RAW_MUTATION_CUE_RE = re.compile(
    r"(?:수정|변경|바꾸|바꿔|고치|고쳐|편집|작성|덮어쓰|건드리|건드려|업데이트|変更|修正|編集|書き込み|修改|更改|编辑|写入|"
    r"\b(?:modify|modifying|change|changing|edit|editing|write|writing)\b)",
    re.IGNORECASE,
)
_RAW_PRESERVATIVE_CUE_RE = re.compile(
    r"(?:백업|유지|보존|보관|보호|저장|복구|복원|읽기\s*전용|격리|이동|옮기|옮겨|バックアップ|維持|保存|保護|备份|保留|保护|"
    r"\b(?:preserve|preserving|keep|keeping|save|saving|backup|back\s+up|protect|protecting|retain|retaining|maintain|maintaining|restore|restoring|read|reading|read-only|readonly|quarantine|isolate|move|moving)\b)",
    re.IGNORECASE,
)
_EN_PRESERVATIVE_VERBS_RE = re.compile(
    r"\b(?:preserve|preserving|keep|keeping|save|saving|backup|back\s+up|protect|protecting|retain|retaining|maintain|maintaining|restore|restoring|read|reading|read-only|readonly|quarantine|isolate|move|moving)\b",
    re.IGNORECASE,
)
_EN_CONSULT_VERBS_RE = re.compile(
    r"\b(?:ask|asking|confirm|confirming|verify|verifying|check|checking|consult|consulting|approval|permission)\b",
    re.IGNORECASE,
)
_RAW_CONSULT_CUE_RE = re.compile(
    r"(?:물어봐|물어보|묻지|묻고|확인|검증|질문|승인|허락|동의|상의|確認|質問|承認|許可|询问|确认|批准|"
    r"\b(?:ask|asking|confirm|confirming|verify|verifying|check|checking|consult|consulting|approval|permission)\b)",
    re.IGNORECASE,
)
_EN_ARBITRARY_RE = re.compile(
    r"\b(?:arbitrarily|arbitrary)\b",
    re.IGNORECASE,
)
_RAW_ARBITRARY_CUE_RE = re.compile(
    r"(?:임의|함부로|마음대로|멋대로|무단|勝手に|任意|随便|擅自|\b(?:arbitrarily|arbitrary)\b)",
    re.IGNORECASE,
)
_EN_EXTRA_DESTRUCTIVE_RE = re.compile(
    r"\b(?:r[m]|r[m]dir|un[l]ink|sh[r]ed|sudo|chmod|chown|dd|mv|overwrite|uninstall|pkill|mkfs)\b",
    re.IGNORECASE,
)
_ALLOWED_CANONICAL_PROSE_WORDS = frozenset({
    "always", "never", "must", "not", "do", "don", "dont", "should", "shouldn",
    "can", "cannot", "may", "will", "won", "required", "require", "requires",
    "be", "sure", "to", "forbid", "forbidden", "prohibit", "prohibited",
    "allow", "allowed", "approve", "approved", "authorize", "authorized", "permit", "permitted",
    "ask", "asking", "confirm", "confirming", "verify", "verifying", "check", "checking",
    "consult", "consulting", "user", "owner", "human", "operator",
    "before", "after", "first", "prior", "permission", "approval", "without", "instead",
    "arbitrarily", "arbitrary",
    "run", "running", "execute", "executing", "call", "calling", "invoke", "invoking",
    "use", "using", "perform", "performing",
    "modify", "modifying", "change", "changing", "edit", "editing", "write", "writing",
    "read", "reading", "readonly", "quarantine", "isolate", "move", "moving",
    "delete", "deleting", "remove", "removing", "drop", "dropping", "destroy", "destroying",
    "wipe", "wiping", "purge", "purging", "erase", "erasing", "kill", "killing",
    "truncate", "truncating", "format", "formatting", "preserve", "preserving",
    "keep", "keeping", "save", "saving", "backup", "protect", "protecting",
    "retain", "retaining", "maintain", "maintaining", "restore", "restoring",
    "repository", "repo", "directory", "dir", "file", "files", "path", "paths",
    "folder", "session", "target", "command", "commands", "tool", "tools", "action", "actions",
    "the", "a", "an", "in", "on", "at", "from", "with", "by", "of",
    "for", "into", "onto", "this", "that", "these", "those",
    "our", "your", "my", "their", "its", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "please",
})
_FORBIDDEN_ANCHOR_JSON_KEYS = frozenset({
    "tool_name", "command", "CommandLine", "cmd", "arguments", "call_id",
    "function", "tool_calls", "input", "output", "tool_use_id",
})
_STANDALONE_AFFIRMATIVE_CLAUSE_RE = re.compile(
    r"^\s*(?:yes|ok|okay|sure|alright|certainly|approved|granted|네|예|좋아|좋아요|좋습니다|그래|그래요|응|알겠어|알겠습니다|はい|ええ|了解|好的|好|可以)\s*$",
    re.IGNORECASE,
)
_ALLOWED_ESCAPED_PUNCT = frozenset(r".^$*+?{}[]\|()/-_=:;,@%~")
_FORBIDDEN_UNESCAPED_META = frozenset(".*+?{}[]|()")
_LEADING_TOOL_HEADER_RE = re.compile(
    r"^(?:\[(?:proposed_)?tool_call:[^\]\r\n]*\](?:[ \t]*\n[ \t]*|[ \t]+)|tool_call[ \t]+name=[^\r\n]*\n[ \t]*)",
    re.IGNORECASE,
)
_NON_USER_CHUNK_HEADER_RE = re.compile(
    r"^\s*\[(?:assistant|system|developer|tool(?:_call|_result)?|proposed_tool_call|function_call|turn(?![:\w])|"
    r"response_item:(?!message:user\b)[A-Za-z0-9_.:/-]+|"
    r"turn:(?!user\b)[A-Za-z0-9_.:/-]+|"
    r"message:(?!user\b)[A-Za-z0-9_.:/-]+|"
    r"event_msg:(?!user_message\b)[A-Za-z0-9_.:/-]+)\]",
    re.IGNORECASE,
)
_EMBEDDED_ROLE_OR_TOOL_HEADER_RE = re.compile(
    r"(?:\[(?:assistant|system|developer|user|turn|message|response_item|event_msg|tool(?:_call|_result)?|proposed_tool_call|function_call)(?:[:\s][^\]]*)?\]"
    r"|\btool_call[ \t]+name=)",
    re.IGNORECASE | re.MULTILINE,
)
_LEADING_CHUNK_HEADER_RE = re.compile(
    r"^\s*\[(?:response_item:message:user|turn:user|user|message:user|event_msg:user_message)\]\s*\n?",
    re.IGNORECASE,
)


def _has_disallowed_unicode_or_homoglyphs(text: str) -> bool:
    """Reject non-NFKC text (e.g. fullwidth Latin), non-ASCII scripts/digits/symbols outside Hangul/Kana/CJK, and token-glued script mixing."""
    if not isinstance(text, str):
        return True
    if unicodedata.normalize("NFKC", text) != text:
        return True
    if re.search(r"[A-Za-z0-9][^\x00-\x7F]+[A-Za-z0-9]", text) is not None:
        return True
    if re.search(r"(?:[_./:=@+~%*-][^\x00-\x7F]|[^\x00-\x7F](?:[A-Za-z0-9_/:=@+~%*-]|\.(?=[A-Za-z0-9_./:=@+~%*-])))", text) is not None:
        return True
    for m_glued in re.finditer(r"[A-Za-z0-9]([^\x00-\x7F]+)", text):
        if not re.match(
            r"^(?:을|를|이|가|은|는|도|만|에|에서|으로|로|과|와|이나|나|의|부터|까지|처럼|보다|라고|라는|を|は|が|に|で|と|も|だけ|のみ|から|まで)[。！，；：]*$",
            m_glued.group(1),
        ):
            return True
    for ch in text:
        code = ord(ch)
        if code < 128:
            if code < 32 and ch not in ("\n", "\t"):
                return True
            if code == 127:
                return True
            continue
        if ch in "。！，；：":
            continue
        cat = unicodedata.category(ch)
        if cat.startswith(("C", "M", "Z", "N", "S", "P")):
            return True
        is_hangul = 0xAC00 <= code <= 0xD7A3
        is_hiragana_katakana = 0x3040 <= code <= 0x30FF
        is_cjk = 0x4E00 <= code <= 0x9FFF
        if not (is_hangul or is_hiragana_katakana or is_cjk):
            return True
    return False


def _strip_chunk_header(text: str) -> str:
    """Strip leading indexed user chunk header and unwrap structured user message content JSON if present; reject non-user headers/roles."""
    if not isinstance(text, str) or not text.strip() or len(text) > 4096:
        return ""
    if _NON_USER_CHUNK_HEADER_RE.search(text) is not None or _LEADING_TOOL_HEADER_RE.search(text) is not None:
        return ""
    had_user_header = _LEADING_CHUNK_HEADER_RE.search(text) is not None
    body = _LEADING_CHUNK_HEADER_RE.sub("", text, count=1).strip()
    if (
        _NON_USER_CHUNK_HEADER_RE.search(body) is not None
        or _LEADING_TOOL_HEADER_RE.search(body) is not None
        or _EMBEDDED_ROLE_OR_TOOL_HEADER_RE.search(body) is not None
    ):
        return ""
    if body.startswith("[") or body.startswith("{"):
        try:
            parsed = json.loads(body)
        except Exception:
            return ""
        if isinstance(parsed, list):
            if not parsed:
                return ""
            parts: list[str] = []
            for item in parsed:
                if not isinstance(item, dict):
                    return ""
                if any(k in _FORBIDDEN_ANCHOR_JSON_KEYS for k in item):
                    return ""
                role = item.get("role")
                role_str = str(role).strip().lower() if role is not None else None
                if role_str is not None and role_str != "user":
                    return ""
                if role_str is None and not had_user_header:
                    return ""
                raw_type = item.get("type")
                type_str = str(raw_type).strip().lower() if raw_type is not None else ""
                if type_str not in ("text", "input_text"):
                    return ""
                txt = item.get("text")
                if not isinstance(txt, str) or not txt.strip():
                    return ""
                if (
                    _NON_USER_CHUNK_HEADER_RE.search(txt) is not None
                    or _LEADING_TOOL_HEADER_RE.search(txt) is not None
                    or _EMBEDDED_ROLE_OR_TOOL_HEADER_RE.search(txt) is not None
                    or _has_disallowed_unicode_or_homoglyphs(txt)
                ):
                    return ""
                parts.append(txt.strip())
            if not parts:
                return ""
            joined = "\n".join(parts).strip()
            if (
                _NON_USER_CHUNK_HEADER_RE.search(joined) is not None
                or _LEADING_TOOL_HEADER_RE.search(joined) is not None
                or _EMBEDDED_ROLE_OR_TOOL_HEADER_RE.search(joined) is not None
                or _has_disallowed_unicode_or_homoglyphs(joined)
            ):
                return ""
            return joined
        if isinstance(parsed, dict):
            if any(k in _FORBIDDEN_ANCHOR_JSON_KEYS for k in parsed):
                return ""
            role = parsed.get("role")
            role_str = str(role).strip().lower() if role is not None else None
            if role_str is not None and role_str != "user":
                return ""
            if role_str is None and not had_user_header:
                return ""
            raw_type = parsed.get("type")
            type_str = str(raw_type).strip().lower() if raw_type is not None else ""
            if type_str not in ("text", "input_text"):
                return ""
            txt = parsed.get("text")
            if isinstance(txt, str) and txt.strip():
                if (
                    _NON_USER_CHUNK_HEADER_RE.search(txt) is not None
                    or _LEADING_TOOL_HEADER_RE.search(txt) is not None
                    or _EMBEDDED_ROLE_OR_TOOL_HEADER_RE.search(txt) is not None
                    or _has_disallowed_unicode_or_homoglyphs(txt)
                ):
                    return ""
                return txt.strip()
            return ""
        return ""
    if _has_disallowed_unicode_or_homoglyphs(body):
        return ""
    return body


def _extract_shell_tokens(text: str) -> list[str]:
    """Extract full shell/technical tokens preserving internal dots/colons/equals/globs while stripping trailing sentence punctuation."""
    raw_tokens = re.findall(r"[A-Za-z0-9_./:=@+~%*-]+", text)
    out: list[str] = []
    for tok in raw_tokens:
        cleaned = tok.rstrip(".,;:!?")
        if len(cleaned) >= 2:
            out.append(cleaned.lower())
    return out


def _extract_tech_tokens(text: str) -> list[str]:
    return [
        tok
        for tok in _extract_shell_tokens(text)
        if (
            "_" in tok
            or "/" in tok
            or "." in tok
            or ":" in tok
            or "=" in tok
            or "*" in tok
            or tok.startswith("--")
            or any(c.isdigit() for c in tok)
        )
        and len(tok) >= 3
    ]


def _extract_tech_tokens_exact(text: str) -> list[str]:
    """Extract technical tokens preserving original case so case-sensitive paths/identifiers/globs cannot drift."""
    raw_tokens = re.findall(r"[A-Za-z0-9_./:=@+~%*-]+", text)
    out: list[str] = []
    for tok in raw_tokens:
        cleaned = tok.rstrip(".,;:!?")
        if (
            len(cleaned) >= 3
            and (
                "_" in cleaned
                or "/" in cleaned
                or "." in cleaned
                or ":" in cleaned
                or "=" in cleaned
                or "*" in cleaned
                or cleaned.startswith("--")
                or any(c.isdigit() for c in cleaned)
            )
        ):
            out.append(cleaned)
    return out


def _strip_tech_tokens_for_cues(text: str) -> str:
    """Remove technical tokens (paths, filenames, flags, globs, identifiers) so path basenames cannot satisfy prose verb cues."""
    return _TECH_SPAN_FOR_CUES_RE.sub(" ", text)


def extract_canonical_english_candidates(
    *,
    source_event_id: str,
    raw_quote: str,
    canonical_english: str,
) -> list[dict[str, Any]]:
    """Extract normative contract candidates from an Orchestration-LLM Canonical English turn anchored to source_event_id."""
    if not isinstance(raw_quote, str) or not raw_quote.strip() or len(raw_quote) > 4096:
        return []
    if "?" in raw_quote or "？" in raw_quote:
        return []
    raw_body = _strip_chunk_header(raw_quote)
    if not raw_body or len(raw_body) > 200:
        return []
    if not isinstance(canonical_english, str) or not canonical_english.strip():
        return []
    eng_body = canonical_english.strip()
    if "?" in raw_body or "？" in raw_body or "?" in eng_body or "？" in eng_body:
        return []
    if not eng_body or len(eng_body) > 200 or not all(ord(ch) < 128 for ch in eng_body):
        return []
    if _has_disallowed_unicode_or_homoglyphs(raw_body) or _has_disallowed_unicode_or_homoglyphs(eng_body):
        return []
    if raw_body != mask_secrets(raw_body) or eng_body != mask_secrets(eng_body):
        return []
    if _SHELL_METACHAR_SEQ_RE.findall(raw_body) != _SHELL_METACHAR_SEQ_RE.findall(eng_body):
        return []
    raw_shell_set = set(_extract_shell_tokens(raw_body))
    raw_tech_tokens = _extract_tech_tokens(raw_body)
    turn_tech_tokens = _extract_tech_tokens(eng_body)
    if not raw_tech_tokens:
        return []
    if any(tok not in raw_shell_set for tok in turn_tech_tokens):
        return []
    if set(raw_tech_tokens) != set(turn_tech_tokens) or raw_tech_tokens != turn_tech_tokens:
        return []
    if _extract_tech_tokens_exact(raw_body) != _extract_tech_tokens_exact(eng_body):
        return []
    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(raw_body) if s and s.strip()]
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(eng_body) if s and s.strip()]
    if not sentences or len(sentences) > 4 or len(raw_sentences) != len(sentences):
        return []
    out: list[dict[str, Any]] = []
    for idx, sent in enumerate(sentences):
        raw_sent = raw_sentences[idx]
        if "?" in raw_sent or "？" in raw_sent or "?" in sent or "？" in sent:
            return []
        if _SHELL_METACHAR_SEQ_RE.findall(raw_sent) != _SHELL_METACHAR_SEQ_RE.findall(sent):
            return []
        if _CANDIDATE_TEMPORARY_SCOPE_RE.search(sent) is not None or _CANDIDATE_TEMPORARY_SCOPE_RE.search(raw_sent) is not None:
            return []
        raw_sent_shell = set(_extract_shell_tokens(raw_sent))
        sent_shell = set(_extract_shell_tokens(sent))
        raw_sent_tech_list = _extract_tech_tokens(raw_sent)
        raw_sent_tech = set(raw_sent_tech_list)
        tech_tokens = _extract_tech_tokens(sent)
        sent_tech = set(tech_tokens)
        non_flag_sent_tech = [t for t in tech_tokens if not _CLI_FLAG_RE.match(t)]
        non_flag_raw_tech = [t for t in raw_sent_tech_list if not _CLI_FLAG_RE.match(t)]
        if (
            len(non_flag_sent_tech) > 1
            or len(non_flag_raw_tech) > 1
            or len(sent_tech) != len(tech_tokens)
            or len(raw_sent_tech) != len(raw_sent_tech_list)
        ):
            return []
        if any(tok not in raw_shell_set for tok in tech_tokens):
            return []
        if raw_sent_tech != sent_tech or raw_sent_tech_list != tech_tokens:
            return []
        if _extract_tech_tokens_exact(raw_sent) != _extract_tech_tokens_exact(sent):
            return []
        if not raw_sent_shell.issubset(sent_shell):
            return []
        if all(ord(ch) < 128 for ch in raw_sent) and sent_shell != raw_sent_shell:
            return []
        extra_shell = sent_shell - raw_sent_shell
        if any(not tok.isalpha() or tok not in _ALLOWED_CANONICAL_PROSE_WORDS for tok in extra_shell):
            return []
        raw_prose = _strip_tech_tokens_for_cues(raw_sent)
        en_prose = _strip_tech_tokens_for_cues(sent)
        if (_EN_EXTRA_DESTRUCTIVE_RE.search(en_prose) is not None) != (_EN_EXTRA_DESTRUCTIVE_RE.search(raw_prose) is not None):
            return []
        for raw_sub_re, en_sub_re in _VERB_SUBCATEGORY_PAIRS:
            if (raw_sub_re.search(raw_prose) is not None) != (en_sub_re.search(en_prose) is not None):
                return []
        has_en_destructive = _EN_DESTRUCTIVE_VERBS_RE.search(en_prose) is not None
        has_raw_destructive = _RAW_DESTRUCTIVE_CUE_RE.search(raw_prose) is not None
        has_en_mutation = _EN_MUTATION_VERBS_RE.search(en_prose) is not None
        has_raw_mutation = _RAW_MUTATION_CUE_RE.search(raw_prose) is not None
        has_en_preservative = _EN_PRESERVATIVE_VERBS_RE.search(en_prose) is not None
        has_raw_preservative = _RAW_PRESERVATIVE_CUE_RE.search(raw_prose) is not None
        has_en_exec = _EN_EXEC_USE_VERBS_RE.search(en_prose) is not None
        has_en_consult = _EN_CONSULT_VERBS_RE.search(en_prose) is not None
        has_raw_consult = _RAW_CONSULT_CUE_RE.search(raw_prose) is not None
        has_en_arbitrary = _EN_ARBITRARY_RE.search(en_prose) is not None
        has_raw_arbitrary = _RAW_ARBITRARY_CUE_RE.search(raw_prose) is not None
        if has_en_destructive != has_raw_destructive:
            return []
        if has_en_mutation != has_raw_mutation:
            return []
        if has_en_preservative != has_raw_preservative:
            return []
        if has_en_consult != has_raw_consult:
            return []
        if has_en_arbitrary != has_raw_arbitrary:
            return []
        if sum((has_en_destructive, has_en_mutation, has_en_preservative, has_en_exec)) > 1:
            return []
        if not raw_sent_tech and not (has_en_consult and has_raw_consult):
            return []
        if _CANONICAL_MUST_NOT_RE.search(sent):
            sent_without_cue = _CANONICAL_MUST_NOT_RE.sub(" ", sent)
            if _CANONICAL_MUST_RE.search(sent_without_cue) is not None:
                return []
            if (
                _RAW_MUST_NOT_CUE_RE.search(raw_sent) is None
                or _RAW_NEGATIVE_PREDICATE_CUE_RE.search(raw_sent) is None
                or _RAW_AFFIRMATIVE_REQUEST_RE.search(raw_sent) is not None
                or _RAW_POSITIVE_ONLY_CUE_RE.search(raw_sent) is not None
            ):
                return []
            raw_without_cue = _RAW_MUST_NOT_CUE_RE.sub(" ", raw_sent)
            if _CANONICAL_NEGATION_RE.search(sent_without_cue) is not None or _CANONICAL_NEGATION_RE.search(raw_without_cue) is not None:
                return []
            suggested_type = "must_not"
        elif _CANONICAL_MUST_RE.search(sent):
            if _CANONICAL_NEGATION_RE.search(sent) is not None or _CANONICAL_NEGATION_RE.search(raw_sent) is not None:
                return []
            if _RAW_MUST_CUE_RE.search(raw_sent) is None or _RAW_MUST_NOT_CUE_RE.search(raw_sent) is not None:
                return []
            suggested_type = "must"
        else:
            return []
        slug_words = re.findall(r"[a-z0-9]+", sent.lower())[:6]
        slug = "-".join(slug_words) if slug_words else f"candidate-{idx + 1}"
        out.append({
            "candidate_id": f"turn_{source_event_id}_{idx + 1}",
            "suggested_id": slug,
            "suggested_type": suggested_type,
            "description": sent,
            "source_event_id": source_event_id,
            "source_quote": raw_sent,
            "canonical_english_quote": sent,
        })
    return out


_HELD_GRANT_LOCKS: dict[str, tuple[threading.RLock, list[Any]]] = {}


def _reset_grant_locks_in_child() -> None:
    """Reset in-process grant locks and close inherited lock file descriptors in a forked child so the parent's flock can never stick."""
    global _PROCESS_GRANTS_LOCK
    _PROCESS_GRANTS_LOCK = threading.RLock()
    for _, state in list(_HELD_GRANT_LOCKS.values()):
        fd = state[0]
        state[0] = None
        state[1] = 0
        if isinstance(fd, int):
            try:
                os.close(fd)
            except OSError:
                pass
    _HELD_GRANT_LOCKS.clear()


if hasattr(os, "register_at_fork"):
    try:
        os.register_at_fork(after_in_child=_reset_grant_locks_in_child)
    except RuntimeError:  # pragma: no cover
        pass


def _acquire_verified_lock_fd(lock_path: Path) -> int:
    """Open lock_path via pinned parent dir_fd with O_NOFOLLOW | O_CLOEXEC, acquire LOCK_EX, and verify ownership/inode/mode."""
    assert fcntl is not None
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    dir_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    dir_fd = os.open(str(lock_path.parent), dir_flags)
    try:
        dir_stat = os.fstat(dir_fd)
        euid = os.geteuid() if hasattr(os, "geteuid") else None
        if not stat.S_ISDIR(dir_stat.st_mode) or (euid is not None and dir_stat.st_uid != euid):
            raise RuntimeError(f"unsafe lock directory ownership or type for {lock_path.parent}")
        os.fchmod(dir_fd, 0o700)
        flags = (
            os.O_CREAT
            | os.O_RDWR
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for _ in range(8):
            fd = os.open(lock_path.name, flags, 0o600, dir_fd=dir_fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                fd_stat = os.fstat(fd)
                try:
                    path_stat = os.stat(lock_path.name, dir_fd=dir_fd, follow_symlinks=False)
                except FileNotFoundError:
                    os.close(fd)
                    continue
                if (
                    not stat.S_ISREG(fd_stat.st_mode)
                    or fd_stat.st_dev != path_stat.st_dev
                    or fd_stat.st_ino != path_stat.st_ino
                ):
                    os.close(fd)
                    continue
                if euid is not None and fd_stat.st_uid != euid:
                    raise RuntimeError(f"unsafe lock file owner for {lock_path}")
                if fd_stat.st_nlink != 1 or path_stat.st_nlink != 1:
                    raise RuntimeError(f"unsafe hardlinked lock file for {lock_path}")
                os.fchmod(fd, 0o600)
                return fd
            except BaseException:
                os.close(fd)
                raise
    finally:
        os.close(dir_fd)
    raise RuntimeError(f"failed to acquire stable lock inode for {lock_path}")


@contextmanager
def grants_lock(path: Path) -> Iterator[None]:
    """Cross-process and re-entrant in-process exclusive lock for atomic grants.json load-modify-save cycles."""
    if fcntl is None:
        raise RuntimeError("grants_lock requires POSIX fcntl file locking support")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.with_suffix(".lock")
    key = str(lock_path.resolve(strict=False))
    with _PROCESS_GRANTS_LOCK:
        entry = _HELD_GRANT_LOCKS.get(key)
        if entry is None:
            entry = (threading.RLock(), [None, 0])
            _HELD_GRANT_LOCKS[key] = entry
        path_rlock, state = entry
    with path_rlock:
        if state[1] == 0:
            state[0] = _acquire_verified_lock_fd(lock_path)
        else:
            held_fd = state[0]
            if not isinstance(held_fd, int):
                raise RuntimeError(f"missing held lock descriptor for {lock_path}")
            fd_stat = os.fstat(held_fd)
            path_stat = os.lstat(str(lock_path))
            if (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
                raise RuntimeError(f"lock inode changed while held for {lock_path}")
        state[1] += 1
        try:
            yield
        finally:
            state[1] -= 1
            if state[1] <= 0:
                state[1] = 0
                fd = state[0]
                state[0] = None
                if isinstance(fd, int):
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)


def parse_expires_in(value: str | int | float | None, default_sec: float = 900.0) -> float:
    """Parse an expiration duration such as 15m, 60s, 1h, or numeric seconds (capped at MAX_GRANT_TTL_SEC)."""
    if value is None:
        if not math.isfinite(default_sec) or default_sec <= 0 or default_sec > MAX_GRANT_TTL_SEC:
            raise ValueError("default_sec must be a finite positive duration within MAX_GRANT_TTL_SEC")
        return float(default_sec)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        sec = float(value)
        if not math.isfinite(sec) or sec <= 0:
            raise ValueError("expires_in must be a finite positive number")
        if sec > MAX_GRANT_TTL_SEC:
            raise ValueError(f"expires_in cannot exceed {int(MAX_GRANT_TTL_SEC)}s (1h)")
        return sec
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expires_in must be a non-empty duration string or positive number")
    token = value.strip()
    idx = 0
    seen_dot = False
    while idx < len(token):
        ch = token[idx]
        if ch.isdigit():
            idx += 1
        elif ch == "." and not seen_dot:
            seen_dot = True
            idx += 1
        else:
            break
    num_part = token[:idx]
    unit = token[idx:].strip().lower()
    if not num_part or num_part == "." or unit not in _DURATION_MULTIPLIERS:
        raise ValueError(f"invalid expires_in format: {value!r} (expected e.g. 15m, 60s, 1h)")
    sec = float(num_part) * _DURATION_MULTIPLIERS[unit]
    if not math.isfinite(sec) or sec <= 0:
        raise ValueError("expires_in must be a finite positive number")
    if sec > MAX_GRANT_TTL_SEC:
        raise ValueError(f"expires_in cannot exceed {int(MAX_GRANT_TTL_SEC)}s (1h)")
    return sec


def _iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scan_and_extract_grant_tokens(action_regex: str) -> list[str]:
    """Deterministically scan action_regex character-by-character, rejecting wildcards/classes/quantifiers and extracting literal tokens."""
    if not isinstance(action_regex, str) or not action_regex.strip():
        raise ValueError("action_regex must be a non-empty regex string")
    if _SHELL_CHAIN_RE.search(action_regex) is not None:
        raise ValueError("action_regex is overly broad: cannot contain shell chaining, redirection, or subshell operators")
    buf: list[str] = []
    n = len(action_regex)
    i = 0
    while i < n:
        ch = action_regex[i]
        if ch == "\\":
            if i + 1 >= n:
                raise ValueError("action_regex is overly broad: trailing backslash is not permitted")
            nxt = action_regex[i + 1]
            if nxt == "s":
                if i + 2 < n and action_regex[i + 2] == "+":
                    i += 3
                elif i + 2 < n and action_regex[i + 2] in "*?{":
                    raise ValueError("action_regex is overly broad: only '\\s' or '\\s+' whitespace quantifiers are permitted")
                else:
                    i += 2
                buf.append(" ")
                continue
            if nxt in _ALLOWED_ESCAPED_PUNCT:
                if i + 2 < n and action_regex[i + 2] in "*+?{":
                    raise ValueError("action_regex is overly broad: quantifiers are not permitted on escaped literals")
                buf.append(nxt)
                i += 2
                continue
            raise ValueError(
                f"action_regex is overly broad: escape sequence '\\{nxt}' is not permitted; specify concrete literal characters"
            )
        if ch == "^":
            if i != 0:
                raise ValueError("action_regex is overly broad: unescaped '^' is only permitted at the start of action_regex")
            buf.append(" ")
            i += 1
            continue
        if ch == "$":
            if i != n - 1:
                raise ValueError("action_regex is overly broad: unescaped '$' is only permitted at the end of action_regex")
            buf.append(" ")
            i += 1
            continue
        if ch in _FORBIDDEN_UNESCAPED_META:
            raise ValueError(
                f"action_regex is overly broad: unescaped metacharacter {ch!r} is not permitted; specify the exact approved command or target"
            )
        buf.append(ch)
        i += 1
    cleaned = "".join(buf)
    return [tok for tok in cleaned.split() if len(tok) >= 2]


def extract_grant_action_tokens(action_regex: str) -> list[str]:
    """Extract concrete literal tokens from action_regex using the character-by-character scanner."""
    return _scan_and_extract_grant_tokens(action_regex)


def validate_grant_action_regex(action_regex: str) -> list[str]:
    """Ensure action_regex is a specific, single-action literal pattern without wildcards, chaining, or alternation."""
    tokens = _scan_and_extract_grant_tokens(action_regex)
    literal_chars = sum(len(t) for t in tokens)
    if literal_chars < 6 or not any(len(t) >= 4 for t in tokens):
        raise ValueError(
            "action_regex is overly broad; it must contain at least 6 literal characters and a specific token of >= 4 chars"
        )
    compiled = compile_contract(action_regex)
    with contract_budget():
        if compiled.search("") is not None or compiled.search("\n") is not None or compiled.search(_BENIGN_PROBE_TEXT) is not None:
            raise ValueError("action_regex is overly broad (matches empty or arbitrary benign input)")
    return tokens


def _compile_clause_action_regex(action_regex: str) -> re.Pattern[str]:
    pat = action_regex.strip()
    if pat.startswith("^"):
        pat = pat[1:]
    if pat.endswith("$") and not pat.endswith(r"\$"):
        pat = pat[:-1]
    return compile_contract(
        rf"(?<![\w./:=@+~%-])(?:{pat})(?![\w/:=@+~%-]|\.[\w./:=@+~%-])"
    )


def _clause_approves_tokens(
    text: str,
    tokens: list[str],
    clause_action_re: re.Pattern[str],
    approval_re: re.Pattern[str],
) -> bool:
    """Return True iff text has zero negations/questions and at least one clause contains both approval_re and the boundary-anchored action_regex."""
    if not tokens:
        return False
    if _NON_NEWLINE_LINEBREAK_RE.search(text) is not None:
        return False
    if _CANONICAL_NEGATION_RE.search(text) is not None:
        return False
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c and c.strip()]
    if not clauses:
        return False
    approved_any = False
    for clause in clauses:
        if _CANONICAL_NEGATION_RE.search(clause) is not None:
            return False
        m_cue = approval_re.search(clause)
        if m_cue is None:
            if _STANDALONE_AFFIRMATIVE_CLAUSE_RE.match(clause) is not None:
                continue
            return False
        with contract_budget():
            m_act = clause_action_re.search(clause)
        if m_act is None:
            if _STANDALONE_AFFIRMATIVE_CLAUSE_RE.match(clause) is not None:
                continue
            return False
        act_lower = m_act.group(0).lower()
        if not all(tok.lower() in act_lower for tok in tokens):
            return False
        if approval_re is _CANONICAL_APPROVAL_RE:
            if not all(ord(ch) < 128 for ch in clause):
                return False
            prefix = clause[: m_act.start()]
            suffix = clause[m_act.end() :]
            active_ok = (
                _ENG_APPROVAL_PREFIX_RE.match(prefix) is not None
                and _ENG_APPROVAL_SUFFIX_RE.match(suffix) is not None
            )
            passive_ok = (
                _ENG_PASSIVE_PREFIX_RE.match(prefix) is not None
                and _ENG_PASSIVE_SUFFIX_RE.match(suffix) is not None
            )
            if not (active_ok or passive_ok):
                return False
        else:
            rem_outside_act = clause[: m_act.start()] + " " + clause[m_act.end() :]
            if re.search(r"[A-Za-z0-9_./:=@+~%-]", rem_outside_act) is not None:
                return False
            spans = sorted([m_act.span(), m_cue.span()])
            connector = (
                clause[: spans[0][0]]
                + " "
                + clause[spans[0][1] : spans[1][0]]
                + " "
                + clause[spans[1][1] :]
            )
            if _NON_ASCII_ALLOWED_CONNECTOR_RE.match(connector) is None:
                return False
        approved_any = True
    return approved_any


def verify_raw_anchor_approves_grant(
    *,
    anchor_text: str,
    action_regex: str,
    rule: ContractRule,
) -> tuple[bool, str]:
    """Verify that the raw indexed user anchor text ITSELF approves action_regex in a single non-negated clause with zero extra shell tokens."""
    tokens = validate_grant_action_tokens_and_rule(action_regex, rule)
    if _CANONICAL_NEGATION_RE.search(anchor_text or "") is not None:
        return False, "user anchor message contains a prohibition, negation, conditional, or question and cannot authorize a grant"
    raw = _strip_chunk_header(anchor_text)
    if not raw:
        return False, "user anchor message is empty"
    clause_action_re = _compile_clause_action_regex(action_regex)
    with contract_budget():
        m_raw_act = clause_action_re.search(raw)
    if m_raw_act is None:
        return (
            False,
            f"user message does not contain the exact granted action pattern ({tokens!r}); "
            "a user turn can only authorize an action that the user explicitly named in that turn",
        )
    if _CANONICAL_NEGATION_RE.search(raw) is not None:
        return False, "user anchor message contains a prohibition, negation, conditional, or question and cannot authorize a grant"
    is_raw_ascii = all(ord(ch) < 128 for ch in raw)
    if is_raw_ascii:
        if not _clause_approves_tokens(raw, tokens, clause_action_re, _CANONICAL_APPROVAL_RE):
            return (
                False,
                "user message does not contain an explicit Canonical English approval phrase in the same clause as the action",
            )
        return True, "anchor_approves_action_in_canonical_english"
    raw_shell_set = set(_extract_shell_tokens(raw))
    raw_act_shell_set = set(_extract_shell_tokens(m_raw_act.group(0)))
    if raw_shell_set != raw_act_shell_set:
        return (
            False,
            "multilingual user message contains extra command or shell tokens outside the granted action_regex",
        )
    if not _clause_approves_tokens(raw, tokens, clause_action_re, _NON_ASCII_APPROVAL_RE):
        return (
            False,
            "multilingual user message does not contain an explicit approval cue in the same clause as the action",
        )
    return True, "raw_anchor_approves_action"


def verify_user_anchor_approves_grant(
    *,
    anchor_text: str,
    action_regex: str,
    rule: ContractRule,
    canonical_english_quote: str | None = None,
) -> tuple[bool, str]:
    """Verify conjunctively (AND) within a single non-negated clause that the user anchor message references action_regex and approves it."""
    raw_ok, raw_reason = verify_raw_anchor_approves_grant(
        anchor_text=anchor_text,
        action_regex=action_regex,
        rule=rule,
    )
    if not raw_ok:
        return False, raw_reason
    tokens = validate_grant_action_tokens_and_rule(action_regex, rule)
    raw = _strip_chunk_header(anchor_text)
    clause_action_re = _compile_clause_action_regex(action_regex)
    eng = canonical_english_quote.strip() if isinstance(canonical_english_quote, str) and canonical_english_quote.strip() else ""
    is_raw_ascii = all(ord(ch) < 128 for ch in raw)
    if is_raw_ascii:
        if eng and eng != raw:
            return (
                False,
                "canonical_english_quote cannot override or differ from an already-ASCII English user message",
            )
        return True, "anchor_approves_action_in_canonical_english"

    if not eng:
        return (
            False,
            "user message does not contain a Canonical English approval phrase; "
            "when the user speaks in another language, pass canonical_english_quote translated by the orchestration LLM",
        )
    if not all(ord(ch) < 128 for ch in eng):
        return False, "canonical_english_quote must be ASCII Canonical English"
    raw_shell_set = set(_extract_shell_tokens(raw))
    raw_tech_set = set(_extract_tech_tokens(raw))
    eng_tech_tokens = _extract_tech_tokens(eng)
    eng_tech_set = set(eng_tech_tokens)
    if any(tok not in raw_shell_set for tok in eng_tech_tokens) or not raw_tech_set.issubset(eng_tech_set):
        return (
            False,
            "canonical_english_quote contains technical identifiers/flags that do not match the exact shell tokens in the user anchor message",
        )
    with contract_budget():
        m_eng_act = clause_action_re.search(eng)
    if m_eng_act is None:
        return (
            False,
            f"canonical_english_quote does not contain the exact granted action pattern ({tokens!r})",
        )
    if set(_extract_shell_tokens(m_eng_act.group(0))) != raw_shell_set:
        return (
            False,
            "canonical_english_quote action span does not preserve all shell tokens from the user anchor message",
        )
    if _CANONICAL_NEGATION_RE.search(eng) is not None:
        return False, "canonical English quote contains a prohibition, negation, conditional, or question and cannot authorize a grant"
    if not _clause_approves_tokens(eng, tokens, clause_action_re, _CANONICAL_APPROVAL_RE):
        return (
            False,
            "canonical_english_quote does not contain an explicit Canonical English approval phrase in the same clause as the action",
        )
    return True, "anchor_approves_action_in_canonical_english"


def validate_grant_action_tokens_and_rule(action_regex: str, rule: ContractRule) -> list[str]:
    """Validate action_regex specificity and ensure its concrete command tokens correspond to the forbidden rule."""
    tokens = validate_grant_action_regex(action_regex)
    synthetic = " ".join(tokens)
    synthetic_tool_call = f"[tool_call:Bash]\ncommand: {synthetic}"
    if rule.regex:
        rule_re = compile_contract(rule.regex)
        exc_re = compile_contract(rule.exception_regex) if rule.exception_regex else None
        with contract_budget():
            if rule_re.search(synthetic) is None and rule_re.search(synthetic_tool_call) is None:
                raise ValueError(
                    f"action_regex {action_regex!r} does not match contract {rule.id!r} pattern; "
                    "a grant must name the specific forbidden action covered by the contract"
                )
            if exc_re is not None and (
                exc_re.search(synthetic) is not None or exc_re.search(synthetic_tool_call) is not None
            ):
                raise ValueError(
                    f"action_regex {action_regex!r} is already exempted by contract {rule.id!r} exception_regex"
                )
    return tokens


class GrantReceipt(BaseModel):
    grant_id: str
    contract_id: str
    source_event_id: str
    event_id: str | None = None
    unlinked: bool = False
    used_at: str
    used_at_epoch: float
    matched_text: str | None = None
    matched_text_digest: str | None = None


class GrantRecord(BaseModel):
    grant_id: str
    session_id: str
    contract_id: str
    contract_revision: int = 1
    action_regex: str
    source_event_id: str
    issued_after_event_id: str | None = None
    source_quote: str | None = None
    canonical_english_quote: str | None = None
    created_at: str
    created_at_epoch: float
    expires_at: str
    expires_at_epoch: float
    max_uses: int = 1
    uses_consumed: int = 0
    consumed_event_ids: list[str] = Field(default_factory=list)
    unlinked_check_texts: list[str] = Field(default_factory=list)
    receipts: list[GrantReceipt] = Field(default_factory=list)

    @property
    def uses_remaining(self) -> int:
        return max(0, self.max_uses - self.uses_consumed)

    def is_expired(self, now: float | None = None) -> bool:
        if isinstance(now, bool):
            return True
        now_ts = time.time() if now is None else float(now)
        if not (
            math.isfinite(now_ts)
            and math.isfinite(self.expires_at_epoch)
            and math.isfinite(self.created_at_epoch)
        ):
            return True
        return now_ts >= self.expires_at_epoch

    def is_active(self, now: float | None = None) -> bool:
        if isinstance(now, bool):
            return False
        now_ts = time.time() if now is None else float(now)
        return (
            math.isfinite(now_ts)
            and math.isfinite(self.created_at_epoch)
            and math.isfinite(self.expires_at_epoch)
            and (now_ts >= self.created_at_epoch)
            and (not self.is_expired(now_ts))
            and self.max_uses == 1
            and self.uses_consumed < 1
            and len(self.consumed_event_ids) == 0
            and len(self.unlinked_check_texts) == 0
        )


class GrantsDocument(BaseModel):
    schema_version: str = "1.0"
    grants: list[GrantRecord] = Field(default_factory=list)


def _load_grants_doc_unlocked(path: Path) -> GrantsDocument:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return GrantsDocument()
    except OSError as exc:
        raise ValueError(f"failed to read grants store at {path}: {exc}") from exc
    try:
        raw = json.loads(raw_text)
        return GrantsDocument.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"corrupted grants store at {path}: {exc}") from exc


def _save_grants_doc_unlocked(path: Path, doc: GrantsDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _merge_grant_records(disk_grants: list[GrantRecord], mem_grants: list[GrantRecord]) -> list[GrantRecord]:
    """Merge disk and in-memory grant records without dropping in-memory records or regressing consumption state."""
    by_id: dict[str, GrantRecord] = {}
    order: list[str] = []
    for g in disk_grants:
        by_id[g.grant_id] = g.model_copy(deep=True)
        order.append(g.grant_id)
    for g in mem_grants:
        if g.grant_id not in by_id:
            by_id[g.grant_id] = g.model_copy(deep=True)
            order.append(g.grant_id)
        else:
            cur = by_id[g.grant_id]
            cur_score = (cur.uses_consumed, len(cur.consumed_event_ids), len(cur.unlinked_check_texts), len(cur.receipts))
            new_score = (g.uses_consumed, len(g.consumed_event_ids), len(g.unlinked_check_texts), len(g.receipts))
            if new_score > cur_score:
                by_id[g.grant_id] = g.model_copy(deep=True)
    return [by_id[gid] for gid in order]


def load_grants_doc(path: Path) -> GrantsDocument:
    with grants_lock(path):
        return _load_grants_doc_unlocked(path)


def save_grants_doc(path: Path, doc: GrantsDocument) -> None:
    with grants_lock(path):
        _save_grants_doc_unlocked(path, doc)


def create_grant_record(
    *,
    session_id: str,
    rule: ContractRule,
    action_regex: str,
    source_event_id: str,
    issued_after_event_id: str | None = None,
    source_quote: str | None,
    canonical_english_quote: str | None = None,
    max_uses: int = 1,
    expires_in: str | int | float = "15m",
    now: float | None = None,
    chunks: list[AuditEvent] | None = None,
    existing_grants: list[GrantRecord] | GrantsDocument | None = None,
    grants_path: Path | None = None,
    max_anchor_events: int = 20,
    max_anchor_user_turns: int = 2,
) -> GrantRecord:
    if grants_path is not None:
        with grants_lock(grants_path):
            disk_doc = _load_grants_doc_unlocked(grants_path)
            mem_list: list[GrantRecord] = []
            if existing_grants is not None:
                mem_list = list(existing_grants.grants if isinstance(existing_grants, GrantsDocument) else existing_grants)
            merged_existing = _merge_grant_records(disk_doc.grants, mem_list)
            rec = create_grant_record(
                session_id=session_id,
                rule=rule,
                action_regex=action_regex,
                source_event_id=source_event_id,
                issued_after_event_id=issued_after_event_id,
                source_quote=source_quote,
                canonical_english_quote=canonical_english_quote,
                max_uses=max_uses,
                expires_in=expires_in,
                now=now,
                chunks=chunks,
                existing_grants=merged_existing,
                grants_path=None,
                max_anchor_events=max_anchor_events,
                max_anchor_user_turns=max_anchor_user_turns,
            )
            merged_existing.append(rec)
            disk_doc.grants = merged_existing
            _save_grants_doc_unlocked(grants_path, disk_doc)
            if isinstance(existing_grants, GrantsDocument):
                existing_grants.grants = list(merged_existing)
            elif isinstance(existing_grants, list):
                existing_grants[:] = list(merged_existing)
            return rec
    if not isinstance(source_event_id, str) or not source_event_id.strip():
        raise ValueError("grant_once requires a non-empty source_event_id")
    if rule.type != "must_not":
        raise ValueError(f"grant_once only applies to 'must_not' contracts (contract {rule.id!r} is {rule.type!r})")
    if getattr(rule, "path_guard", None) is not None or not rule.regex:
        raise ValueError(f"Contract {rule.id!r} uses path_guard or lacks a regex and does not support grant_once")
    if isinstance(max_uses, bool) or not isinstance(max_uses, int) or max_uses != 1:
        raise ValueError("grant_once only supports max_uses=1")
    if not chunks:
        raise ValueError("anchor_unresolved: chunks is required to verify source_event_id anchor")
    validate_grant_action_tokens_and_rule(action_regex, rule)
    if existing_grants is not None:
        grant_list = existing_grants.grants if isinstance(existing_grants, GrantsDocument) else existing_grants
        if any(
            g.session_id == session_id
            and g.contract_id == rule.id
            and g.source_event_id == source_event_id
            for g in grant_list
        ):
            raise ValueError(
                f"anchor_already_used: source_event_id {source_event_id!r} has already been used to authorize a grant for contract {rule.id!r}"
            )
    chunk_positions: dict[str, int] = {}
    for idx, c in enumerate(chunks):
        if c.chunk_id in chunk_positions:
            raise ValueError(f"duplicate chunk_id {c.chunk_id!r} in session chunks")
        chunk_positions[c.chunk_id] = idx
    anchor_idx = chunk_positions.get(source_event_id, -1)
    if anchor_idx < 0:
        raise ValueError(f"anchor_unresolved: source_event_id {source_event_id!r} not found in session chunks")
    anchor = chunks[anchor_idx]
    anchor_meta = anchor.meta if isinstance(anchor.meta, dict) else {}
    anchor_sid = getattr(anchor, "session_id", None) or anchor_meta.get("session_id")
    if not anchor_sid or anchor_sid != session_id:
        raise ValueError(f"anchor_unresolved: source_event_id {source_event_id!r} belongs to a different or unverified session")
    if anchor.kind != "turn" or anchor_meta.get("role") != "user":
        raise ValueError(f"anchor_not_user: source_event_id {source_event_id!r} is not a user turn")
    if rule.applies_from_event_id:
        if rule.applies_from_event_id not in chunk_positions or anchor_idx < chunk_positions[rule.applies_from_event_id]:
            raise ValueError("anchor_predates_contract: source_event_id predates contract activation")
    user_turn_ids = [
        c.chunk_id for c in chunks
        if c.kind == "turn" and isinstance(c.meta, dict) and c.meta.get("role") == "user"
    ]
    if (
        max_anchor_events <= 0
        or max_anchor_user_turns <= 0
        or (len(chunks) - anchor_idx) > max_anchor_events
        or not user_turn_ids
        or source_event_id not in user_turn_ids[-max_anchor_user_turns:]
    ):
        raise ValueError("anchor_stale: source_event_id is outside the recent user-turn/event window")
    if source_quote is not None and source_quote != anchor.text:
        raise ValueError("source_quote does not match the indexed user anchor text")
    eff_quote = anchor.text
    eff_canon = canonical_english_quote
    stored_canon = anchor_meta.get("canonical_english")
    if isinstance(stored_canon, str) and stored_canon.strip():
        if eff_canon is not None and eff_canon.strip() and eff_canon.strip() != stored_canon.strip():
            raise ValueError("canonical_english_quote does not match the indexed canonical_english metadata")
        eff_canon = stored_canon.strip()
    tail_event_id = chunks[-1].chunk_id
    if issued_after_event_id is not None and issued_after_event_id != tail_event_id:
        raise ValueError(
            f"issued_after_event_id ({issued_after_event_id!r}) must equal the current session tail event ({tail_event_id!r})"
        )
    issued_after_event_id = tail_event_id
    approved, approval_detail = verify_user_anchor_approves_grant(
        anchor_text=eff_quote,
        action_regex=action_regex,
        rule=rule,
        canonical_english_quote=eff_canon,
    )
    if not approved:
        raise ValueError(approval_detail)
    ttl_sec = parse_expires_in(expires_in)
    sys_now = time.time()
    if now is not None and (isinstance(now, bool) or not math.isfinite(float(now))):
        raise ValueError("now must be a finite timestamp")
    now_ts = sys_now if now is None else min(max(float(now), 0.0), sys_now)
    exp_ts = now_ts + ttl_sec
    return GrantRecord(
        grant_id=f"grant_{uuid.uuid4().hex[:12]}",
        session_id=session_id,
        contract_id=rule.id,
        contract_revision=rule.revision,
        action_regex=action_regex,
        source_event_id=source_event_id,
        issued_after_event_id=issued_after_event_id,
        source_quote=mask_secrets(eff_quote)[:200] if eff_quote else None,
        canonical_english_quote=(
            mask_secrets(eff_canon.strip())[:200]
            if isinstance(eff_canon, str) and eff_canon.strip()
            else None
        ),
        created_at=_iso_utc(now_ts),
        created_at_epoch=now_ts,
        expires_at=_iso_utc(exp_ts),
        expires_at_epoch=exp_ts,
        max_uses=1,
    )


def session_grant_contract_ids(doc: GrantsDocument, session_id: str) -> set[str]:
    """Return contract IDs that have any grant record in session_id."""
    return {g.contract_id for g in doc.grants if g.session_id == session_id}


def _strip_tool_header(text: str) -> str:
    """Strip at most one leading canonical tool_call header line so the payload body can be full-matched."""
    return _LEADING_TOOL_HEADER_RE.sub("", text, count=1)


_CMD_JSON_KEYS = ("command", "CommandLine", "cmd")
_ALLOWED_CMD_JSON_META_KEYS = frozenset({
    "description", "timeout", "WaitMsBeforeAsync",
    "toolSummary", "justification",
})


def _load_unique_json_dict(raw_json: str) -> dict[str, Any] | None:
    """Parse a JSON object while rejecting any duplicate keys or non-finite numbers at any nesting level."""
    dup = False

    def _pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal dup
        out: dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                dup = True
            if isinstance(v, float) and not math.isfinite(v):
                dup = True
            out[k] = v
        return out

    def _reject_constant(_c: str) -> Any:
        raise ValueError("non-finite JSON constant")

    try:
        parsed = json.loads(raw_json, object_pairs_hook=_pairs_hook, parse_constant=_reject_constant)
    except Exception:
        return None
    if dup or not isinstance(parsed, dict):
        return None
    return parsed


def _has_control_or_format_chars(s: str) -> bool:
    if unicodedata.normalize("NFKC", s) != s:
        return True
    for ch in s:
        code = ord(ch)
        if code < 32 or code == 127:
            return True
        if ch != " " and (ch.isspace() or unicodedata.category(ch).startswith(("C", "M", "Z"))):
            return True
    return False


def _extract_command_body(text: str) -> tuple[str, str | None]:
    """Classify and extract a single command line body from canonical tool_call text.

    Returns `(kind, cmd)` where `kind` is one of:
    - `"invalid"`: malformed input, control/non-ASCII-space chars, multi-line body, malformed `command:`,
      or any JSON-like/bracketed payload that fails strict command schema validation.
    - `"command_colon"`: single-line `command: <cmd>` payload.
    - `"json_str"`: single-line JSON object whose single command key holds a clean non-padded string.
    - `"json_argv"`: single-line JSON object whose single command key holds a non-empty list of whitespace-free tokens.
    - `"not_command"`: single-line non-command, non-JSON plain text body.
    """
    if not isinstance(text, str) or _NON_NEWLINE_LINEBREAK_RE.search(text) is not None:
        return "invalid", None
    raw_body = _strip_tool_header(text)
    if not raw_body or "\n" in raw_body:
        return "invalid", None
    if _has_control_or_format_chars(raw_body):
        return "invalid", None
    if raw_body != raw_body.strip(" "):
        return "invalid", None
    body = raw_body
    if body.lower().startswith("command:"):
        if not body.startswith("command:"):
            return "invalid", None
        after_colon = body[len("command:") :]
        cmd = after_colon[1:] if after_colon.startswith(" ") else after_colon
        if not cmd or cmd != cmd.strip(" ") or _SHELL_CHAIN_RE.search(cmd) is not None:
            return "invalid", None
        return "command_colon", cmd
    if body.startswith("{") or "{" in body or "}" in body or "[" in body or "]" in body:
        if not (body.startswith("{") and body.endswith("}")):
            return "invalid", None
        payload = _load_unique_json_dict(body)
        if payload is None:
            return "invalid", None
        if any(k not in _CMD_JSON_KEYS and k not in _ALLOWED_CMD_JSON_META_KEYS for k in payload):
            return "invalid", None
        present_cmd_keys = [k for k in _CMD_JSON_KEYS if k in payload]
        if len(present_cmd_keys) != 1:
            return "invalid", None
        for k, v in payload.items():
            if k not in _CMD_JSON_KEYS:
                if v is not None and not isinstance(v, (str, int, float, bool)):
                    return "invalid", None
                if isinstance(v, float) and not math.isfinite(v):
                    return "invalid", None
                if isinstance(v, str) and (
                    _NON_NEWLINE_LINEBREAK_RE.search(v) is not None
                    or "\n" in v
                    or _has_control_or_format_chars(v)
                    or _SHELL_CHAIN_RE.search(v) is not None
                ):
                    return "invalid", None
        val = payload[present_cmd_keys[0]]
        if isinstance(val, str):
            if (
                not val
                or val != val.strip(" ")
                or _NON_NEWLINE_LINEBREAK_RE.search(val) is not None
                or "\n" in val
                or _has_control_or_format_chars(val)
                or _SHELL_CHAIN_RE.search(val) is not None
            ):
                return "invalid", None
            return "json_str", val
        if isinstance(val, list) and val and all(isinstance(x, str) for x in val):
            if any(
                not x
                or any(ch.isspace() for ch in x)
                or _has_control_or_format_chars(x)
                or _NON_NEWLINE_LINEBREAK_RE.search(x) is not None
                or _SHELL_CHAIN_RE.search(x) is not None
                for x in val
            ):
                return "invalid", None
            cmd = " ".join(val)
            if not cmd or _SHELL_CHAIN_RE.search(cmd) is not None:
                return "invalid", None
            return "json_argv", cmd
        return "invalid", None
    return "not_command", None


def _grant_match_digest(grant: GrantRecord, rule: ContractRule, text: str) -> tuple[bool, str | None, str | None]:
    """Return (matched, canonical_action_digest, matched_text_preview) under contract_budget.

    Prevents command-chaining, wrapper commands, extra arguments, and multi-violation piggybacking:
    1) Rejects non-newline linebreaks (\r, \v, \f, \u2028, etc.), control/non-ASCII-space characters,
       and shell chaining (&&, ||, ;, |, backticks, $(), newlines).
    2) Treats any malformed JSON/command structure as terminal ("invalid") without falling through to raw body matching.
    3) Requires `grant.action_regex` to match the ENTIRE extracted command (`m.span() == (0, len(cmd_body))`)
       or the ENTIRE single-line non-command payload body (`m_grant.span() == (0, len(target_body))`).
    4) Requires `rule.regex` to match the target body and verifies that non-command fields/headers do not match `rule.regex`.
    """
    if not isinstance(text, str) or _NON_NEWLINE_LINEBREAK_RE.search(text) is not None:
        return False, None, None
    try:
        validate_grant_action_tokens_and_rule(grant.action_regex, rule)
        compiled = compile_contract(grant.action_regex)
        rule_regex = compile_contract(rule.regex) if rule.regex else None
    except (ValueError, re.error):
        return False, None, None

    raw_body = _strip_tool_header(text)
    if not raw_body or "\n" in raw_body or raw_body != raw_body.strip(" "):
        return False, None, None
    if _has_control_or_format_chars(raw_body):
        return False, None, None
    target_body = raw_body
    header_part = text[: len(text) - len(raw_body)]

    cmd_kind, cmd_body = _extract_command_body(text)
    if cmd_kind == "invalid":
        return False, None, None
    if cmd_kind in ("command_colon", "json_str", "json_argv") and cmd_body is not None:
        if _SHELL_CHAIN_RE.search(cmd_body) is not None:
            return False, None, None
        m = compiled.search(cmd_body)
        if m is None or m.span() != (0, len(cmd_body)):
            return False, None, None
        if rule_regex is not None:
            if rule_regex.search(cmd_body) is None:
                return False, None, None
            if header_part and rule_regex.search(header_part) is not None:
                return False, None, None
            if cmd_kind in ("json_str", "json_argv"):
                payload_obj = _load_unique_json_dict(target_body)
                if payload_obj is None:
                    return False, None, None
                for k, v in payload_obj.items():
                    if rule_regex.search(str(k)) is not None:
                        return False, None, None
                    if k not in _CMD_JSON_KEYS and v is not None and rule_regex.search(str(v)) is not None:
                        return False, None, None
                without_cmd_obj = {k: v for k, v in payload_obj.items() if k not in _CMD_JSON_KEYS}
                if rule_regex.search(json.dumps(without_cmd_obj, ensure_ascii=False)) is not None:
                    return False, None, None
            else:
                if not target_body.startswith("command:") or rule_regex.search("command:") is not None:
                    return False, None, None
        preview = mask_secrets(cmd_body)[:120]
        return True, _text_digest(f"cmd:{cmd_kind}:{cmd_body}|raw:{target_body}"), preview

    if _SHELL_CHAIN_RE.search(target_body) is not None:
        return False, None, None
    m_grant = compiled.search(target_body)
    if m_grant is None or m_grant.span() != (0, len(target_body)):
        return False, None, None
    if rule_regex is not None:
        if rule_regex.search(target_body) is None:
            return False, None, None
        if header_part and rule_regex.search(header_part) is not None:
            return False, None, None
    preview = mask_secrets(target_body)[:120]
    return True, _text_digest(f"body:{target_body}"), preview


def _is_valid_consumed_grant(g: GrantRecord) -> bool:
    return (
        g.max_uses == 1
        and len(g.consumed_event_ids) == 1
        and len(g.unlinked_check_texts) == 0
        and len(g.receipts) == 1
        and 1 <= g.uses_consumed <= g.max_uses
    )


def _is_valid_unlinked_grant(g: GrantRecord) -> bool:
    return (
        g.max_uses == 1
        and len(g.consumed_event_ids) == 0
        and len(g.unlinked_check_texts) == 1
        and len(g.receipts) == 1
        and 1 <= g.uses_consumed <= g.max_uses
    )


def _is_valid_fresh_grant(g: GrantRecord, now_ts: float) -> bool:
    return (
        g.max_uses == 1
        and g.is_active(now_ts)
        and g.uses_remaining > 0
        and g.uses_consumed == 0
        and len(g.consumed_event_ids) == 0
        and len(g.unlinked_check_texts) == 0
        and len(g.receipts) == 0
    )


_ALLOWED_GRANT_PROSE_WORDS = frozenset({
    "please", "yes", "you", "may", "can", "have", "my", "explicit", "permission",
    "is", "are", "granted", "to", "hereby", "allow", "allowed", "approve", "approved",
    "authorize", "authorized", "permit", "permitted", "grant", "go", "ahead", "and",
    "ok", "okay", "feel", "free", "now", "once", "one", "time", "right", "explicitly",
    "run", "running", "execute", "executing", "perform", "performing", "invoke", "invoking",
    "call", "calling", "use", "using", "do", "proceed", "with", "in", "for", "this", "session",
})


def apply_grants_to_finding(
    *,
    rule: ContractRule,
    finding: Finding,
    doc: GrantsDocument,
    session_id: str,
    chunks: list[AuditEvent],
    proposed: AuditEvent | None = None,
    rejected: dict[str, Any] | None = None,
    now: float | None = None,
    grants_path: Path | None = None,
) -> tuple[Finding, bool]:
    """Apply session grants for rule to a must_not finding.

    Stages all mutations on deep copies inside `contract_budget()` and commits back to `doc.grants`
    only on normal completion so a regex timeout is strictly atomic (zero partial mutations).
    When `grants_path` is provided, acquires `grants_lock(grants_path)` and persists `doc` on disk
    atomically whenever a grant slot is consumed or linked.
    """
    if grants_path is not None:
        with grants_lock(grants_path):
            disk_doc = _load_grants_doc_unlocked(grants_path)
            merged_grants = _merge_grant_records(disk_doc.grants, doc.grants)
            disk_doc.grants = merged_grants
            doc.grants = [g.model_copy(deep=True) for g in merged_grants]
            updated_finding, dirty = apply_grants_to_finding(
                rule=rule,
                finding=finding,
                doc=disk_doc,
                session_id=session_id,
                chunks=chunks,
                proposed=proposed,
                rejected=rejected,
                now=now,
                grants_path=None,
            )
            doc.grants = [g.model_copy(deep=True) for g in disk_doc.grants]
            if dirty:
                _save_grants_doc_unlocked(grants_path, disk_doc)
            return updated_finding, dirty

    with _PROCESS_GRANTS_LOCK:
        if finding.contract_id != rule.id or finding.contract_revision != rule.revision:
            return finding, False
        if rule.type != "must_not" or finding.verdict not in ("violation", "suspicious"):
            return finding, False
        if getattr(rule, "path_guard", None) is not None or not rule.regex:
            return finding, False
        if rejected or finding.proposals_rejected:
            return finding, False

        raw_ev_list = list(finding.evidence or [])
        cited_ids: list[str] = []
        for ev in raw_ev_list:
            cid = ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None)
            if not isinstance(cid, str) or not cid:
                return finding, False
            cited_ids.append(cid)

        ids: dict[str, int] = {}
        chunks_by_id: dict[str, AuditEvent] = {}
        for i, c in enumerate(chunks):
            if c.chunk_id in ids:
                return finding, False
            ids[c.chunk_id] = i
            chunks_by_id[c.chunk_id] = c
        anchor_counts: dict[tuple[str, str, str], int] = {}
        id_counts: dict[str, int] = {}
        for g in doc.grants:
            if g.session_id == session_id and g.contract_id == rule.id:
                akey = (g.session_id, g.contract_id, g.source_event_id)
                anchor_counts[akey] = anchor_counts.get(akey, 0) + 1
            if g.grant_id:
                id_counts[g.grant_id] = id_counts.get(g.grant_id, 0) + 1

        sys_now = time.time()
        if now is not None:
            if isinstance(now, bool) or not isinstance(now, (int, float)):
                return finding, False
            now_val = float(now)
            if not math.isfinite(now_val) or now_val <= 0.0 or abs(now_val - sys_now) > 5.0:
                return finding, False
        now_ts = sys_now
        dirty = False

        try:
            with contract_budget():
                def _charge_budget(step_t0: float) -> float:
                    now_p = time.perf_counter()
                    rem = _remaining.get()
                    budget_val = SEARCH_TIMEOUT_SEC if rem is None else rem
                    new_rem = budget_val - (now_p - step_t0)
                    _remaining.set(new_rem)
                    if new_rem <= 0.0:
                        raise ContractRegexTimeout(rule.regex or rule.id)
                    return now_p

                def _timed_regex_search(compiled_pat: Any, text_val: str) -> Any:
                    t_s0 = _charge_budget(time.perf_counter())
                    rem_now = _remaining.get()
                    limit_s = SEARCH_TIMEOUT_SEC if rem_now is None else min(SEARCH_TIMEOUT_SEC, rem_now)
                    if limit_s <= 0.0:
                        raise ContractRegexTimeout(rule.regex or rule.id)
                    raw_pat = getattr(compiled_pat, "_compiled", None)
                    try:
                        if raw_pat is not None:
                            return raw_pat.search(text_val, timeout=limit_s)
                        return compiled_pat.search(text_val)
                    except TimeoutError:
                        raise ContractRegexTimeout(rule.regex or rule.id) from None
                    finally:
                        _charge_budget(t_s0)

                def _timed_grant_match_digest(
                    g_rec: GrantRecord, r_obj: ContractRule, text_val: str
                ) -> tuple[bool, str | None, str | None]:
                    t_m0 = _charge_budget(time.perf_counter())
                    try:
                        return _grant_match_digest(g_rec, r_obj, text_val)
                    finally:
                        _charge_budget(t_m0)

                seen_anchors: set[tuple[str, str, str]] = set()
                grant_indices: list[int] = []
                for idx, g in enumerate(doc.grants):
                    cand_t0 = _charge_budget(time.perf_counter())
                    try:
                        if not (
                            g.session_id == session_id
                            and g.contract_id == rule.id
                            and g.contract_revision == rule.revision
                            and g.max_uses == 1
                        ):
                            continue
                        dedup_anchor_key = (g.session_id, g.contract_id, g.source_event_id)
                        if (
                            dedup_anchor_key in seen_anchors
                            or anchor_counts.get(dedup_anchor_key, 0) != 1
                            or not g.grant_id
                            or id_counts.get(g.grant_id, 0) != 1
                        ):
                            continue
                        if not g.issued_after_event_id or g.issued_after_event_id not in ids:
                            continue
                        anchor_chunk = chunks_by_id.get(g.source_event_id)
                        if anchor_chunk is None:
                            continue
                        anchor_meta = anchor_chunk.meta if isinstance(anchor_chunk.meta, dict) else {}
                        anchor_sid = getattr(anchor_chunk, "session_id", None) or anchor_meta.get("session_id")
                        if not anchor_sid or anchor_sid != session_id:
                            continue
                        if ids[g.issued_after_event_id] < ids[g.source_event_id]:
                            continue
                        anchor_attr_role = getattr(anchor_chunk, "role", None)
                        if (
                            anchor_chunk.kind != "turn"
                            or anchor_meta.get("role") != "user"
                            or (anchor_attr_role is not None and anchor_attr_role != "user")
                        ):
                            continue
                        if rule.applies_from_event_id:
                            if rule.applies_from_event_id not in ids or ids[g.source_event_id] < ids[rule.applies_from_event_id]:
                                continue
                        raw_canon = g.canonical_english_quote
                        stored_canon = anchor_meta.get("canonical_english")
                        eff_canon: str | None = None
                        if stored_canon is not None:
                            if not isinstance(stored_canon, str) or not stored_canon.strip() or len(stored_canon.strip()) > 200:
                                continue
                            s_clean = stored_canon.strip()
                            if s_clean != mask_secrets(s_clean):
                                continue
                            if raw_canon is not None:
                                if not isinstance(raw_canon, str) or raw_canon.strip() != s_clean:
                                    continue
                            eff_canon = s_clean
                        elif raw_canon is not None:
                            if not isinstance(raw_canon, str) or not raw_canon.strip() or len(raw_canon.strip()) > 200:
                                continue
                            c_clean = raw_canon.strip()
                            if c_clean != mask_secrets(c_clean):
                                continue
                            eff_canon = c_clean
                        raw_anchor_full = anchor_chunk.text or ""
                        if (
                            "?" in raw_anchor_full
                            or "？" in raw_anchor_full
                            or _CANONICAL_NEGATION_RE.search(raw_anchor_full) is not None
                        ):
                            continue
                        anchor_stripped = _strip_chunk_header(raw_anchor_full)
                        if (
                            not anchor_stripped
                            or "?" in anchor_stripped
                            or "？" in anchor_stripped
                            or _CANONICAL_NEGATION_RE.search(anchor_stripped) is not None
                        ):
                            continue
                        raw_shell_list = _extract_shell_tokens(anchor_stripped)
                        if not raw_shell_list:
                            continue
                        raw_shell_set = set(raw_shell_list)
                        try:
                            action_tokens = validate_grant_action_tokens_and_rule(g.action_regex, rule)
                            raw_anchor_ok, _ = verify_raw_anchor_approves_grant(
                                anchor_text=anchor_stripped,
                                action_regex=g.action_regex,
                                rule=rule,
                            )
                        except (ValueError, re.error):
                            continue
                        if not raw_anchor_ok or not action_tokens or any(tok.lower() not in raw_shell_set for tok in action_tokens):
                            continue
                        is_ascii_anchor = all(ord(ch) < 128 for ch in anchor_stripped)
                        if is_ascii_anchor:
                            if _CANONICAL_APPROVAL_RE.search(anchor_stripped) is None:
                                continue
                            if eff_canon is not None and eff_canon != anchor_stripped:
                                continue
                        else:
                            if _NON_ASCII_APPROVAL_RE.search(anchor_stripped) is None:
                                continue
                            if eff_canon is None:
                                continue
                            if (
                                not all(ord(ch) < 128 for ch in eff_canon)
                                or "?" in eff_canon
                                or "？" in eff_canon
                                or _CANONICAL_NEGATION_RE.search(eff_canon) is not None
                                or _CANONICAL_APPROVAL_RE.search(eff_canon) is None
                                or _extract_tech_tokens_exact(anchor_stripped) != _extract_tech_tokens_exact(eff_canon)
                                or _extract_tech_tokens(anchor_stripped) != _extract_tech_tokens(eff_canon)
                                or _SHELL_METACHAR_SEQ_RE.findall(anchor_stripped) != _SHELL_METACHAR_SEQ_RE.findall(eff_canon)
                            ):
                                continue
                            raw_cue_text = _strip_tech_tokens_for_cues(anchor_stripped)
                            eng_cue_text = _strip_tech_tokens_for_cues(eff_canon)
                            if (
                                _extract_verb_subcat_mask(raw_cue_text) != _extract_verb_subcat_mask(eng_cue_text)
                                or bool(_DESTRUCTIVE_CUE_RE.search(raw_cue_text)) != bool(_DESTRUCTIVE_CUE_RE.search(eng_cue_text))
                                or bool(_MUTATION_CUE_RE.search(raw_cue_text)) != bool(_MUTATION_CUE_RE.search(eng_cue_text))
                                or bool(_PRESERVATIVE_CUE_RE.search(raw_cue_text)) != bool(_PRESERVATIVE_CUE_RE.search(eng_cue_text))
                                or bool(_EXEC_USE_CUE_RE.search(raw_cue_text)) != bool(_EXEC_USE_CUE_RE.search(eng_cue_text))
                            ):
                                continue
                            canon_shell_list = _extract_shell_tokens(eff_canon)
                            if [t for t in canon_shell_list if t in raw_shell_set] != raw_shell_list:
                                continue
                            extra_canon_tokens = [t for t in canon_shell_list if t not in raw_shell_set]
                            if (
                                len(extra_canon_tokens) != len(set(extra_canon_tokens))
                                or any(not t.isalpha() or t not in _ALLOWED_GRANT_PROSE_WORDS for t in extra_canon_tokens)
                            ):
                                continue
                        try:
                            approved, _ = verify_user_anchor_approves_grant(
                                anchor_text=anchor_stripped,
                                action_regex=g.action_regex,
                                rule=rule,
                                canonical_english_quote=eff_canon,
                            )
                        except (ValueError, re.error):
                            continue
                        if not approved:
                            continue
                        seen_anchors.add(dedup_anchor_key)
                        grant_indices.append(idx)
                    finally:
                        _charge_budget(cand_t0)
                if not grant_indices:
                    return finding, False

                working_grants = [doc.grants[idx].model_copy(deep=True) for idx in grant_indices]
                compile_t0 = _charge_budget(time.perf_counter())
                try:
                    rule_regex = compile_contract(rule.regex)
                    exc_regex = compile_contract(rule.exception_regex) if rule.exception_regex else None
                except re.error:
                    return finding, False
                finally:
                    _charge_budget(compile_t0)

                if rule.applies_from_event_id and rule.applies_from_event_id not in ids:
                    return finding, False
                start = (
                    ids[rule.applies_from_event_id] + 1
                    if (rule.applies_from_event_id and rule.applies_from_event_id in ids)
                    else 0
                )
                active = chunks[start:]
                hits: list[AuditEvent] = []
                for c in active:
                    _charge_budget(time.perf_counter())
                    if not _eligible(c, rule):
                        continue
                    if _timed_regex_search(rule_regex, c.text) is None:
                        continue
                    if exc_regex is not None and _timed_regex_search(exc_regex, c.text) is not None:
                        continue
                    hits.append(c)

                # Case 1: check_action (proposed is not None) — strictly single consumption across fresh -> unlinked -> consumed, never masks ungranted historical hits
                if proposed is not None:
                    prop_sid = getattr(proposed, "session_id", None) or (
                        proposed.meta.get("session_id") if isinstance(proposed.meta, dict) else None
                    )
                    if not prop_sid or prop_sid != session_id:
                        return finding, False
                    if not _eligible(proposed, rule):
                        return finding, False
                    prop_in_log = proposed.chunk_id in ids
                    if prop_in_log:
                        if ids[proposed.chunk_id] < start:
                            return finding, False
                        existing_prop = chunks_by_id.get(proposed.chunk_id)
                        if existing_prop is None or existing_prop.text != proposed.text or existing_prop.kind != proposed.kind:
                            return finding, False
                    if _timed_regex_search(rule_regex, proposed.text) is None:
                        return finding, False
                    if exc_regex is not None and _timed_regex_search(exc_regex, proposed.text) is not None:
                        return finding, False
                    hist_hits = [h for h in hits if h.chunk_id != proposed.chunk_id]
                    allowed_cited = {h.chunk_id for h in hist_hits} | {proposed.chunk_id}
                    if cited_ids and not set(cited_ids).issubset(allowed_cited):
                        return finding, False
                    used_hist_grants: set[str] = set()
                    hist_applied_receipts: list[dict[str, Any]] = []
                    for hist_hit in hist_hits:
                        _charge_budget(time.perf_counter())
                        h_sid = getattr(hist_hit, "session_id", None) or (
                            hist_hit.meta.get("session_id") if isinstance(hist_hit.meta, dict) else None
                        )
                        if not h_sid or h_sid != session_id:
                            return finding, False
                        hist_pos = ids.get(hist_hit.chunk_id, -1)
                        hist_covered = False
                        for wg in working_grants:
                            if wg.grant_id in used_hist_grants:
                                continue
                            if _is_valid_consumed_grant(wg) and hist_hit.chunk_id == wg.consumed_event_ids[0]:
                                a_pos = ids.get(wg.source_event_id, -1)
                                i_pos = ids.get(wg.issued_after_event_id, -1) if wg.issued_after_event_id else -1
                                if a_pos < 0 or i_pos < 0 or hist_pos <= a_pos or hist_pos <= i_pos:
                                    continue
                                h_ok, h_dig, h_prev = _timed_grant_match_digest(wg, rule, hist_hit.text)
                                if not h_ok or not h_dig:
                                    continue
                                hist_rcpt = wg.receipts[0]
                                if hist_rcpt.event_id != hist_hit.chunk_id or hist_rcpt.unlinked:
                                    continue
                                if (
                                    not math.isfinite(hist_rcpt.used_at_epoch)
                                    or not math.isfinite(wg.created_at_epoch)
                                    or not math.isfinite(wg.expires_at_epoch)
                                    or hist_rcpt.used_at_epoch < wg.created_at_epoch
                                    or hist_rcpt.used_at_epoch > wg.expires_at_epoch
                                ):
                                    continue
                                if not hist_rcpt.matched_text_digest or hist_rcpt.matched_text_digest != h_dig:
                                    continue
                                safe_h_prev = mask_secrets(h_prev or "")[:120]
                                used_hist_grants.add(wg.grant_id)
                                hist_applied_receipts.append({
                                    "grant_id": wg.grant_id,
                                    "contract_id": rule.id,
                                    "source_event_id": wg.source_event_id,
                                    "event_id": hist_hit.chunk_id,
                                    "unlinked": False,
                                    "used_at": hist_rcpt.used_at,
                                    "uses_remaining": wg.uses_remaining,
                                    "max_uses": wg.max_uses,
                                    "expires_at": wg.expires_at,
                                    "matched_text": safe_h_prev,
                                    "matched_text_digest": h_dig,
                                })
                                hist_covered = True
                                break
                        if not hist_covered:
                            for wg in working_grants:
                                if wg.grant_id in used_hist_grants:
                                    continue
                                a_pos = ids.get(wg.source_event_id, -1)
                                i_pos = ids.get(wg.issued_after_event_id, -1) if wg.issued_after_event_id else -1
                                if a_pos < 0 or i_pos < 0 or hist_pos <= a_pos or hist_pos <= i_pos:
                                    continue
                                if wg.is_expired(now_ts) or not _is_valid_unlinked_grant(wg):
                                    continue
                                h_ok, h_dig, h_prev = _timed_grant_match_digest(wg, rule, hist_hit.text)
                                if not h_ok or not h_dig or h_dig != wg.unlinked_check_texts[0]:
                                    continue
                                hist_unlinked = wg.receipts[0]
                                if (
                                    not hist_unlinked.unlinked
                                    or (hist_unlinked.event_id is not None and hist_unlinked.event_id != hist_hit.chunk_id)
                                    or hist_unlinked.matched_text_digest != h_dig
                                    or not math.isfinite(hist_unlinked.used_at_epoch)
                                    or not math.isfinite(wg.created_at_epoch)
                                    or not math.isfinite(wg.expires_at_epoch)
                                    or not (wg.created_at_epoch <= hist_unlinked.used_at_epoch <= wg.expires_at_epoch)
                                ):
                                    continue
                                safe_h_prev = mask_secrets(h_prev or "")[:120]
                                wg.unlinked_check_texts.remove(h_dig)
                                wg.consumed_event_ids.append(hist_hit.chunk_id)
                                wg.uses_consumed = max(
                                    wg.uses_consumed,
                                    len(wg.consumed_event_ids) + len(wg.unlinked_check_texts),
                                )
                                hist_unlinked.unlinked = False
                                hist_unlinked.event_id = hist_hit.chunk_id
                                hist_unlinked.matched_text = safe_h_prev
                                dirty = True
                                used_hist_grants.add(wg.grant_id)
                                hist_applied_receipts.append({
                                    "grant_id": wg.grant_id,
                                    "contract_id": rule.id,
                                    "source_event_id": wg.source_event_id,
                                    "event_id": hist_hit.chunk_id,
                                    "unlinked": False,
                                    "used_at": hist_unlinked.used_at,
                                    "uses_remaining": wg.uses_remaining,
                                    "max_uses": wg.max_uses,
                                    "expires_at": wg.expires_at,
                                    "matched_text": safe_h_prev,
                                    "matched_text_digest": h_dig,
                                })
                                hist_covered = True
                                break
                        if not hist_covered:
                            return finding, False

                    pinned_event_id = (
                        proposed.chunk_id
                        if (
                            proposed.chunk_id
                            and proposed.chunk_id not in ("proposed", "proposed_action")
                            and not proposed.chunk_id.startswith("proposed:")
                        )
                        else None
                    )
                    chosen_grant: GrantRecord | None = None
                    chosen_receipt_dict: dict[str, Any] | None = None

                    # Step A: If proposed is already in the log and already bound to a consumed grant, replay that exact binding
                    if prop_in_log:
                        prop_pos = ids[proposed.chunk_id]
                        for grant in working_grants:
                            if grant.grant_id in used_hist_grants:
                                continue
                            if _is_valid_consumed_grant(grant) and grant.consumed_event_ids[0] == proposed.chunk_id:
                                a_pos = ids.get(grant.source_event_id, -1)
                                i_pos = ids.get(grant.issued_after_event_id, -1) if grant.issued_after_event_id else -1
                                if a_pos < 0 or i_pos < 0 or prop_pos <= a_pos or prop_pos <= i_pos:
                                    continue
                                matched, match_digest, match_preview = _timed_grant_match_digest(grant, rule, proposed.text)
                                if not matched or not match_digest:
                                    continue
                                rcpt = grant.receipts[0]
                                if rcpt.event_id != proposed.chunk_id or rcpt.unlinked:
                                    continue
                                if (
                                    not math.isfinite(rcpt.used_at_epoch)
                                    or not math.isfinite(grant.created_at_epoch)
                                    or not math.isfinite(grant.expires_at_epoch)
                                    or not (grant.created_at_epoch <= rcpt.used_at_epoch <= grant.expires_at_epoch)
                                    or rcpt.matched_text_digest != match_digest
                                ):
                                    continue
                                safe_match_preview = mask_secrets(match_preview or "")[:120]
                                chosen_grant = grant
                                chosen_receipt_dict = {
                                    "grant_id": grant.grant_id,
                                    "contract_id": rule.id,
                                    "source_event_id": grant.source_event_id,
                                    "event_id": proposed.chunk_id,
                                    "unlinked": False,
                                    "used_at": rcpt.used_at,
                                    "uses_remaining": grant.uses_remaining,
                                    "max_uses": grant.max_uses,
                                    "expires_at": grant.expires_at,
                                    "matched_text": safe_match_preview,
                                    "matched_text_digest": match_digest,
                                }
                                break

                    # Step B: ONLY when proposed is now in the log (prop_in_log is True), promote a matching unlinked pre-check to consumed.
                    # Pre-log check_action calls (prop_in_log is False) NEVER reuse an unlinked grant, keeping pre-log check_action strictly one-time!
                    if chosen_grant is None and prop_in_log:
                        for grant in working_grants:
                            if grant.grant_id in used_hist_grants:
                                continue
                            if grant.source_event_id not in ids or not grant.issued_after_event_id or grant.issued_after_event_id not in ids:
                                continue
                            if ids[grant.issued_after_event_id] < ids[grant.source_event_id]:
                                continue
                            if ids[proposed.chunk_id] <= ids[grant.source_event_id] or ids[proposed.chunk_id] <= ids[grant.issued_after_event_id]:
                                continue
                            if grant.is_expired(now_ts) or not _is_valid_unlinked_grant(grant):
                                continue
                            matched, match_digest, match_preview = _timed_grant_match_digest(grant, rule, proposed.text)
                            if not matched or not match_digest or match_digest != grant.unlinked_check_texts[0]:
                                continue
                            rcpt = grant.receipts[0]
                            if (
                                not rcpt.unlinked
                                or (rcpt.event_id is not None and rcpt.event_id != pinned_event_id)
                                or rcpt.matched_text_digest != match_digest
                                or not math.isfinite(rcpt.used_at_epoch)
                                or not math.isfinite(grant.created_at_epoch)
                                or not math.isfinite(grant.expires_at_epoch)
                                or not (grant.created_at_epoch <= rcpt.used_at_epoch <= grant.expires_at_epoch)
                            ):
                                continue
                            safe_match_preview = mask_secrets(match_preview or "")[:120]
                            grant.unlinked_check_texts.remove(match_digest)
                            grant.consumed_event_ids.append(proposed.chunk_id)
                            grant.uses_consumed = max(
                                grant.uses_consumed,
                                len(grant.consumed_event_ids) + len(grant.unlinked_check_texts),
                            )
                            rcpt.unlinked = False
                            rcpt.event_id = proposed.chunk_id
                            rcpt.matched_text = safe_match_preview
                            dirty = True
                            chosen_grant = grant
                            chosen_receipt_dict = {
                                "grant_id": grant.grant_id,
                                "contract_id": rule.id,
                                "source_event_id": grant.source_event_id,
                                "event_id": proposed.chunk_id,
                                "unlinked": False,
                                "used_at": rcpt.used_at,
                                "uses_remaining": grant.uses_remaining,
                                "max_uses": grant.max_uses,
                                "expires_at": grant.expires_at,
                                "matched_text": safe_match_preview,
                                "matched_text_digest": match_digest,
                            }
                            break

                    # Step C: Otherwise, consume a fresh grant slot (strictly one-time: uses_consumed transitions 0 -> 1)
                    if chosen_grant is None:
                        for grant in working_grants:
                            if grant.grant_id in used_hist_grants:
                                continue
                            if grant.source_event_id not in ids or not grant.issued_after_event_id or grant.issued_after_event_id not in ids:
                                continue
                            if ids[grant.issued_after_event_id] < ids[grant.source_event_id]:
                                continue
                            if prop_in_log:
                                if ids[proposed.chunk_id] <= ids[grant.source_event_id] or ids[proposed.chunk_id] <= ids[grant.issued_after_event_id]:
                                    continue
                            if not _is_valid_fresh_grant(grant, now_ts):
                                continue
                            matched, match_digest, match_preview = _timed_grant_match_digest(grant, rule, proposed.text)
                            if matched and match_digest:
                                safe_match_preview = mask_secrets(match_preview or "")[:120]
                                used_iso = _iso_utc(now_ts)
                                if prop_in_log:
                                    grant.consumed_event_ids.append(proposed.chunk_id)
                                    eff_event_id: str | None = proposed.chunk_id
                                    eff_unlinked = False
                                else:
                                    grant.unlinked_check_texts.append(match_digest)
                                    eff_event_id = pinned_event_id
                                    eff_unlinked = True
                                grant.uses_consumed = max(
                                    grant.uses_consumed + 1,
                                    len(grant.consumed_event_ids) + len(grant.unlinked_check_texts),
                                )
                                receipt = GrantReceipt(
                                    grant_id=grant.grant_id,
                                    contract_id=rule.id,
                                    source_event_id=grant.source_event_id,
                                    event_id=eff_event_id,
                                    unlinked=eff_unlinked,
                                    used_at=used_iso,
                                    used_at_epoch=now_ts,
                                    matched_text=safe_match_preview,
                                    matched_text_digest=match_digest,
                                )
                                grant.receipts.append(receipt)
                                dirty = True
                                chosen_grant = grant
                                chosen_receipt_dict = {
                                    "grant_id": grant.grant_id,
                                    "contract_id": rule.id,
                                    "source_event_id": grant.source_event_id,
                                    "event_id": eff_event_id,
                                    "unlinked": eff_unlinked,
                                    "used_at": used_iso,
                                    "uses_remaining": grant.uses_remaining,
                                    "max_uses": grant.max_uses,
                                    "expires_at": grant.expires_at,
                                    "matched_text": safe_match_preview,
                                    "matched_text_digest": match_digest,
                                }
                                break

                    if chosen_grant is not None and chosen_receipt_dict is not None:
                        if dirty:
                            for idx, wg in zip(grant_indices, working_grants):
                                doc.grants[idx] = wg
                        reason = (
                            f"allowed_by_grant: {chosen_grant.grant_id} anchored to user event {chosen_grant.source_event_id} "
                            f"(uses_remaining={chosen_grant.uses_remaining}/{chosen_grant.max_uses})"
                        )
                        basis = (
                            f"allowed_by_grant (grant_id={chosen_grant.grant_id}, source_event_id={chosen_grant.source_event_id}, "
                            f"used_at={chosen_receipt_dict['used_at']}, matched_text_digest={chosen_receipt_dict['matched_text_digest']})"
                        )
                        clean_ev = ([evidence(h) for h in hist_hits] + [evidence(proposed)])[:20]
                        return Finding(
                            contract_id=finding.contract_id,
                            contract_revision=finding.contract_revision,
                            severity=finding.severity,
                            verdict="clean",
                            reason=reason,
                            basis=basis,
                            evidence=clean_ev,
                            applicability=finding.applicability,
                            proposals_rejected=finding.proposals_rejected,
                            grants_applied=hist_applied_receipts + [chosen_receipt_dict],
                        ), dirty
                    return finding, False

                # Case 2: audit_session (proposed is None)
                if not hits:
                    return finding, False
                if cited_ids and not set(cited_ids).issubset({h.chunk_id for h in hits}):
                    return finding, False

                uncovered_hits: list[AuditEvent] = []
                covered_hits: list[AuditEvent] = []
                applied_receipts: list[dict[str, Any]] = []
                used_audit_grants: set[str] = set()

                for hit in hits:
                    _charge_budget(time.perf_counter())
                    h_sid = getattr(hit, "session_id", None) or (
                        hit.meta.get("session_id") if isinstance(hit.meta, dict) else None
                    )
                    if not h_sid or h_sid != session_id:
                        return finding, False
                    hit_pos = ids.get(hit.chunk_id, -1)
                    covered_by: GrantRecord | None = None
                    used_iso: str | None = None
                    hit_digest: str | None = None
                    hit_preview: str | None = None

                    # First check if this chunk_id was already bound to a grant in an earlier audit (re-verifying match and ordering!)
                    for grant in working_grants:
                        if grant.grant_id in used_audit_grants:
                            continue
                        if _is_valid_consumed_grant(grant) and hit.chunk_id == grant.consumed_event_ids[0]:
                            anchor_pos = ids.get(grant.source_event_id, -1)
                            issued_pos = ids.get(grant.issued_after_event_id, -1) if grant.issued_after_event_id else -1
                            if anchor_pos < 0 or issued_pos < 0 or hit_pos <= anchor_pos or hit_pos <= issued_pos:
                                continue
                            matched, candidate_digest, candidate_preview = _timed_grant_match_digest(grant, rule, hit.text)
                            if not matched or not candidate_digest:
                                continue
                            matching_rcpt = grant.receipts[0]
                            if matching_rcpt.event_id != hit.chunk_id or matching_rcpt.unlinked:
                                continue
                            if (
                                not math.isfinite(matching_rcpt.used_at_epoch)
                                or not math.isfinite(grant.created_at_epoch)
                                or not math.isfinite(grant.expires_at_epoch)
                                or matching_rcpt.used_at_epoch < grant.created_at_epoch
                                or matching_rcpt.used_at_epoch > grant.expires_at_epoch
                            ):
                                continue
                            if not matching_rcpt.matched_text_digest or matching_rcpt.matched_text_digest != candidate_digest:
                                continue
                            used_audit_grants.add(grant.grant_id)
                            covered_by = grant
                            used_iso = matching_rcpt.used_at
                            hit_digest = candidate_digest
                            hit_preview = mask_secrets(candidate_preview or "")[:120]
                            break

                    # Otherwise try to bind an active grant whose source_event_id AND issued_after_event_id precede hit
                    if covered_by is None:
                        for grant in working_grants:
                            if grant.grant_id in used_audit_grants:
                                continue
                            anchor_pos = ids.get(grant.source_event_id, -1)
                            issued_pos = ids.get(grant.issued_after_event_id, -1) if grant.issued_after_event_id else -1
                            if anchor_pos < 0 or issued_pos < 0 or hit_pos <= anchor_pos or hit_pos <= issued_pos:
                                continue
                            if grant.is_expired(now_ts):
                                continue
                            matched, candidate_digest, candidate_preview = _timed_grant_match_digest(grant, rule, hit.text)
                            if not matched or not candidate_digest:
                                continue
                            safe_cand_prev = mask_secrets(candidate_preview or "")[:120]
                            # 1) Bind a pre-checked slot from check_action ONLY if not expired and receipt matches!
                            if (
                                _is_valid_unlinked_grant(grant)
                                and candidate_digest == grant.unlinked_check_texts[0]
                            ):
                                matching_unlinked = grant.receipts[0]
                                if (
                                    not matching_unlinked.unlinked
                                    or (
                                        matching_unlinked.event_id is not None
                                        and matching_unlinked.event_id != hit.chunk_id
                                    )
                                    or matching_unlinked.matched_text_digest != candidate_digest
                                    or not math.isfinite(matching_unlinked.used_at_epoch)
                                    or not math.isfinite(grant.created_at_epoch)
                                    or not math.isfinite(grant.expires_at_epoch)
                                    or not (grant.created_at_epoch <= matching_unlinked.used_at_epoch <= grant.expires_at_epoch)
                                ):
                                    continue
                                grant.unlinked_check_texts.remove(candidate_digest)
                                grant.consumed_event_ids.append(hit.chunk_id)
                                grant.uses_consumed = max(
                                    grant.uses_consumed,
                                    len(grant.consumed_event_ids) + len(grant.unlinked_check_texts),
                                )
                                matching_unlinked.unlinked = False
                                matching_unlinked.event_id = hit.chunk_id
                                matching_unlinked.matched_text = safe_cand_prev
                                used_iso = matching_unlinked.used_at
                                hit_digest = candidate_digest
                                hit_preview = safe_cand_prev
                                used_audit_grants.add(grant.grant_id)
                                covered_by = grant
                                dirty = True
                                break
                            # 2) Consume an unconsumed grant slot directly
                            elif _is_valid_fresh_grant(grant, now_ts):
                                grant.consumed_event_ids.append(hit.chunk_id)
                                grant.uses_consumed = max(
                                    grant.uses_consumed + 1,
                                    len(grant.consumed_event_ids) + len(grant.unlinked_check_texts),
                                )
                                used_iso = _iso_utc(now_ts)
                                hit_digest = candidate_digest
                                hit_preview = safe_cand_prev
                                grant.receipts.append(
                                    GrantReceipt(
                                        grant_id=grant.grant_id,
                                        contract_id=rule.id,
                                        source_event_id=grant.source_event_id,
                                        event_id=hit.chunk_id,
                                        used_at=used_iso,
                                        used_at_epoch=now_ts,
                                        matched_text=safe_cand_prev,
                                        matched_text_digest=candidate_digest,
                                    )
                                )
                                used_audit_grants.add(grant.grant_id)
                                covered_by = grant
                                dirty = True
                                break

                    if covered_by is not None:
                        covered_hits.append(hit)
                        applied_receipts.append({
                            "grant_id": covered_by.grant_id,
                            "contract_id": rule.id,
                            "source_event_id": covered_by.source_event_id,
                            "event_id": hit.chunk_id,
                            "unlinked": False,
                            "used_at": used_iso or covered_by.created_at,
                            "uses_remaining": covered_by.uses_remaining,
                            "max_uses": covered_by.max_uses,
                            "expires_at": covered_by.expires_at,
                            "matched_text": hit_preview,
                            "matched_text_digest": hit_digest,
                        })
                    else:
                        uncovered_hits.append(hit)
        except ContractRegexTimeout:
            return finding, False

        if dirty:
            for idx, wg in zip(grant_indices, working_grants):
                doc.grants[idx] = wg

        if not applied_receipts:
            return finding, dirty

        if not uncovered_hits:
            first = applied_receipts[0]
            reason = (
                f"allowed_by_grant: {len(applied_receipts)} matching event(s) covered by user-anchored grant "
                f"{first['grant_id']} (source_event_id={first['source_event_id']})"
            )
            basis = (
                f"allowed_by_grant (grant_id={first['grant_id']}, source_event_id={first['source_event_id']}, "
                f"used_at={first['used_at']}, matched_text_digest={first.get('matched_text_digest')})"
            )
            preserved_evidence = [evidence(c) for c in covered_hits[:20]]
            return Finding(
                contract_id=finding.contract_id,
                contract_revision=finding.contract_revision,
                severity=finding.severity,
                verdict="clean",
                reason=reason,
                basis=basis,
                evidence=preserved_evidence,
                applicability=finding.applicability,
                proposals_rejected=finding.proposals_rejected,
                grants_applied=applied_receipts,
            ), dirty

        reason = f"{rule.id}.regex matched {len(uncovered_hits)} ungranted event(s) ({len(applied_receipts)} granted)"
        if rule.source_quote:
            reason += f"; user said: {mask_secrets(rule.source_quote)[:120]}"
        return Finding(
            contract_id=finding.contract_id,
            contract_revision=finding.contract_revision,
            severity=finding.severity,
            verdict=finding.verdict,
            reason=reason,
            basis=finding.basis,
            evidence=[evidence(c) for c in uncovered_hits[:20]],
            applicability=finding.applicability,
            proposals_rejected=finding.proposals_rejected,
            grants_applied=applied_receipts,
        ), dirty
