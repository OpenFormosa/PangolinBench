"""Tokenizer adapter interfaces for PangolinBench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Any

from .fixtures import REQUIRED_NON_SPEECH_LABELS, REQUIRED_STRUCTURAL_TOKENS


class TokenizerAdapter(ABC):
    """Generic tokenizer interface used by PangolinBench."""

    name: str

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        ...

    @abstractmethod
    def token_to_id(self, token: str) -> int | None:
        ...

    @abstractmethod
    def get_vocab(self) -> dict[str, int]:
        ...

    @abstractmethod
    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        ...

    @abstractmethod
    def supports_special_tokens(self) -> bool:
        ...

    @abstractmethod
    def supports_token_strings(self) -> bool:
        ...


class ByteRoundtripTestAdapter(TokenizerAdapter):
    """UTF-8 byte adapter used only for PangolinBench unit tests and smoke runs."""

    name = "ByteRoundtripTestAdapter"

    def __init__(self) -> None:
        self._special_vocab = {
            token: 100_000 + index
            for index, token in enumerate(REQUIRED_STRUCTURAL_TOKENS + REQUIRED_NON_SPEECH_LABELS)
        }
        self._reverse_special_vocab = {value: key for key, value in self._special_vocab.items()}

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        byte_values: list[int] = []
        text_chunks: list[str] = []
        for token_id in ids:
            if 0 <= int(token_id) <= 255:
                byte_values.append(int(token_id))
                continue
            if token_id in self._reverse_special_vocab:
                if byte_values:
                    text_chunks.append(bytes(byte_values).decode("utf-8", errors="strict"))
                    byte_values = []
                if not skip_special_tokens:
                    text_chunks.append(self._reverse_special_vocab[token_id])
                continue
            raise ValueError(f"ByteRoundtripTestAdapter cannot decode token id {token_id}")
        if byte_values:
            text_chunks.append(bytes(byte_values).decode("utf-8", errors="strict"))
        return "".join(text_chunks)

    def token_to_id(self, token: str) -> int | None:
        if len(token) == 1:
            encoded = token.encode("utf-8")
            if len(encoded) == 1:
                return encoded[0]
        return self._special_vocab.get(token)

    def get_vocab(self) -> dict[str, int]:
        vocab = {f"<byte_{index:02x}>": index for index in range(256)}
        vocab.update(self._special_vocab)
        return vocab

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        tokens: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if 0 <= token_id <= 255:
                tokens.append(f"<byte_{token_id:02x}>")
            elif token_id in self._reverse_special_vocab:
                tokens.append(self._reverse_special_vocab[token_id])
            else:
                tokens.append(f"<unk_{token_id}>")
        return tokens

    def supports_special_tokens(self) -> bool:
        return True

    def supports_token_strings(self) -> bool:
        return True


REFERENCE_SEGMENTATION_TERMS = [
    "全民健康保險",
    "戶政事務所",
    "勞保年金",
    "科學園區",
    "晶圓代工",
    "半導體",
    "數位服務",
    "校務系統",
    "繳費紀錄",
    "門診流程",
    "金融服務",
    "測試資料",
    "風險控管",
    "技術文件",
    "模型訓練",
    "資料驗證",
    "版本控管",
    "這樣可以嗎",
    "不會吧",
    "有差嗎",
    "看一下",
    "等一下",
    "地名詞表",
    "離島",
    "東部",
    "欄位",
    "制度",
    "生活詞",
    "產業詞",
    "台北",
    "臺北",
    "新北",
    "桃園",
    "新竹",
    "台中",
    "臺中",
    "台南",
    "臺南",
    "高雄",
    "花蓮",
    "台東",
    "臺東",
    "澎湖",
    "金門",
    "馬祖",
    "健保",
    "勞保",
    "里長",
    "捷運",
    "台語",
    "母語",
    "漢羅混寫",
    "壓力測試",
    "客語",
    "客庄",
    "四縣",
    "海陸",
    "大埔",
    "饒平",
    "詔安",
    "基礎注音",
    "注音測試",
    "混合測試",
    "中文",
    "注音",
    "今天",
    "下降",
    "訓練成本",
    "需要",
    "記錄",
    "都要",
    "保留",
    "格式",
    "大家好",
    "歡迎",
    "回來",
    "我們",
    "討論",
    "台灣",
    "本土",
    "語音模型",
    "注音",
    "混合",
    "測試",
    "長會議",
    "時間戳",
    "全形標點",
    "頓號",
    "句號",
    "日期",
    "電話",
    "括號",
    "本院",
    "今日",
    "公布",
    "新的",
    "規範",
    "更新",
    "學生",
    "查詢",
    "課程",
    "調整",
    "報到",
    "量測",
    "最後",
    "完成",
    "諮詢",
    "包含",
    "匯款",
    "帳戶",
    "描述",
    "結果",
    "怪",
    "覺得",
    "蠻順",
    "可以",
    "先",
    "不要",
]


class ReferenceSegmentationAdapter(TokenizerAdapter):
    """Deterministic reference segmentation baseline for PangolinBench fixtures.

    This is a plausible tokenizer best line for the synthetic benchmark, not an
    authoritative linguistic word segmenter and not a production tokenizer.
    """

    name = "PangolinBenchReferenceSegmentationAdapter"

    _url_re = re.compile(r"https?://[^\s，。！？、；：「」『』（）《》〈〉【】]+")
    _email_re = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
    _date_re = re.compile(r"\d{4}-\d{2}-\d{2}")
    _phone_re = re.compile(r"\d{2}-\d{4}-\d{4}")
    _number_re = re.compile(r"\d+(?:\.\d+)?%?")
    _latin_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ɏ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿĀ-ɏ]+)*(?:\d+)?")
    _fullwidth_latin_re = re.compile(r"[Ａ-Ｚａ-ｚ]+")
    _bopomofo_re = re.compile(r"[\u3100-\u312f\u31a0-\u31bfˊˇˋ˙]+")

    def __init__(self) -> None:
        self._terms = sorted(set(REFERENCE_SEGMENTATION_TERMS), key=len, reverse=True)
        self._special_tokens = sorted(
            set(REQUIRED_STRUCTURAL_TOKENS + REQUIRED_NON_SPEECH_LABELS),
            key=len,
            reverse=True,
        )
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        for token in self._special_tokens + self._terms:
            self._id_for_token(token)

    def encode(self, text: str) -> list[int]:
        return [self._id_for_token(token) for token in self.segment(text)]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        chunks: list[str] = []
        for token_id in ids:
            token = self._id_to_token[int(token_id)]
            if skip_special_tokens and token in self._special_tokens:
                continue
            chunks.append(token)
        return "".join(chunks)

    def token_to_id(self, token: str) -> int | None:
        return self._token_to_id.get(token)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_token[int(token_id)] for token_id in ids]

    def supports_special_tokens(self) -> bool:
        return True

    def supports_token_strings(self) -> bool:
        return True

    def segment(self, text: str) -> list[str]:
        tokens: list[str] = []
        index = 0
        while index < len(text):
            matched = self._match_special_or_term(text, index)
            if matched is not None:
                tokens.append(matched)
                index += len(matched)
                continue
            regex_match = self._match_regex(text, index)
            if regex_match is not None:
                tokens.append(regex_match)
                index += len(regex_match)
                continue
            char = text[index]
            if _is_han(char):
                han_run = self._consume_han_run(text, index)
                tokens.extend(han_run)
                index += sum(len(token) for token in han_run)
                continue
            tokens.append(char)
            index += 1
        return tokens

    def _id_for_token(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = len(self._token_to_id)
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]

    def _match_special_or_term(self, text: str, index: int) -> str | None:
        for token in self._special_tokens:
            if text.startswith(token, index):
                return token
        for term in self._terms:
            if text.startswith(term, index):
                return term
        return None

    def _match_regex(self, text: str, index: int) -> str | None:
        for regex in [
            self._url_re,
            self._email_re,
            self._date_re,
            self._phone_re,
            self._number_re,
            self._latin_re,
            self._fullwidth_latin_re,
            self._bopomofo_re,
        ]:
            match = regex.match(text, index)
            if match:
                return match.group(0)
        return None

    def _consume_han_run(self, text: str, index: int) -> list[str]:
        tokens: list[str] = []
        cursor = index
        while cursor < len(text) and _is_han(text[cursor]):
            matched = None
            for term in self._terms:
                if text.startswith(term, cursor):
                    matched = term
                    break
            if matched is not None:
                tokens.append(matched)
                cursor += len(matched)
            else:
                tokens.append(text[cursor])
                cursor += 1
        return tokens


def _is_han(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x30000 <= codepoint <= 0x3134F
    )


class HuggingFaceTokenizersAdapter(TokenizerAdapter):
    """Adapter for tokenizer.json files handled by the optional tokenizers package."""

    def __init__(self, tokenizer: Any, *, name: str) -> None:
        self._tokenizer = tokenizer
        self.name = name

    @classmethod
    def from_file(cls, path: str | Path) -> "HuggingFaceTokenizersAdapter":
        try:
            try:
                from ubi_tokenizer import Tokenizer  # type: ignore
            except ImportError:
                from tokenizers import Tokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Loading tokenizer.json requires either the optional 'ubi_tokenizer' "
                "package or the 'tokenizers' package to already be installed"
            ) from exc
        path = Path(path)
        return cls(Tokenizer.from_file(str(path)), name=f"tokenizer.json:{path}")

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return str(self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens))

    def token_to_id(self, token: str) -> int | None:
        value = self._tokenizer.token_to_id(token)
        return int(value) if value is not None else None

    def get_vocab(self) -> dict[str, int]:
        return {str(token): int(token_id) for token, token_id in self._tokenizer.get_vocab().items()}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [str(self._tokenizer.id_to_token(int(token_id))) for token_id in ids]

    def supports_special_tokens(self) -> bool:
        return True

    def supports_token_strings(self) -> bool:
        return True


def load_tokenizer_adapter(path: str | Path) -> TokenizerAdapter:
    """Load a tokenizer adapter from a path using repository-compatible conventions."""

    path = Path(path)
    if path.is_dir():
        tokenizer_json = path / "tokenizer.json"
        if tokenizer_json.exists():
            return HuggingFaceTokenizersAdapter.from_file(tokenizer_json)
    if path.name == "tokenizer.json" or path.suffix == ".json":
        return HuggingFaceTokenizersAdapter.from_file(path)
    raise ValueError(f"Unsupported PangolinBench tokenizer path: {path}")
