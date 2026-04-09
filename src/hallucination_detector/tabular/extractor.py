"""
Извлечение признаков (4 слоя): дешёвые текстовые, TF-IDF, семантика (эмбеддинги),
опционально LM-статистики, judge-колонки (если есть в таблице).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

_SENT_SPLIT = re.compile(r"[.!?…\n]+")
_DIGITS = re.compile(r"\d+")
_DISCLAIMER = re.compile(
    r"(не уверен|не могу|i don't know|as an ai|я языковая модель|"
    r"hallucination|не имею доступа|cannot verify)",
    re.IGNORECASE,
)
_LATIN = re.compile(r"[a-zA-Z]")
_CYR = re.compile(r"[а-яА-ЯёЁ]")


def _word_tokens(s: str) -> list[str]:
    return re.findall(r"\S+", s.lower())


def _text_stats_row(prompt: str, response: str) -> list[float]:
    p, r = str(prompt), str(response)
    wp, wr = _word_tokens(p), _word_tokens(r)
    all_w = set(wp) | set(wr)
    inter = set(wp) & set(wr)
    jacc = len(inter) / len(all_w) if all_w else 0.0
    dup_r = 1.0 - (len(set(wr)) / len(wr)) if wr else 0.0

    def lat_share(s: str) -> float:
        lat, cyr = len(_LATIN.findall(s)), len(_CYR.findall(s))
        tot = lat + cyr
        return lat / tot if tot else 0.0

    def cyr_share(s: str) -> float:
        lat, cyr = len(_LATIN.findall(s)), len(_CYR.findall(s))
        tot = lat + cyr
        return cyr / tot if tot else 0.0

    return [
        float(len(p)),
        float(len(r)),
        float(len(wp)),
        float(len(wr)),
        float(len([x for x in _SENT_SPLIT.split(p) if x.strip()])),
        float(len([x for x in _SENT_SPLIT.split(r) if x.strip()])),
        float(len(_DIGITS.findall(p))),
        float(len(_DIGITS.findall(r))),
        lat_share(p),
        lat_share(r),
        cyr_share(p),
        cyr_share(r),
        float(jacc),
        float(dup_r),
        float(1.0 if _DISCLAIMER.search(r) else 0.0),
        float(len(p) / max(len(r), 1)),
    ]


STAT_NAMES = [
    "len_chars_prompt",
    "len_chars_response",
    "len_words_prompt",
    "len_words_response",
    "num_sent_prompt",
    "num_sent_response",
    "num_numbers_prompt",
    "num_numbers_response",
    "latin_share_prompt",
    "latin_share_response",
    "cyrillic_share_prompt",
    "cyrillic_share_response",
    "word_jaccard",
    "response_dup_token_ratio",
    "has_disclaimer_response",
    "len_ratio_pr",
]


def _stats_matrix(prompts: list[str], responses: list[str]) -> np.ndarray:
    return np.array(
        [_text_stats_row(p, r) for p, r in zip(prompts, responses)],
        dtype=np.float64,
    )


def _lm_nll_ppl(
    model: Any,
    tokenizer: Any,
    device: Any,
    texts: list[str],
    max_length: int = 256,
) -> np.ndarray:
    import torch

    model.eval()
    nlls: list[float] = []
    with torch.no_grad():
        for t in texts:
            t = str(t).strip()
            if not t:
                nlls.append(0.0)
                continue
            enc = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            out = model(**enc, labels=enc["input_ids"])
            nlls.append(float(out.loss.item()))
    arr = np.array(nlls, dtype=np.float64).reshape(-1, 1)
    ppl = np.exp(np.clip(arr, -20, 20))
    return np.hstack([arr, ppl])


@dataclass
class TabularFeatureExtractor:
    embedding_model: str = "all-MiniLM-L6-v2"
    tfidf_max_features: int = 1500
    use_lm_features: bool = False
    lm_model_name: str = "distilgpt2"
    judge_columns: tuple[str, ...] = (
        "judge_relevance",
        "judge_factuality",
        "judge_consistency",
        "judge_completeness",
    )

    _tfidf_prompt: TfidfVectorizer | None = field(default=None, init=False)
    _tfidf_response: TfidfVectorizer | None = field(default=None, init=False)
    _tfidf_concat: TfidfVectorizer | None = field(default=None, init=False)
    _scaler_stats: StandardScaler | None = field(default=None, init=False)
    _embedder: Any = field(default=None, init=False, repr=False)
    _lm_model: Any = field(default=None, init=False, repr=False)
    _lm_tokenizer: Any = field(default=None, init=False, repr=False)
    _lm_device: Any = field(default=None, init=False, repr=False)
    _embed_dim: int = field(default=384, init=False)
    _feature_names: list[str] = field(default_factory=list, init=False)

    def fit(self, df: pd.DataFrame, prompt_col: str, response_col: str) -> TabularFeatureExtractor:
        prompts = df[prompt_col].astype(str).tolist()
        responses = df[response_col].astype(str).tolist()
        concat = [f"{p} [SEP] {r}" for p, r in zip(prompts, responses)]

        self._tfidf_prompt = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        self._tfidf_response = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        self._tfidf_concat = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        self._tfidf_prompt.fit(prompts)
        self._tfidf_response.fit(responses)
        self._tfidf_concat.fit(concat)

        st = _stats_matrix(prompts, responses)
        self._scaler_stats = StandardScaler()
        self._scaler_stats.fit(st)

        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.embedding_model)
        self._embed_dim = int(self._embedder.get_sentence_embedding_dimension())

        if self.use_lm_features:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._lm_tokenizer = AutoTokenizer.from_pretrained(self.lm_model_name)
            self._lm_model = AutoModelForCausalLM.from_pretrained(self.lm_model_name).to(self._lm_device)
            self._lm_model.eval()

        self._feature_names = self._make_feature_names()
        return self

    def _make_feature_names(self) -> list[str]:
        names: list[str] = [f"stat__{n}" for n in STAT_NAMES]
        assert self._tfidf_prompt is not None
        vp = len(self._tfidf_prompt.vocabulary_)
        vr = len(self._tfidf_response.vocabulary_)  # type: ignore[union-attr]
        vc = len(self._tfidf_concat.vocabulary_)  # type: ignore[union-attr]
        names += [f"tfidf_prompt__{i}" for i in range(vp)]
        names += [f"tfidf_response__{i}" for i in range(vr)]
        names += [f"tfidf_concat__{i}" for i in range(vc)]
        names.append("sem__cosine")
        names += [f"sem__diff_{i}" for i in range(self._embed_dim)]
        names += [f"sem__prod_{i}" for i in range(self._embed_dim)]
        if self.use_lm_features:
            names += ["lm__response_nll", "lm__response_ppl"]
        for jc in self.judge_columns:
            names.append(f"judge__{jc}")
        return names

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def transform(self, df: pd.DataFrame, prompt_col: str, response_col: str) -> sparse.csr_matrix:
        prompts = df[prompt_col].astype(str).tolist()
        responses = df[response_col].astype(str).tolist()
        concat = [f"{p} [SEP] {r}" for p, r in zip(prompts, responses)]

        st = _stats_matrix(prompts, responses)
        st = self._scaler_stats.transform(st)  # type: ignore[union-attr]

        tp = self._tfidf_prompt.transform(prompts)  # type: ignore[union-attr]
        tr = self._tfidf_response.transform(responses)  # type: ignore[union-attr]
        tc = self._tfidf_concat.transform(concat)  # type: ignore[union-attr]

        ep = np.asarray(self._embedder.encode(prompts, batch_size=64, show_progress_bar=False))
        er = np.asarray(self._embedder.encode(responses, batch_size=64, show_progress_bar=False))
        cos = np.sum(ep * er, axis=1) / (
            np.linalg.norm(ep, axis=1) * np.linalg.norm(er, axis=1) + 1e-8
        )
        cos = cos.reshape(-1, 1)
        diff = ep - er
        prod = ep * er
        sem = np.hstack([cos, diff, prod])

        blocks: list[Any] = [sparse.csr_matrix(st), tp, tr, tc, sparse.csr_matrix(sem)]

        if self.use_lm_features and self._lm_model is not None:
            lm = _lm_nll_ppl(self._lm_model, self._lm_tokenizer, self._lm_device, responses)
            blocks.append(sparse.csr_matrix(lm))

        judge = np.zeros((len(df), len(self.judge_columns)), dtype=np.float64)
        for j, col in enumerate(self.judge_columns):
            if col in df.columns:
                judge[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
        blocks.append(sparse.csr_matrix(judge))

        return sparse.hstack(blocks, format="csr")

    def fit_transform(
        self, df: pd.DataFrame, prompt_col: str, response_col: str
    ) -> sparse.csr_matrix:
        return self.fit(df, prompt_col, response_col).transform(df, prompt_col, response_col)
