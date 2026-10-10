"""Sunwa-AI — Internationalization (i18n) Helper.
Supports: English (en), Hindi (hi), Marathi (mr).
Language stored in Streamlit session_state and falls back to English.
Original complaint text is NEVER translated.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

_LOCALES_DIR = Path(__file__).parent
_CACHE: dict = {}

SUPPORTED_LANGS = ["en", "hi", "mr"]
DEFAULT_LANG = "en"


def _load_translations(lang: str) -> dict:
    if lang in _CACHE:
        return _CACHE[lang]
    path = _LOCALES_DIR / f"{lang}.json"
    if not path.exists():
        logger.warning("Locale file not found: %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        _CACHE[lang] = data
        return data
    except Exception as exc:
        logger.error("Failed to load locale %s: %s", lang, exc)
        return {}


def get_current_lang() -> str:
    return st.session_state.get("_lang", DEFAULT_LANG)


def set_current_lang(lang: str) -> None:
    if lang in SUPPORTED_LANGS:
        st.session_state["_lang"] = lang


def t(key: str, **kwargs) -> str:
    """Translate key to current language, falling back to English then key itself."""
    lang = get_current_lang()
    translations = _load_translations(lang)
    text = translations.get(key)
    if text is None and lang != DEFAULT_LANG:
        en_trans = _load_translations(DEFAULT_LANG)
        text = en_trans.get(key)
    if text is None:
        text = key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text


# Load all translations eagerly for import convenience
TRANSLATIONS = {
    lang: _load_translations(lang) for lang in SUPPORTED_LANGS
}
