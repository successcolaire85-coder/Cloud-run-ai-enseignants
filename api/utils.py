# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from config import Q_COLUMNS, HOURS_COL

def log(msg: str):
    print(msg, flush=True)

def form_to_text(form: Dict[str, Any]) -> str:
    if not isinstance(form, dict):
        return str(form or "")
    parts: List[str] = []
    for _, v in form.items():
        if v is None:
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    txt = ". ".join(parts)
    if txt and not txt.endswith("."):
        txt += "."
    return txt

def concat_row_for_csv(row: Dict[str, Any], include_hours: bool = True) -> str:
    parts: List[str] = []
    for c in Q_COLUMNS:
        v = row.get(c, "")
        s = str("" if v is None else v).replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()
        if s:
            parts.append(s)
    if include_hours and HOURS_COL in row:
        hs = str("" if row[HOURS_COL] is None else row[HOURS_COL]).strip()
        if hs:
            parts.append(hs)
    txt = ". ".join(parts)
    if txt and not txt.endswith("."):
        txt += "."
    return txt
