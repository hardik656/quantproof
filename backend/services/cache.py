"""In-memory LRU cache of validation results (used to bind PDF reports to a run).

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

import io
import os
import time
import asyncio
import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel

import collections
import asyncio as _asyncio
_PDF_SEMAPHORE = _asyncio.Semaphore(2)
VALIDATION_RESULT_CACHE: dict = collections.OrderedDict()
VALIDATION_CACHE_MAX = 50

def _save_validation_cache():
    # No persistence needed - use validation hash as direct cache
    pass

def _load_validation_cache():
    # No persistence needed - use validation hash as direct cache
    pass

def _cache_validation(v_hash: str, vr: dict):
    print(f"[DEBUG] Cache - Storing: '{v_hash}'")
    VALIDATION_RESULT_CACHE[v_hash] = vr
    if len(VALIDATION_RESULT_CACHE) > VALIDATION_CACHE_MAX:
        VALIDATION_RESULT_CACHE.popitem(last=False)
    print(f"[DEBUG] Cache - Size: {len(VALIDATION_RESULT_CACHE)} items")

def _get_cached_validation(v_hash: str) -> dict | None:
    print(f"[DEBUG] Cache Get - Looking for: '{v_hash}'")
    print(f"[DEBUG] Cache Get - Keys: {list(VALIDATION_RESULT_CACHE.keys())}")
    
    if v_hash in VALIDATION_RESULT_CACHE:
        print(f"[DEBUG] Cache Get - FOUND: '{v_hash}'")
        VALIDATION_RESULT_CACHE.move_to_end(v_hash)
        return VALIDATION_RESULT_CACHE[v_hash]
    
    print(f"[DEBUG] Cache Get - MISS: '{v_hash}'")
    return None
