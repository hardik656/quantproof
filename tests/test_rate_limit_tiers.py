"""Daily validation limits match /pricing; only valid tokens raise them."""

import time

from services import tokens


def _drain(ip, token=""):
    n = 0
    while tokens.check_rate_limit(ip, token)[0]:
        n += 1
        if n > 50:
            break
    return n


def setup_function():
    tokens._validation_counts.clear()


def test_free_limit_is_3():
    assert _drain("1.1.1.1") == 3


def test_fake_pro_prefix_does_not_bypass():
    assert _drain("1.1.1.2", "PRO-forged-token") == 3


def test_report_and_portfolio_limits():
    tokens.REPORT_ACCESS_TOKENS.update({"RPT-t1", "PORT-t1", "LITE-t1"})
    assert _drain("1.1.1.3", "RPT-t1") == 5
    assert _drain("1.1.1.4", "PORT-t1") == 10
    assert _drain("1.1.1.5", "LITE-t1") == 3


def test_valid_pro_is_unlimited():
    tokens._PRO_TOKENS["PRO-valid"] = time.time() + 3600
    assert _drain("1.1.1.6", "PRO-valid") > 50
