"""Make yfinance work behind TLS-reterminating proxies that reject Chrome/Firefox
TLS fingerprints. curl_cffi's "chrome" impersonation gets its handshake reset;
"safari" (and no impersonation) pass. Certificate verification stays enabled.

Import this module BEFORE using yfinance:

    from src import yf_compat  # noqa: F401  (patches yfinance on import)
    import yfinance as yf
"""
IMPERSONATE = "safari"


def _make_session():
    from curl_cffi import requests as cr
    return cr.Session(impersonate=IMPERSONATE)


def apply():
    import yfinance._http as yh
    yh.new_session = _make_session
    # These modules imported new_session by name, so patch their references too.
    import yfinance.base
    import yfinance.data
    import yfinance.multi
    yfinance.base.new_session = _make_session
    yfinance.data.new_session = _make_session
    yfinance.multi.new_session = _make_session
    # If the YfData singleton already exists with a bad session, replace it.
    try:
        yfinance.data.YfData(session=_make_session())
    except Exception:
        pass


apply()
