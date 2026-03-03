"""统一响应格式"""
from datetime import datetime, timezone, timedelta

SHANGHAI_TZ = timezone(timedelta(hours=8))


class R:
    def __init__(self):
        self._code = 200
        self._msg = "success"
        self._data = None

    @staticmethod
    def ok():
        return R()

    @staticmethod
    def error(msg: str):
        r = R()
        r._code = 400
        r._msg = msg
        return r._build()

    @staticmethod
    def fail(code: int, msg: str):
        r = R()
        r._code = code
        r._msg = msg
        return r._build()

    def data(self, data):
        self._data = data
        return self._build()

    def _build(self):
        return {
            "code": self._code,
            "msg": self._msg,
            "data": self._data,
            "timestamp": datetime.now(SHANGHAI_TZ).isoformat(),
        }
