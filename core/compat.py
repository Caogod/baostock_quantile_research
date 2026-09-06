"""pandas 兼容性补丁。

baostock 0.9.x 内部仍使用 pandas 2.0+ 已移除的 DataFrame.append，
本模块在导入时为其打补丁，使其委托到 pd.concat，保证在 pandas 3.x 下正常运作。
"""
import pandas as pd

if not hasattr(pd.DataFrame, "append"):
    def _append(self, other, ignore_index=False, sort=False):
        return pd.concat([self, other], ignore_index=ignore_index, sort=sort)

    pd.DataFrame.append = _append
