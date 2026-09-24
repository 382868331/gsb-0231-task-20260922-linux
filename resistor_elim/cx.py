"""精确复数（实部、虚部均为 ``fractions.Fraction``）与稠密复线性代数。

交流相量分析中的所有量——支路导纳 ``jωC``、``1/(jωL)``、MNA 矩阵元素、
端口复导纳——都是 ``a + jb``（``a,b`` 为有理分数）。用 :class:`QComplex`
做精确四则运算，消元过程不引入任何浮点误差，零主元也是精确判定。

仅使用 Python 标准库。
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

from .errors import SingularSystemError


def _as_fraction(value, what: str) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{what} 不得为 bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"{what} 不得为 NaN")
        if math.isinf(value):
            raise ValueError(f"{what} 不得为 Infinity")
        return Fraction(value)
    raise ValueError(
        f"{what} 必须是 int、float（有限）或 Fraction，收到 {type(value).__name__}"
    )


class QComplex:
    """精确复数 ``re + j*im``（``re``、``im`` 为 ``Fraction``）。

    构造接受 ``int`` / ``Fraction`` / 有限 ``float``（按二进制值精确转
    分数），也接受单个 Python 内置 ``complex``（实虚部均须有限）。
    """

    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        if isinstance(re, complex):
            if im != 0:
                raise ValueError("从 complex 构造时不得再给虚部参数")
            self.re = _as_fraction(re.real, "复数实部")
            self.im = _as_fraction(re.imag, "复数虚部")
        else:
            self.re = _as_fraction(re, "复数实部")
            self.im = _as_fraction(im, "复数虚部")

    # ------------------------------------------------------------------ #
    def is_zero(self) -> bool:
        return self.re == 0 and self.im == 0

    def conj(self) -> "QComplex":
        return QComplex(self.re, -self.im)

    def abs_sq(self) -> Fraction:
        """``|z|² = re² + im²``（精确分数）。"""
        return self.re * self.re + self.im * self.im

    # ------------------------------------------------------------------ #
    @staticmethod
    def _wrap(other) -> "QComplex":
        if isinstance(other, QComplex):
            return other
        if isinstance(other, (int, Fraction)):
            return QComplex(other, 0)
        if isinstance(other, float):
            return QComplex(_as_fraction(other, "运算数"), 0)
        return NotImplemented

    def __add__(self, other):
        o = self._wrap(other)
        if o is NotImplemented:
            return NotImplemented
        return QComplex(self.re + o.re, self.im + o.im)

    def __radd__(self, other):
        o = self._wrap(other)
        if o is NotImplemented:
            return NotImplemented
        return QComplex(o.re + self.re, o.im + self.im)

    def __sub__(self, other):
        o = self._wrap(other)
        if o is NotImplemented:
            return NotImplemented
        return QComplex(self.re - o.re, self.im - o.im)

    def __rsub__(self, other):
        o = self._wrap(other)
        if o is NotImplemented:
            return NotImplemented
        return QComplex(o.re - self.re, o.im - self.im)

    def __neg__(self):
        return QComplex(-self.re, -self.im)

    def __mul__(self, other):
        o = self._wrap(other)
        if o is NotImplemented:
            return NotImplemented
        return QComplex(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        o = self._wrap(other)
        if o is NotImplemented:
            return NotImplemented
        denom = o.re * o.re + o.im * o.im
        if denom == 0:
            raise ZeroDivisionError("复数除以零")
        return QComplex(
            (self.re * o.re + self.im * o.im) / denom,
            (self.im * o.re - self.re * o.im) / denom,
        )

    def __rtruediv__(self, other):
        o = self._wrap(other)
        if o is NotImplemented:
            return NotImplemented
        return o / self

    def __eq__(self, other):
        o = self._wrap(other)
        if o is NotImplemented:
            return NotImplemented
        return self.re == o.re and self.im == o.im

    def __bool__(self):
        return not self.is_zero()

    def __hash__(self):
        return hash((self.re, self.im))

    def __repr__(self):
        sign = "+" if self.im >= 0 else "-"
        return f"({self.re}{sign}j*{abs(self.im)})"

    def __str__(self):
        if self.im == 0:
            return str(self.re)
        if self.re == 0:
            return f"j*({self.im})"
        sign = "+" if self.im >= 0 else "-"
        return f"{self.re}{sign}j*{abs(self.im)}"


#: 虚数单位 j。
J = QComplex(0, 1)


class CxMat:
    """``QComplex`` 稠密矩阵上的静态工具方法（不实例化）。"""

    @staticmethod
    def zeros(rows: int, cols: int) -> list[list[QComplex]]:
        return [[QComplex() for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def identity(n: int) -> list[list[QComplex]]:
        m = CxMat.zeros(n, n)
        for i in range(n):
            m[i][i] = QComplex(1)
        return m

    @staticmethod
    def copy(m: Sequence[Sequence[QComplex]]) -> list[list[QComplex]]:
        return [list(row) for row in m]

    @staticmethod
    def is_symmetric(m: Sequence[Sequence[QComplex]]) -> bool:
        """复对称：``m == m^T``（注意不是 Hermitian 共轭对称）。"""
        n = len(m)
        return all(len(m[i]) == n for i in range(n)) and all(
            m[i][j] == m[j][i] for i in range(n) for j in range(i + 1, n)
        )

    @staticmethod
    def submatrix(
        a: Sequence[Sequence[QComplex]],
        rows: Sequence[int],
        cols: Sequence[int],
    ) -> list[list[QComplex]]:
        return [[a[i][j] for j in cols] for i in rows]

    @staticmethod
    def rref(
        a: list[list[QComplex]],
        b: list[list[QComplex]] | None = None,
    ) -> list[list[QComplex]]:
        """对增广矩阵 ``[a | b]`` 做高斯-若尔当消元（列选主元）。

        主元只接受**精确非零**的 :class:`QComplex`；奇异时抛
        :class:`SingularSystemError`，不存在数值容差误判。
        """
        n = len(a)
        rhs = CxMat.identity(n) if b is None else CxMat.copy(b)
        if n == 0:
            return rhs
        if any(len(row) != n for row in a):
            raise ValueError("rref 只支持方阵")
        k = len(rhs[0]) if rhs else 0
        if any(len(row) != k for row in rhs) or len(rhs) != n:
            raise ValueError("右侧形状不匹配")

        for col in range(n):
            pivot = next(
                (r for r in range(col, n) if not a[r][col].is_zero()), None
            )
            if pivot is None:
                raise SingularSystemError(
                    f"修正结点系统在第 {col} 个主元处奇异"
                    "（内部变量块不可逆；请检查频率是否为零、组件是否浮置）"
                )
            if pivot != col:
                a[col], a[pivot] = a[pivot], a[col]
                rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
            pivot_val = a[col][col]
            a[col] = [x / pivot_val for x in a[col]]
            rhs[col] = [x / pivot_val for x in rhs[col]]
            for r in range(n):
                if r == col:
                    continue
                factor = a[r][col]
                if not factor.is_zero():
                    a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
                    rhs[r] = [x - factor * y for x, y in zip(rhs[r], rhs[col])]
        return rhs

    @staticmethod
    def inverse(a: Sequence[Sequence[QComplex]]) -> list[list[QComplex]]:
        """精确求逆；奇异时抛 :class:`SingularSystemError`。"""
        return CxMat.rref(CxMat.copy(a))

    @staticmethod
    def matvec(
        a: Sequence[Sequence[QComplex]],
        x: Sequence[QComplex],
    ) -> list[QComplex]:
        return [
            sum((a_ij * x_j for a_ij, x_j in zip(row, x)), QComplex())
            for row in a
        ]

    @staticmethod
    def matmul(
        a: Sequence[Sequence[QComplex]],
        b: Sequence[Sequence[QComplex]],
    ) -> list[list[QComplex]]:
        if not a or not b:
            return []
        inner = len(b)
        if any(len(row) != inner for row in a):
            raise ValueError("矩阵维度不匹配")
        cols = len(b[0])
        b_cols = [[b[r][c] for r in range(inner)] for c in range(cols)]
        return [
            [
                sum((a[i][r] * b_cols[c][r] for r in range(inner)), QComplex())
                for c in range(cols)
            ]
            for i in range(len(a))
        ]
