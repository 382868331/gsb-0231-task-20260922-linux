"""纯 ``fractions.Fraction`` 的稠密有理线性代数（无第三方依赖）。

矩阵用 ``list[list[Fraction]]``（行主序）表示，向量用 ``list[Fraction]``。
规模上限为 :data:`resistor_elim.network.MAX_NODES`（30），稠密消元足够快，
且全程精确——Schur 补、逆、解都不引入浮点误差。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from .errors import SingularSystemError


class Mat:
    """``Fraction`` 稠密矩阵上的静态工具方法（不实例化）。"""

    @staticmethod
    def zeros(rows: int, cols: int) -> list[list[Fraction]]:
        return [[Fraction(0) for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def identity(n: int) -> list[list[Fraction]]:
        m = Mat.zeros(n, n)
        for i in range(n):
            m[i][i] = Fraction(1)
        return m

    @staticmethod
    def copy(m: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
        return [list(row) for row in m]

    @staticmethod
    def is_symmetric(m: Sequence[Sequence[Fraction]]) -> bool:
        n = len(m)
        return all(len(m[i]) == n for i in range(n)) and all(
            m[i][j] == m[j][i] for i in range(n) for j in range(i + 1, n)
        )

    @staticmethod
    def rref(
        a: list[list[Fraction]],
        b: list[list[Fraction]] | None = None,
    ) -> list[list[Fraction]]:
        """对增广矩阵 ``[a | b]`` 做高斯-若尔当消元（部分主元，选列）。

        返回消元后的右侧；``a`` 被原地化为单位阵。系统奇异时抛出
        :class:`SingularSystemError`。主元只接受非零 ``Fraction``，
        因此不存在数值容差把“近奇异”误判为奇异的问题。
        """
        n = len(a)
        rhs = Mat.identity(n) if b is None else Mat.copy(b)
        if n == 0:
            return rhs
        if any(len(row) != n for row in a):
            raise ValueError("rref 只支持方阵")
        k = len(rhs[0]) if rhs else 0
        if any(len(row) != k for row in rhs) or len(rhs) != n:
            raise ValueError("右侧形状不匹配")

        for col in range(n):
            pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
            if pivot is None:
                raise SingularSystemError(
                    f"线性系统在第 {col} 个主元处奇异（存在无电导连接的内部节点）"
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
                if factor != 0:
                    a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
                    rhs[r] = [x - factor * y for x, y in zip(rhs[r], rhs[col])]
        return rhs

    @staticmethod
    def inverse(a: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
        """精确求逆；奇异时抛 :class:`SingularSystemError`。"""
        return Mat.rref(Mat.copy(a))

    @staticmethod
    def solve(
        a: Sequence[Sequence[Fraction]],
        b: Sequence[Fraction],
    ) -> list[Fraction]:
        """解 ``a x = b``（精确）；奇异时抛 :class:`SingularSystemError`。"""
        if len(a) != len(b):
            raise ValueError("矩阵与向量维度不匹配")
        rhs_col = Mat.copy([[x] for x in b])
        out = Mat.rref(Mat.copy(a), rhs_col)
        return [row[0] for row in out]

    @staticmethod
    def matvec(
        a: Sequence[Sequence[Fraction]],
        x: Sequence[Fraction],
    ) -> list[Fraction]:
        return [sum((a_ij * x_j for a_ij, x_j in zip(row, x)), Fraction(0)) for row in a]

    @staticmethod
    def matmul(
        a: Sequence[Sequence[Fraction]],
        b: Sequence[Sequence[Fraction]],
    ) -> list[list[Fraction]]:
        if not a or not b:
            return []
        inner = len(b)
        if any(len(row) != inner for row in a):
            raise ValueError("矩阵维度不匹配")
        cols = len(b[0])
        b_cols = [[b[r][c] for r in range(inner)] for c in range(cols)]
        return [
            [
                sum((a[i][r] * b_cols[c][r] for r in range(inner)), Fraction(0))
                for c in range(cols)
            ]
            for i in range(len(a))
        ]

    @staticmethod
    def submatrix(
        a: Sequence[Sequence[Fraction]],
        rows: Sequence[int],
        cols: Sequence[int],
    ) -> list[list[Fraction]]:
        return [[a[i][j] for j in cols] for i in rows]
