"""Interface de progresso em terminal (desacoplada da lógica de cálculo e do matplotlib).

Este módulo cuida APENAS da apresentação do progresso no terminal. A lógica de
otimização (em ``orchestrator.py``) permanece agnóstica à interface: ela apenas
emite atualizações via callback, que aqui são renderizadas como uma barra de
progresso com porcentagem, iteração atual, tempo decorrido, ETA e taxa (it/s).
"""

from __future__ import annotations

import sys
import time
from typing import Optional, TextIO


def format_duration(seconds: float) -> str:
    """Formata uma duração em segundos como ``HhMMmSSs`` de forma compacta."""
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class ProgressReporter:
    """Barra de progresso ASCII para o terminal, com ETA e taxa.

    Uso típico::

        reporter = ProgressReporter(total=500, label="Evolution")
        reporter.start()
        for step in range(1, 501):
            ...
            reporter.update(step, best_value=current_best)
        reporter.finish(best_value=final_best)

    É deliberadamente livre de dependências pesadas (só ``sys``/``time``) e usa
    apenas caracteres ASCII para evitar problemas de codificação no Windows.
    """

    def __init__(
        self,
        total: int,
        bar_length: int = 40,
        label: str = "Progress",
        value_label: str = "Best",
        stream: Optional[TextIO] = None,
    ) -> None:
        self.total = max(1, int(total))
        self.bar_length = bar_length
        self.label = label
        self.value_label = value_label
        self.stream: TextIO = stream if stream is not None else sys.stdout
        self.start_time: Optional[float] = None
        self._finished = False
        self._last_line_length = 0

    def start(self) -> None:
        """Marca o início da contagem de tempo (necessário para ETA/elapsed)."""
        self.start_time = time.perf_counter()
        self._finished = False

    def update(self, current: int, best_value: Optional[float] = None) -> None:
        """Renderiza a barra para o passo ``current`` (1-indexado)."""
        if self.start_time is None:
            self.start()

        current = max(0, min(int(current), self.total))
        fraction = current / self.total
        filled = int(self.bar_length * fraction)
        bar = "#" * filled + "-" * (self.bar_length - filled)

        elapsed = time.perf_counter() - self.start_time
        rate = current / elapsed if elapsed > 0 and current > 0 else 0.0
        eta = (self.total - current) / rate if rate > 0 else 0.0

        value_str = ""
        if best_value is not None:
            value_str = f" | {self.value_label}: {best_value:>12.2f}"

        line = (
            f"{self.label} |{bar}| {fraction * 100:6.2f}% "
            f"({current}/{self.total}){value_str} "
            f"| Elapsed: {format_duration(elapsed)} "
            f"| ETA: {format_duration(eta)} "
            f"| {rate:5.1f} it/s"
        )

        padding = max(0, self._last_line_length - len(line))
        self.stream.write("\r" + line + " " * padding)
        self.stream.flush()
        self._last_line_length = len(line)

    def finish(self, best_value: Optional[float] = None) -> None:
        """Completa a barra em 100% e quebra a linha (idempotente)."""
        if self._finished:
            return
        self.update(self.total, best_value)
        self.stream.write("\n")
        self.stream.flush()
        self._finished = True
