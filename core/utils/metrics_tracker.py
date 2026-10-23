from dataclasses import dataclass

from loguru import logger


@dataclass
class MetricsTracker:
    """Centralized metrics tracking for TAME-O."""

    turn_count: int = 0
    question_count: int = 0

    def log_cra(self, cra: float, correct: int, total: int):
        """Log CRA metric with turn counter."""
        self.turn_count += 1
        logger.debug(f"CRA={cra:.3f} ({correct}/{total}) at turn {self.turn_count}")

    def log_cfr(self, cfr: float, fissions: int, denominator: int, history_step: int):
        """Log Eq. (11) CFR = fissions / max(t - |C_t|, 1) with the history step t as counter."""
        logger.debug(f"CFR={cfr:.3f} ({fissions}/{denominator}) at history step {history_step}")

    def log_fcra(self, fcra: float, correct: float, total: int):
        """Log fork-aware CRA metric with turn counter."""
        logger.debug(f"FCRA={fcra:.3f} ({correct:.3f}/{total}) at turn {self.turn_count}")

    def log_acc(self, acc: float, correct: int, total: int):
        """Log ACC metric with question counter."""
        self.question_count += 1
        logger.debug(f"ACC={acc:.3f} ({correct}/{total}) at question {self.question_count}")
