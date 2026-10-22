"""Wall-clock scheduling, not pose QC. Duty targets worker occupancy, not GPU utilization."""
from dataclasses import dataclass
import math

VIEW_REFRESH_REASONS = frozenset(('view_direction_changed', 'new_regions_visible',
                                  'view_direction_approaching', 'view_direction_emerging'))


@dataclass
class BackgroundRefresh:
    cooldown: float = 1.
    duty: float = .35
    track_budget_ms: float = 100.
    last_completed: float = -math.inf
    last_global: float = -math.inf
    search_ms: float = 0.
    tracking_ms: float = 0.
    failures: int = 0
    added_total: int = 0
    last_added: int = 0
    outcome: str = 'not_started'
    reason: str = 'initial'
    state: str = 'waiting'
    fast_recovery: bool = False

    def track_sample(self, milliseconds):
        # Fast attack, slow recovery protects an overloaded foreground loop.
        weight = .5 if milliseconds > self.tracking_ms else .1
        self.tracking_ms += weight * (milliseconds-self.tracking_ms)

    def submitted(self, mode, now):
        self.state = 'running'
        if mode == 'global':
            self.last_global = now

    def completed(self, now, milliseconds):
        self.last_completed = now
        self.search_ms = max(0., milliseconds)
        self.state = 'cooldown'

    def accepted(self, added=0):
        self.failures = 0
        self.last_added = added
        self.added_total += added
        self.outcome = 'added' if added else 'verified'

    def rejected(self, reason):
        self.failures += 1
        self.last_added = 0
        self.outcome = reason

    def decide(self, now, *, busy, weak_reasons, safety_due=False):
        if busy:
            self.state = 'running'
            return None
        new_view = bool(VIEW_REFRESH_REASONS.intersection(weak_reasons))
        urgent = any(reason not in VIEW_REFRESH_REASONS for reason in weak_reasons)
        if self.tracking_ms > self.track_budget_ms and not urgent:
            self.state = 'foreground_busy'
            return None
        idle = max(self.cooldown, self.search_ms/1000 * (1-self.duty)/self.duty)
        idle *= min(4, 1+self.failures)
        if urgent:
            # Face handoff must start BEFORE loss, not after a multi-second
            # healthy occupancy wait/backoff. Still one job and bounded retries.
            retry_limit = 2. if 'tracking_lost' in weak_reasons else 1.
            idle = max(self.cooldown, min(retry_limit, idle))
            if 'support_critical' in weak_reasons:
                # Measured loss predicts that support will vanish before the
                # next result. Still one worker; avoid the extra one-second gap.
                idle = min(idle, .2)
            if self.fast_recovery and 'tracking_lost' in weak_reasons:
                idle = min(idle, .3)
        if new_view:
            # Do this after weak-support timing: coverage warnings must not
            # lengthen the early-face delay. Failures retain a bounded gap.
            idle = min(idle, 1. if self.failures else .2)
        self.reason = (','.join(weak_reasons) if weak_reasons else
                       'safety_review' if safety_due else 'continuous_refresh')
        if now-self.last_completed < idle:
            self.state = 'cooldown'
            return None
        if (safety_due or (urgent and self.failures >= 3)) and now-self.last_global >= 10.:
            self.reason = 'safety_review' if safety_due else 'near_search_failed'
            self.state = 'ready'
            return 'global'
        self.state = 'ready'
        return 'guided'

    def snapshot(self):
        return dict(state=self.state, reason=self.reason, outcome=self.outcome,
                    tracking_ms=self.tracking_ms, search_ms=self.search_ms,
                    failures=self.failures, last_added=self.last_added,
                    added_total=self.added_total, duty_target=self.duty)
