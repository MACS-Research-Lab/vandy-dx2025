class FaultRule:
    """Fault Rule class"""

    def __init__(self, required_count: int = 100, trigger_percent: float = 0.8):
        """
        Initialize the hysteresis rule.

        Args:
            required_count (int): Size of the sliding window.
            trigger_percent (float): Minimum proportion of 1s to trigger fault.
        """
        self.required_count = required_count
        self.trigger_percent = trigger_percent
        self.reset()

    def reset(self):
        """Reset internal state for reuse."""
        self.state = 0
        self.count = 0
        self.window = []
        self.len_w = 0
        self.trigger_count = int(self.required_count * self.trigger_percent)

    def apply(self, value: int) -> int:
        """
        Apply one step of the rule to a new input value.

        Args:
            value (int): New boolean value (0 or 1).

        Returns:
            int: Current state (0 = nominal, 1 = faulty)
        """
        self.window.append(value)
        self.count += value
        self.len_w += 1

        if self.len_w > self.required_count:
            removed = self.window.pop(0)
            self.count -= removed
            self.len_w -= 1

        if self.state == 0 and self.count >= self.trigger_count:
            self.state = 1

        return self.state