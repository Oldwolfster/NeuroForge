from src.NNA.engine.early_stopping.Signal__BASE import Signal__BASE

class Signal_PerfectAccuracy(Signal__BASE):
    #Superclass handle construction def __init__(self, threshold: float, error_hist: dict[int, float]):

    def evaluate(self) -> bool:
        """Returns True if"""
        if not self.error_hist:            return False
        #print (f"self.TRI.bd_correct{self.TRI.accuracy}")
        if self.TRI.training_data.is_binary_decision and self.TRI.accuracy>=100: return True
        return False # self.is_classic_decay_pattern()
        #latest_epoch = max(self.error_hist.keys())
        return False



    def is_classic_decay_pattern(self) -> bool:
        """
        Returns True if error_hist shows classic exponential decay pattern.
        Pattern = steep slope in first 25% of epochs, flat slope in last 25%.
        """
        min_epochs = 8  # Need enough to split into quarters
        if len(self.error_hist) < min_epochs:
            return False

        epochs = sorted(self.error_hist.keys())
        n = len(epochs)

        # Split into first 25% and last 25%
        first_quarter_end = max(2, n // 4)
        last_quarter_start = n - max(2, n // 4)

        # Calculate average slope for first 25%
        first_slopes = []
        for i in range(1, first_quarter_end):
            improvement = self.error_hist[epochs[i - 1]] - self.error_hist[epochs[i]]
            first_slopes.append(improvement)
        avg_first = sum(first_slopes) / len(first_slopes)

        # Calculate average slope for last 25%
        last_slopes = []
        for i in range(last_quarter_start, n):
            improvement = self.error_hist[epochs[i - 1]] - self.error_hist[epochs[i]]
            last_slopes.append(improvement)
        avg_last = sum(last_slopes) / len(last_slopes)

        #print(f"First 25% avg slope: {avg_first:.6f}")
        #print(f"Last 25% avg slope:  {avg_last:.6f}")
       # print(f"Ratio: {avg_first / avg_last:.1f}x" if avg_last > 0 else "Ratio: infinite")

        # Classic decay = early steep, late flat (big ratio difference)
        ratio = avg_first / avg_last if avg_last > 0.0001 else 999
        is_decay = ratio > 50  # Early was 50x steeper than late

        # print(f"Classic decay pattern: {is_decay}")
        return is_decay