from fuzzysearch import find_near_matches


class FuzzyMatcher:
    def __init__(self, max_dist_ratio: float = 0.2):
        """        
        :param max_dist_ratio: Maxiimum Levenshtein distance ratio (distance / length of quote) to still consider a match valid.
        :type max_dist_ratio: float
        """
        self.max_dist_ratio = max_dist_ratio

    def _get_best_dist(self, target: str, window: str) -> int | None:
        """ Get the best Levenshtein distance between target and window using fuzzysearch """
        # If the target is empty, distance/penalty is 0
        if not target:
            return 0

        max_dist = int(len(target) * self.max_dist_ratio)
        matches = find_near_matches(target, window, max_l_dist=max_dist)
        return min((m.dist for m in matches), default=None)

    def find_best_span(self, full_text: str, quote: str, context_before: str | None = None, context_after: str | None = None) -> tuple[int, int] | None:
        if not quote:
            return None

        max_l_dist_quote = int(len(quote) * self.max_dist_ratio)
        matches = find_near_matches(
            quote, full_text, max_l_dist=max_l_dist_quote)

        if not matches:
            return None

        best_match_coords = None
        min_total_penalty = float('inf')

        # We analyze before and after contexts for each match to find the best one
        for m in matches:
            # ---- BEFORE CONTEXT ----
            penalty_before = 0
            if context_before:
                search_start = max(0, m.start - len(context_before) - 30)
                window_before = full_text[search_start:m.start]
                dist_before = self._get_best_dist(
                    context_before, window_before)

                penalty_before = dist_before if dist_before is not None else (
                    int(len(context_before) * self.max_dist_ratio) + 1)

            # ---- AFTER CONTEXT ----
            penalty_after = 0
            if context_after:
                search_end = min(len(full_text), m.end +
                                 len(context_after) + 30)
                window_after = full_text[m.end:search_end]
                dist_after = self._get_best_dist(context_after, window_after)

                penalty_after = dist_after if dist_after is not None else (
                    int(len(context_after) * self.max_dist_ratio) + 1)

            # ---- TOTAL PENALTY ----
            total_penalty = m.dist + penalty_before + penalty_after
            print(
                f"Match at ({m.start}, {m.end}) with quote dist {m.dist}, \n"
                f"before dist {penalty_before}, after dist {penalty_after}, total {total_penalty}\n"
                f"Context before: '{context_before}'\n"
                f"Context after: '{context_after}'"
            )

            if total_penalty < min_total_penalty:
                min_total_penalty = total_penalty
                best_match_coords = (m.start, m.end)

        return best_match_coords
