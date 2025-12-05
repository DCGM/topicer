import re

def find_exact_span(full_text: str, quote: str, context_before: str) -> tuple[int, int] | None:   
    """
    Finds the start and end indices for a given quote within the full_text.
    The search is resilient to variations in whitespace (spaces, tabs, newlines) using regex
    and resolves duplicates using the provided context_before.
    
    Args:
        full_text (str): The complete text in which to search.
        quote (str): The exact substring to find within the full_text.
        context_before (str): The text immediately preceding the quote to help disambiguate duplicates.
    Returns:
        tuple[int, int] | None: A tuple of (start_index, end_index) if found, else None.
    """
    
    # If quote is empty, return None immediately
    if not quote:
        return None

    # We split the quote into words
    words = quote.split()
    if not words:
        return None
        
    # Each word is escaped to avoid regex special characters interfering and joined with \s+
    # \s+ means: "match a space, tab, newline - anything invisible"
    pattern_str = r'\s+'.join([re.escape(w) for w in words])
    
    #We find all the matches at once using regex
    matches = list(re.finditer(pattern_str, full_text))

    # CASE A: No matches found
    if not matches:
        return None

    # CASE B: One match -> return it directly

    if len(matches) == 1:
        m = matches[0]
        return m.start(), m.end()

    # CASE C: Multiple matches -> decide based on context_before
    # (This part is similar to your original, just adapted for regex match objects)
    best_match_span = None
    best_score = -1
    
    # Normalize context for comparison (normalize is useful here)
    norm_context = ' '.join(context_before.split())

    for m in matches:
        start_idx = m.start()
        end_idx = m.end()

        # We take a piece of text before this occurrence
        # Length of context + 50 characters buffer to have some room to search
        search_window_start = max(0, start_idx - len(context_before) - 50)
        text_before_span = full_text[search_window_start:start_idx]
        
        # Normalize the text extracted from the document
        norm_text_before = ' '.join(text_before_span.split())

        score = 0
        # 1. Try to find the entire context phrase
        if norm_context in norm_text_before:
            score = 100 # Jackpot
        else:
            # 2. If not, calculate word intersection (Jaccard similarity)
            set_ctx = set(norm_context.split())
            set_txt = set(norm_text_before.split())
            intersection = set_ctx & set_txt
            score = len(intersection) # The more common words, the better

        if score > best_score:
            best_score = score
            best_match_span = (start_idx, end_idx)

    # If comparison failed (best_score is 0), simply return the first occurrence
    if best_match_span is None:
        m = matches[0]
        return m.start(), m.end()

    return best_match_span