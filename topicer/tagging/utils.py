def find_exact_span(full_text: str, quote: str, context_before: str) -> tuple[int, int] | None:
        """
        Najde start a end indexy pro quote. Pokud je v textu vícekrát,
        použije context_before k určení správného výskytu.
        """
        
        # Normalizace pro vyhledávání (ignorujeme rozdíly v typech mezer)
        def normalize(s):
            return ' '.join(s.split())

        norm_text = normalize(full_text)
        norm_quote = normalize(quote)
        norm_context = normalize(context_before)

        # Pokud text vůbec nesedí (LLM si vymýšlela), vrátíme None
        if norm_quote not in norm_text:
            return None

        # Najdeme všechny výskyty 'quote' v textu
        # Používáme re.finditer na originálním textu s trochou volnosti pro whitespace
        # (Pro jednoduchost zde použijeme prosté hledání stringů, což stačí v 99% případů,
        # pokud LLM skutečně kopíruje text).
        
        start_indices = []
        start_search = 0
        while True:
            idx = full_text.find(quote, start_search)
            if idx == -1:
                break
            start_indices.append(idx)
            start_search = idx + 1
        
        # SCÉNÁŘ A: Quote se v textu vůbec nenašel přesně (LLM změnila formátování)
        if not start_indices:
            # Záchranná brzda: zkusíme najít alespoň normalizovanou verzi
            # (implementace by byla složitější, pro teď vrátíme None a přeskočíme)
            return None

        # SCÉNÁŘ B: Quote je tam jen jednou -> máme vyhráno
        if len(start_indices) == 1:
            return start_indices[0], start_indices[0] + len(quote)

        # SCÉNÁŘ C: Quote je tam vícekrát (Duplicita) -> použijeme kontext
        best_match_idx = -1
        best_score = 0

        for idx in start_indices:
            # Vezmeme kus textu před tímto výskytem
            # Délka kontextu + malá rezerva
            chunk_before_start = max(0, idx - len(context_before) - 20) 
            text_before_span = full_text[chunk_before_start:idx]
            
            # Spočítáme shodu (jednoduchá heuristika: kolik slov z kontextu sedí)
            # Čím více slov z context_before najdeme v text_before_span, tím lépe.
            score = 0
            norm_text_before = normalize(text_before_span)
            
            # Hledáme, zda se normalizovaný kontext vyskytuje těsně před
            if norm_context in norm_text_before:
                score = 100 # Perfektní shoda
            else:
                # Fallback: částečná shoda (pokud LLM zkrátilo kontext)
                common_words = set(norm_context.split()) & set(norm_text_before.split())
                score = len(common_words)
            
            if score > best_score:
                best_score = score
                best_match_idx = idx
                
        # Pokud se kontext nepodařilo namapovat, vezmeme první výskyt (better than nothing)
        if best_match_idx == -1:
            best_match_idx = start_indices[0]

        return best_match_idx, best_match_idx + len(quote)