from __future__ import annotations

import json
import re
import sys
import sqlite3

from . import llm


def select_articles(source_name: str, rows: list[sqlite3.Row], limit: int, prefs: dict) -> tuple[list[str], list[str]]:
    selected_hashes = []
    rejected_hashes = []

    # Get preferences with default fine-tuning weights
    bl_titles = prefs.get("blacklist_titles", [])
    bl_words = prefs.get("blacklist_words", [])
    pref_cats = prefs.get("prefer_category", [])
    less_pref_cats = prefs.get("less_prefer_category", [])
    
    # You can override these in sources.toml under [preferences] if desired
    weight_bl_title = prefs.get("weight_blacklist_title", -15)
    weight_bl_word = prefs.get("weight_blacklist_word", -8)
    weight_pref_cat = prefs.get("weight_prefer_category", 3)
    weight_less_pref = prefs.get("weight_less_prefer_category", -3)

    surviving = []

    # --- Stage 1: Hard Filter ---
    # Only reject articles that are functionally useless (e.g., extraction failed/too short)
    for r in rows:
        if len(r["text"]) < 500:
            rejected_hashes.append(r["url_hash"])
        else:
            surviving.append(r)

    # --- Stage 2: Heuristic Ranking (Soft Blacklist & Weighting) ---
    scored_articles = []
    for r in surviving:
        score = 0
        text = r["text"]
        title = r["title"]
        search_target = (title + " " + r["source"]).lower()
        
        # Positive / Negative Categories
        for cat in pref_cats:
            if cat.lower() in search_target:
                score += weight_pref_cat
        for cat in less_pref_cats:
            if cat.lower() in search_target:
                score += weight_less_pref

        # Soft Blacklist (Titles)
        for bt in bl_titles:
            if bt.lower() in title.lower():
                score += weight_bl_title
        
        # Soft Blacklist (Words in Title or Text)
        for bw in bl_words:
            pattern = r'\b' + re.escape(bw) + r'\b'
            if re.search(pattern, title, re.IGNORECASE) or re.search(pattern, text, re.IGNORECASE):
                score += weight_bl_word
                
        scored_articles.append((score, r))
    
    # Sort descending by heuristic score
    scored_articles.sort(key=lambda x: x[0], reverse=True)
    candidates = [x[1] for x in scored_articles]

    # --- Stage 3: LLM Concierge Downselection ---
    batch_size = 5
    prefer_categories = prefs.get("prefer_category", [])
    ignore_categories = prefs.get("less_prefer_category", [])
    interests = prefs.get("interest", [])
    disinterests = prefs.get("disinterest", [])
    
    # Pre-reject long tails to save tokens, but give the LLM a wider pool to pick from
    max_eval_pool = limit * 4
    top_candidates = candidates[:max_eval_pool]
    rejected_hashes.extend([r["url_hash"] for r in candidates[max_eval_pool:]])

    # Debug: Clearly indicate heuristic decisions
    if top_candidates:
        sys.stderr.write(f"  [debug] Heuristic ranking for top {len(top_candidates)} candidates:\n")
        for score, r in scored_articles[:max_eval_pool]:
            sys.stderr.write(f"    [score: {score:3d}] {r['title'][:70]}\n")

    for i in range(0, len(top_candidates), batch_size):
        if len(selected_hashes) >= limit:
            # Target met! Reject all remaining unevaluated candidates
            rejected_hashes.extend([r["url_hash"] for r in top_candidates[i:]])
            break
            
        batch = top_candidates[i:i + batch_size]
        
        prompt = "Review these articles:\n\n"
        for idx, r in enumerate(batch):
            prompt += f"[{idx}] Title: {r['title']} | Source: {r['source']} | Snippet: {r['text'][:300]}...\n\n"
        
        sys_prompt = (
            "As a concierge editor, evaluate the interest, novelty, and value characteristics "
            "of the following articles given this selection guidance.\n"
            f"Interest Categories: {prefer_categories}\n"
            f"Interest Statements: {interests}\n"
            f"Disinterest Categories: {ignore_categories}\n"
            f"Disinterest Statement: {disinterests}\n"
            "Select the articles from this batch that meet the high-quality threshold. "
            "Return strictly a JSON array of their integer indices (e.g., [0, 2, 4]). "
            "If none meet the threshold, return index of the single best entry.\n"
            "CRITICAL: Do not explain your reasoning. Output ONLY the JSON array."
        )

        try:
            # Increased max_tokens so conversational fluff doesn't truncate the array
            reply = llm.chat(sys_prompt, prompt, max_tokens=300)
            
            # Robustly extract JSON array (ignores markdown blocks and conversational padding)
            match = re.search(r'\[(.*?)\]', reply, re.DOTALL)
            if match:
                try:
                    indices = json.loads("[" + match.group(1) + "]")
                    if not isinstance(indices, list):
                        indices = []
                        
                    for idx, r in enumerate(batch):
                        if idx in indices:
                            selected_hashes.append(r["url_hash"])
                        else:
                            rejected_hashes.append(r["url_hash"])
                except json.JSONDecodeError:
                    sys.stderr.write(f"  [warn] JSON parse failed. Raw LLM output: {reply.strip()}\n")
                    rejected_hashes.extend([r["url_hash"] for r in batch])
            else:
                sys.stderr.write(f"  [warn] No JSON array found. Raw LLM output: {reply.strip()}\n")
                rejected_hashes.extend([r["url_hash"] for r in batch])
                
        except Exception as e:
            sys.stderr.write(f"  [warn] LLM API call failed: {e}\n")
            rejected_hashes.extend([r["url_hash"] for r in batch])
            
    # Absolute failsafe to ensure we don't exceed the requested limit
    if len(selected_hashes) > limit:
        overshoot = selected_hashes[limit:]
        selected_hashes = selected_hashes[:limit]
        rejected_hashes.extend(overshoot)

    return selected_hashes, rejected_hashes