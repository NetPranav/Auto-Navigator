"""
Magnum On-The-Fly Local Command & Task Intent Detector.

Replaces the rigid wake-word requirement with a fast, local, zero-overhead
classifier that determines if a spoken utterance is an actionable command/task
(e.g., "WhatsApp message ko reply", "Safari kholo", "reply to Vishesh", "open Chrome")
versus casual background chatter (e.g., "I was talking to him", "the weather is nice").

Trained on-the-fly in ~5ms at startup with pure Python and NumPy.
Zero network latency (<1ms inference time).
"""

from __future__ import annotations

import logging
import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Action verbs across English, Hindi, and Hinglish
ACTION_VERBS: Set[str] = {
    # English
    "open", "close", "reply", "send", "check", "search", "find", "click",
    "type", "write", "play", "pause", "resume", "stop", "cancel", "halt",
    "scroll", "turn", "switch", "delete", "remove", "refresh", "reload",
    "show", "look", "inspect", "create", "make", "start", "run", "launch",
    "watch", "monitor", "listen", "translate", "summarize", "help",
    # Hindi / Hinglish
    "kholo", "khol", "band", "chalao", "chala", "bhejo", "bhej", "dekho",
    "dekh", "dhundo", "khojo", "karo", "kardo", "kar", "hatao", "likho",
    "likh", "banao", "sunao", "batao", "ruko", "ruk", "padho", "scan",
}

# Target applications and OS nouns
TARGET_NOUNS: Set[str] = {
    "whatsapp", "chrome", "safari", "youtube", "terminal", "finder",
    "notes", "slack", "mail", "spotify", "google", "bing", "discord",
    "message", "messages", "chat", "msg", "email", "tab", "tabs",
    "window", "windows", "screen", "display", "desktop", "file", "files",
    "folder", "downloads", "documents", "button", "link", "url", "video",
    "song", "music", "status", "watcher", "watchers", "workflow", "run",
}

# Starters that indicate an imperative or polite request to the assistant
COMMAND_STARTERS: Tuple[str, ...] = (
    "can you", "could you", "please", "i want you to", "i need you to",
    "help me", "go ahead and", "would you", "let me see", "zara",
    "ek second", "sun", "suno", "bhai", "kya tum",
)

# Training Corpus: Positive Command / Task Samples
POSITIVE_COMMAND_SAMPLES: List[str] = [
    # Direct app & task commands
    "open Safari", "open Chrome", "open YouTube", "open Terminal", "open Notes",
    "close this window", "close the tab", "close Chrome", "close Safari",
    "reply to Vishesh", "reply to WhatsApp message", "send message to Vishesh",
    "WhatsApp message ko reply", "WhatsApp message ko reply karo",
    "Vishesh ko reply karo", "Vishesh ko message bhejo", "message send karo",
    "Safari kholo", "Chrome band karo", "YouTube pe lofi song chalao",
    "lofi music play karo", "play relaxing music on YouTube", "pause music",
    "search for AI news on Google", "Google pe search karo", "search for weather",
    "scroll down and click submit", "click the login button", "click continue",
    "type hello world in terminal", "run the test suite", "run pytest",
    "check my downloads folder", "downloads folder check karo", "check email",
    "delete the last file", "refresh the page", "reload this tab",
    "look at my screen and tell me what is wrong", "screen dekho kya chal raha hai",
    "take a screenshot", "inspect the active window", "what is on my screen",
    "create a new workflow", "start a watcher", "stop all watchers",
    "stop watching", "cancel watchers", "cancel all", "show status",
    "latest run dikhao", "show latest run", "what did you plan",
    "ek second ruko", "stop right now", "wait a minute", "halt",
    # Natural variations with polite starters
    "can you open WhatsApp for me", "please check my unread messages",
    "could you reply to this message", "i want you to open Spotify",
    "help me search for this code", "zara screen dekhna",
    "kya tum downloads folder open kar sakte ho", "bhai vishesh ko message kardo",
]

# Training Corpus: Negative Casual Chatter / Non-Task Samples
NEGATIVE_CHATTER_SAMPLES: List[str] = [
    "I was talking to him yesterday about the party",
    "the weather in Mumbai is really hot today",
    "haha that was so funny", "lol that is hilarious",
    "yeah I definitely think so too",
    "I had delicious pizza for lunch earlier",
    "my dog is sleeping on the living room couch",
    "this movie was quite interesting actually",
    "she told me she might come over tomorrow afternoon",
    "I am thinking about going on vacation to Spain",
    "so basically what happened next was totally crazy",
    "yes indeed that is true", "well that sounds pretty cool man",
    "I like the color blue much more than green",
    "it is almost midnight already time flies",
    "we should probably eat dinner soon",
    "honestly I have no idea what they were doing",
    "my phone battery is almost dead",
    "yesterday I went to the supermarket and bought groceries",
    "the train was delayed by twenty minutes",
    "I think Arsenal played really well last night",
    "she was talking about her new car",
    "my sister is graduating next month",
    "I forgot where I put my keys",
    "that was a very long meeting today",
]


class CommandIntentClassifier:
    """
    On-the-fly trained local classifier for detecting actionable commands & tasks.
    Combines Naive Bayes statistical n-gram probabilities with imperative grammar heuristics.
    """

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.pos_word_counts: Counter = Counter()
        self.neg_word_counts: Counter = Counter()
        self.total_pos_words: int = 0
        self.total_neg_words: int = 0
        self.prior_pos: float = 0.5
        self.prior_neg: float = 0.5
        self._train()

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into unigrams and bigrams."""
        clean = text.lower().translate(str.maketrans("", "", string.punctuation)).strip()
        words = clean.split()
        if not words:
            return []
        # Unigrams
        tokens = list(words)
        # Bigrams
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def _train(self) -> None:
        """Train the Naive Bayes model in-memory (<5ms)."""
        pos_docs = [self._tokenize(s) for s in POSITIVE_COMMAND_SAMPLES]
        neg_docs = [self._tokenize(s) for s in NEGATIVE_CHATTER_SAMPLES]

        for doc in pos_docs:
            for token in doc:
                self.pos_word_counts[token] += 1
                self.total_pos_words += 1
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        for doc in neg_docs:
            for token in doc:
                self.neg_word_counts[token] += 1
                self.total_neg_words += 1
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        n_pos = len(pos_docs)
        n_neg = len(neg_docs)
        total = n_pos + n_neg
        self.prior_pos = n_pos / total
        self.prior_neg = n_neg / total

        logger.debug(f"CommandIntentClassifier initialized. Vocab size: {len(self.vocab)}")

    def _score_naive_bayes(self, tokens: List[str]) -> float:
        """Calculate the posterior probability that tokens represent a command."""
        if not tokens:
            return 0.0

        vocab_size = len(self.vocab) + 1  # Laplace smoothing
        log_prob_pos = math.log(self.prior_pos)
        log_prob_neg = math.log(self.prior_neg)

        for token in tokens:
            # P(token | POS) with Laplace smoothing
            pos_count = self.pos_word_counts.get(token, 0)
            p_pos = (pos_count + 1.0) / (self.total_pos_words + vocab_size)
            log_prob_pos += math.log(p_pos)

            # P(token | NEG) with Laplace smoothing
            neg_count = self.neg_word_counts.get(token, 0)
            p_neg = (neg_count + 1.0) / (self.total_neg_words + vocab_size)
            log_prob_neg += math.log(p_neg)

        # Softmax normalization
        max_log = max(log_prob_pos, log_prob_neg)
        exp_pos = math.exp(log_prob_pos - max_log)
        exp_neg = math.exp(log_prob_neg - max_log)
        prob_pos = exp_pos / (exp_pos + exp_neg)
        return prob_pos

    def is_actionable_command(self, text: str) -> Tuple[bool, float, str]:
        """
        Evaluate whether an utterance is an actionable command / task.
        
        Returns:
            (is_command: bool, confidence: float, reason: str)
        """
        if not text or not text.strip():
            return False, 0.0, "empty_text"

        clean = text.lower().strip()
        words = clean.translate(str.maketrans("", "", string.punctuation)).split()
        if not words:
            return False, 0.0, "no_words"

        # 1. Direct imperative / Action Verb & Target Noun heuristic checks
        has_action_verb = any(w in ACTION_VERBS for w in words)
        has_target_noun = any(w in TARGET_NOUNS for w in words)
        starts_with_action = words[0] in ACTION_VERBS
        has_polite_starter = any(clean.startswith(prefix) for prefix in COMMAND_STARTERS)

        # Strong immediate qualification:
        # e.g., "WhatsApp message ko reply" -> has_action_verb (reply), has_target_noun (whatsapp, message)
        if has_action_verb and has_target_noun:
            return True, 0.98, "action_verb_and_target"

        if starts_with_action and len(words) >= 2:
            return True, 0.95, "starts_with_action_verb"

        if has_polite_starter and (has_action_verb or has_target_noun):
            return True, 0.92, "command_starter_with_action"

        # 2. Statistical Naive Bayes N-gram Evaluation
        tokens = self._tokenize(clean)
        prob_pos = self._score_naive_bayes(tokens)

        # Apply domain heuristic boosts
        boost = 0.0
        if has_action_verb:
            boost += 0.25
        if has_target_noun:
            boost += 0.20
        if has_polite_starter:
            boost += 0.20

        final_score = min(1.0, prob_pos * 0.5 + boost * 0.5)

        # Threshold decision
        is_cmd = final_score >= 0.50
        reason = "statistical_intent_match" if is_cmd else "casual_chatter"
        return is_cmd, final_score, reason


# Global singleton instance
_intent_classifier: Optional[CommandIntentClassifier] = None


def get_command_classifier() -> CommandIntentClassifier:
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = CommandIntentClassifier()
    return _intent_classifier
