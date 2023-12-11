import re
from rouge_score import rouge_scorer, scoring

class MecabTokenizer:
    """Wrapper for MeCab for rouge scoring.

    This is lazy, so it's only initialized the first time it's used."""

    def __init__(self) -> None:
        self.tagger = None

    def _init_tagger(self):
        from fugashi import Tagger

        self.tagger = Tagger("-Owakati")

    def normalize_answer(self, text):
        """Lower case text, remove punctuation and extra whitespace, etc."""
        import emoji
        import neologdn

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_emoji(text):
            text = "".join(["" if emoji.is_emoji(c) else c for c in text])
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
                "]+",
                flags=re.UNICODE,
            )
            return emoji_pattern.sub(r"", text)

        text = remove_emoji(text)
        # see neologdn docs for details, but handles things like full/half width variation
        text = neologdn.normalize(text)
        text = white_space_fix(text)
        return text

    def tokenize(self, text):
        if self.tagger is None:
            self._init_tagger()
        return self.tagger.parse(self.normalize_answer(text)).split()

TOKENIZER = MecabTokenizer()

def rouge2_mecab(refs, preds, tokenizer):
    """This uses a MeCab tokenizer for Japanese text.

    Besides specifying the tokenizer, this does not perform the rougeLsum
    related sentence/newline normalization, and only calculates rouge2.
    Otherwise it is the same as the generic rouge scoring.
    """
    rouge_types = ["rouge2"]
    # mecab-based rouge

    # XXX it may be better to just specify the global TOKENIZER here and remove
    # it from args
    scorer = rouge_scorer.RougeScorer(
        rouge_types,
        tokenizer=tokenizer,
    )

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}


def rouge2_pre(item):
    # this is called with a pair: (gold, pred)
    return item

def rouge2_mecab_metric(items):
    predictions, references = zip(*items)
    res = rouge2_mecab(refs=references, preds=predictions, tokenizer=TOKENIZER)
    return res["rouge2"]

