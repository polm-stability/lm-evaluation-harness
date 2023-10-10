"""
WikiLingua: A New Benchmark Dataset for Cross-Lingual Abstractive Summarization
https://aclanthology.org/2020.findings-emnlp.360/

We introduce WikiLingua, a large-scale, multilingual dataset for the evaluation of cross-lingual abstractive summarization systems. We extract article and summary pairs in 18 languages from WikiHow, a high quality, collaborative resource of how-to guides on a diverse set of topics written by human authors. We create gold-standard article-summary alignments across languages by aligning the images that are used to describe each how-to step in an article. As a set of baselines for further studies, we evaluate the performance of existing cross-lingual abstractive summarization methods on our dataset. We further propose a method for direct cross-lingual summarization (i.e., without requiring translation at inference time) by leveraging synthetic data and Neural Machine Translation as a pre-training step. Our method significantly outperforms the baseline approaches, while being more cost efficient during inference.

Homepage: https://github.com/esdurmus/Wikilingua
"""
import os
import numpy as np
import datasets
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
from lm_eval.utils import rouge2_mecab


_CITATION = """
@inproceedings{ladhak-etal-2020-wikilingua, title = "{W}iki{L}ingua: A New Benchmark Dataset for Cross-Lingual Abstractive Summarization", author = "Ladhak, Faisal and Durmus, Esin and Cardie, Claire and McKeown, Kathleen", booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020", month = nov, year = "2020", address = "Online", publisher = "Association for Computational Linguistics", url = "https://aclanthology.org/2020.findings-emnlp.360", doi = "10.18653/v1/2020.findings-emnlp.360", pages = "4034--4048", abstract = "We introduce WikiLingua, a large-scale, multilingual dataset for the evaluation of cross-lingual abstractive summarization systems. We extract article and summary pairs in 18 languages from WikiHow, a high quality, collaborative resource of how-to guides on a diverse set of topics written by human authors. We create gold-standard article-summary alignments across languages by aligning the images that are used to describe each how-to step in an article. As a set of baselines for further studies, we evaluate the performance of existing cross-lingual abstractive summarization methods on our dataset. We further propose a method for direct cross-lingual summarization (i.e., without requiring translation at inference time) by leveraging synthetic data and Neural Machine Translation as a pre-training step. Our method significantly outperforms the baseline approaches, while being more cost efficient during inference.", }
"""


# TODO make a summarization task
class Wikilingua(Task):
    VERSION = 1.0
    # custom prompt
    PROMPT_VERSION = 0.0
    DATASET_PATH = "GEM/wiki_lingua"
    DATASET_NAME = "ja"
    DESCRIPTION = "与えられた文章を要約して下さい。\n\n"
    LOAD_TOKENIZER = True

    def __init__(self):
        super().__init__()
        from . import MecabTokenizer

        self.tokenizer = MecabTokenizer()

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def training_docs(self):
        return self.dataset["train"]

    def doc_to_text(self, doc):
        return doc["source"]

    def doc_to_target(self, doc):
        target = doc["target"]

        # XXX: consider fixing weird formatting. In the targets it seems
        # inconsistent whether sentences are separated with "。 " or "\u3000 "
        # (\u3000 = full width space)

        # target = doc["target"].replace(" \u3000", "\u3000").replace("\u3000 ", "。")
        return target

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        completion = rf.greedy_until(ctx, ["\n"])
        return completion

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        completion = results[0].strip()

        ref = doc["source"]

        return {"rouge2": (completion, ref)}

    def _rouge(self, item):
        predictions, references = zip(*item)
        res = rouge2_mecab(refs=references, preds=predictions, tokenizer=self.tokenizer)
        return res["rouge2"]

    def aggregation(self):
        return {
            "rouge2": self._rouge,
        }

    def higher_is_better(self):
        return {
            "rouge2": True,
        }


class WikilinguaWithJAAlpacaPrompt(Wikilingua):
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = "与えられたニュース記事を要約してください。"

    def doc_to_text(self, doc):
        """
        以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

        ### 指示:
        {instruction}

        ### 入力:
        {input}

        ### 応答:
        {response}
        """
        input_text = f"ニュース記事:{doc['text']}"
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"


class WikilinguaWithRinnaInstructionSFT(Wikilingua):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """

    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: 与えられたニュース記事を要約してください。<NL>システム: 分かりました。<NL>"
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"

    def doc_to_text(self, doc):
        input_text = f"ニュース記事:{doc['text']}"
        return f"ユーザー: {input_text}{self.SEP}システム: "

    def preprocess_ctx(self, ctx, max_length):
        return super().preprocess_ctx(
            ctx,
            max_length,
            ctx_prompt=f"{self.SEP}ユーザー: ",
            summary_prompt=f"{self.SEP}システム: ",
        )


class WikilinguaWithRinnaBilingualInstructionSFT(WikilinguaWithRinnaInstructionSFT):
    PROMPT_VERSION = 0.5
    DESCRIPTION = "ユーザー: 与えられたニュース記事を要約してください。\nシステム: 分かりました。\n"
    SEP = "\n"
    FEWSHOT_SEP = "\n"


class WikilinguaWithLlama2(Wikilingua):
    """
    This prompt version follows the Llama2-chat's prompt format:
    ```
    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
    ```
    reference: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """

    PROMPT_VERSION = 0.6
    DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    DESCRIPTION = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
    FEWSHOT_SEP = " </s><s>[INST] "

    def doc_to_text(self, doc):
        """
        Insert the following prompt into `{{ user_msg }}`, which is based on prompt version 0.3
        ```
        与えられたニュース記事を要約してください。

        ニュース記事:{doc} [/INST]
        ```
        """
        input_text = f"ニュース記事:{doc['text']}"
        return f"{self.INSTRUCTION}\n\n{input_text} [/INST] "


VERSIONS = [
    Wikilingua,
    WikilinguaWithJAAlpacaPrompt,
    WikilinguaWithRinnaInstructionSFT,
    WikilinguaWithRinnaBilingualInstructionSFT,
    WikilinguaWithLlama2,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[
            f"wikilingua_ja-{version_class.VERSION}-{version_class.PROMPT_VERSION}"
        ] = version_class
    return tasks
