"""
JGLUE: Japanese General Language Understanding Evaluation
https://aclanthology.org/2022.lrec-1.317/

JGLUE, Japanese General Language Understanding Evaluation, is built to measure the general NLU ability in Japanese. 
JGLUE has been constructed from scratch without translation. 

Homepage: https://github.com/yahoojapan/JGLUE
"""
from lm_eval.base import MultipleChoiceTask, rf
import numpy as np


_CITATION = """
@inproceedings{kurihara-etal-2022-jglue,
    title = "{JGLUE}: {J}apanese General Language Understanding Evaluation",
    author = "Kurihara, Kentaro  and
      Kawahara, Daisuke  and
      Shibata, Tomohide",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.317",
    pages = "2957--2966",
    abstract = "To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}
"""



class JCommonsenseQA(MultipleChoiceTask):
    """
    prompt format is taken from [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
    """
    VERSION = 1.1
    PROMPT_VERSION = 0.1
    DATASET_PATH = "shunk031/JGLUE"
    DATASET_NAME = "JCommonsenseQA"
    DESCRIPTION = "[問題]に対する[答え]を[選択肢]の中から選んでください。\n\n"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        return {
            "goal": doc["question"],
            "choices": [doc[f"choice{i}"] for i in range(5)],
            "gold": doc["label"], 
        }

    def doc_to_text(self, doc):
        """
        [問題]:question
        [選択肢]:[choice0, choice1, ..., choice4]
        [答え]:
        """
        return (
            f"[問題]:{doc['goal']}\n"
            f"[選択肢]:[{', '.join(doc['choices'])}]\n"
            "[答え]:"
        )
    
    def doc_to_target(self, doc):
        return doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, "{}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls

    def process_results(self, doc, results):
        gold = doc["gold"]

        response = np.argmax(results)
        acc = 1.0 if response == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        out = {
            "acc": acc,
            "acc_norm": acc_norm,
        }
        # only include details if we were wrong
        if acc == 0.0:
            # without the cast it won't serialize
            response = int(response)
            out["details"] = {
                "question": doc["goal"],
                "choices": doc["choices"],
                "gold": doc["gold"],
                "response": response,
            }
        return out

class JCommonsenseQAWithFintanPrompt(JCommonsenseQA):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """
    VERSION = 1.1
    PROMPT_VERSION = 0.2
    DESCRIPTION = "質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。なお、回答は選択肢の番号(例:0)でするものとします。 \n\n"


    def doc_to_text(self, doc):
        """
        質問:question
        選択肢:0.choice0,1.choice1, ...,4.choice4
        回答:
        """
        choices = ",".join([f"{idx}.{choice}" for idx, choice in enumerate(doc['choices'])])
        return (
            f"質問:{doc['goal']}\n"
            f"選択肢:{choices}\n"
            "回答:"
        )
    
    def doc_to_target(self, doc):
        return f"{doc['gold']}"
    
class JCommonsenseQAWithJAAlpacaPrompt(JCommonsenseQA):
    """
    This prompt format was inspired by the below data in fujiki/japanese_alpaca_data. 
    ```
    {
        'instruction': 'この課題では、以下の選択肢から文の出典を特定する必要があります。\n\n出力は以下から選択してください：\n- 新聞\n- 教科書\n- オンライン記事\n- 百科事典', 
        'input': '彼はローマの政治家であり哲学者であり、史上最も偉大な軍事指導者の一人と考えられています。', 
        'output': '百科事典'
    }
    ```
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """
    VERSION = 1.1
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = "与えられた選択肢の中から、最適な答えを選んでください。\n\n"
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
        choices = "\n".join([f"- {choice}" for choice in doc['choices']])
        instruction_text = self.INSTRUCTION + f"出力は以下から選択してください：\n{choices}"
        input_text = f"{doc['goal']}"
        return f"### 指示:\n{instruction_text}\n\n### 入力:\n{input_text}\n\n### 応答:\n"
        


class JCommonsenseQAWithRinnaInstructionSFT(JCommonsenseQA):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    VERSION = 1.1
    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: 与えられた選択肢の中から、最適な答えを選んでください。<NL>システム: 分かりました。"
    def doc_to_text(self, doc):
        choices = "<NL>".join([f"- {choice}" for choice in doc['choices']])
        input_text = f"質問：{doc['goal']}<NL>" + f"選択肢：<NL>{choices}"
        return f"<NL>ユーザー: {input_text}<NL>システム: "
        


VERSIONS = [
    JCommonsenseQA,
    JCommonsenseQAWithFintanPrompt,
    JCommonsenseQAWithJAAlpacaPrompt,
    JCommonsenseQAWithRinnaInstructionSFT,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"jcommonsenseqa-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks
