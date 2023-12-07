import datasets
from datasets import Dataset

# this uses the special squad eval metric
from lm_eval.tasks.ja.jsquad.eval import jasquad
jasquad_metric = datasets.load_metric(jasquad.__file__)

def process_docs(dataset: Dataset):
    # small helper to make answer accessible
    def _helper(doc):
        doc["context"] = "\n".join([text for text in doc["ctxs"]["text"]])
        doc["answer"] = doc["answers"]["text"][0]
        return doc
    return dataset.map(_helper)

def _squad_metric(predictions, references):
    return jasqaud_metric.compute(
        predictions=predictions, references=references
    )

def _squad_agg(key, predictions, references):
    return _squad_metric(predictions=predictions, references=references)[key]

# In the old code these were created with `partial`, but that won't
# work in yaml
def f1_agg(predictions, references):
    return _squad_agg("f1", predictions, references)

def exact_match_agg(predictions, references):
    return _squad_agg("exact_match", predictions, references)
