from datasets import Dataset

def process_docs(dataset: Dataset):
    def _helper(doc):

        # There are always two sentences, and which is correct is
        # indicated with an index in a separate key. Here we'll give
        # the sentences the keys "gold" and "distractor". 

        # note this key is 1-index, not 0-index
        akey = int(doc["answer"])
        doc["gold"] = doc[f"sentence{akey}"]
        dkey = 1 if akey == 2 else 2
        doc["distractor"] = doc[f"sentence{dkey}"]

        return doc
    return dataset.map(_helper)
