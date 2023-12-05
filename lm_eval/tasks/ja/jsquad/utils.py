import datasets

def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
        # there can be more than one gold answer - for now just use the first
        # TODO figure out how to use multiple answers
      
        doc["gold"] = doc["text"][0]
        return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object

