import datasets

def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
      
        doc["gold"] = doc[""][0]
        return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object

