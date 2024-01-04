from inspect import cleandoc

def basic_prompt(doc):
    # This is the 0.1 prompt
    choices = ', '.join([doc[f"choice{ii}"] for ii in range(5)])
    prompt = f"""
        [問題]に対する[答え]を[選択肢]の中から選んでください。

        [問題]:{doc['question']}
        [選択肢]:{choices}
        [答え]:"""
    return cleandoc(prompt)

def alpaca_prompt(doc):
    """
    以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

    ### 指示:
    {instruction}

    ### 入力:
    {input}

    ### 応答:
    {response}
    """

    instruction = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    choices = [doc[f"choice{ii}"] for ii in range(5)]
    choices = "\n".join([f"- {choice}" for choice in choices])
    instruction_text = instruction + "出力は以下から選択してください：\n" + choices
    question = doc["question"]
    return f"### 指示:\n{instruction_text}\n\n### 入力:\n{question}\n\n### 応答:\n"
