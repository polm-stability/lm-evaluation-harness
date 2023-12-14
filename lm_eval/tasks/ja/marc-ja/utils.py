from inspect import cleandoc

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
    choices = ["ポジティブ", "ネガティブ"]

    prompt = f"""
      以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

      ### 指示:
      以下のテキストを、ポジティブまたはネガティブの感情クラスのいずれかに分類してください。

      ### 入力:
      {doc["sentence"]}

      ### 応答:"""
    # we want the last newline
    prompt = cleandoc(prompt) + "\n"
    return prompt



