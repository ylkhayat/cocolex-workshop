dataset_to_system_prompt = {
    "clerc": "You are a helpful legal professional.",
    "echr_qa": "You are an ECHR legal expert tasked to answer a question.",
}
        # must pass joined_retrieved_ids
dataset_to_context_prefix = {
    "clerc": (
        "Below are reference cases provided for factual accuracy. When generating content, you must "
        "reference and cross-check the relevant details with the provided reference texts by their "
        "reference IDs. (e.g., {joined_retrieved_ids}). Your output must align with these references."
    ),
    "echr_qa": (
        "The following documents were retrieved and should help you answer the question. "
        "You must refer to these documents when answering the question. (e.g., {joined_retrieved_ids}). "
        "Valid citation formats: [{single_retrieved_id}] or [{joined_retrieved_ids}]. "
    ),
}
dataset_to_prompt_prefix = {
    "clerc": (
        'Continue to write the following case in the style of my writeup. Your answer should range '
        'from 100 to 400 words. Make your answer concise, and avoid redundant languages and assumptions. '
        'Below is what I have written so far:'
    ),
    "echr_qa": (
        "Answer the following question using the retrieved documents. "
        "Reuse the language from the documents! "
        "Cite relevant documents at the end of a sentence! "
        "Accepted formats: sentence [citation(s)]. "
        "You must follow the [Doc i] format! Do NOT use the case names or paragraph numbers to cite documents! "
        "You should NOT provide a list of all used citations at the end of your response!\n\n"
        "Question: "
    ),
}
dataset_to_prompt_suffix = {
    "clerc": "",
    "echr_qa": "\nAnswer: ",
}