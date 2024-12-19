dataset_to_system_prompt = {
    "clerc": "You are a helpful legal professional.",
    "echr_qa": "You are an ECHR legal expert tasked to answer a question.",
    "cuad": (
        "You are a helpful legal professional analyzing clauses in a contract. "
        "Your goal is to identify specific clauses that best suit the task. "
        "Your answer must start with 'Highlights:\\n' and then list the exact contract text segments that "
        "are relevant, and nothing else. "
        "If no relevant text is found, respond with 'No Highlights'."
    ),
    "obli_qa": (
        "You are a regulatory compliance and legal expert reviewing regulatory documents. "
        "Do not directly answer the user's question or provide any interpretation. "
        "Instead, only provide exact excerpts from the regulatory text that are relevant to the question, "
        "verbatim and without additional commentary."
    ),
}

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
    "cuad": (
        "Below are contract excerpts. Identify and highlight the clauses related to the given category. "
        "If a clause is relevant, include it under 'Highlights:' as it appears. "
        "If no clauses match, write 'No Highlights'."
    ),
    "obli_qa": (
        "Below are regulatory documents. Extract only the portions of text that are relevant to the user's question. "
        "Do not answer or explain, just quote relevant text directly."
    ),
}

dataset_to_prompt_prefix = {
    "clerc": (
        "Continue to write the following case in the style of my writeup. Your answer should range "
        "from 100 to 400 words. Make your answer concise, and avoid redundant languages and assumptions. "
        "Below is what I have written so far:"
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
    "cuad": (
        "Do not provide any commentary or explanations. "
        "Only output 'Highlights:\\n' followed by the exact text segments or 'No Highlights'."
    ),
    "obli_qa": (
        "Question: "
        "Extract only the directly relevant regulatory text segments related to the user's query. "
        "Do not answer or provide explanations, only produce the exact text segments that apply."
    ),
}

dataset_to_prompt_suffix = {
    "clerc": "",
    "echr_qa": "\nAnswer: ",
    "cuad": "",
    "obli_qa": ""
}