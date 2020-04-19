import string, copy, en_core_web_sm, json


'''
Tokenizer function. Allows for calculating position of each token in larger text.
Input: (String) Input text to be tokenized
Output: (list[dict]) Outputs list of dicts with token and corresponding indices
'''
def tokenize(text, start_idx):
    s = ""
    tokens = []
    byte_list = []
    tokens = []
    current_bytes =[start_idx, start_idx]
    to_return = []
    for idx, i in enumerate(text):
        if i in string.whitespace:
            if s != "":
                tokens.append(s)
                byte_list.append(tuple(current_bytes))
            current_bytes = [current_bytes[1]+1, current_bytes[1] + 1]
            s = ""
        elif i == ">":
            s += str(i)
            current_bytes[1] = current_bytes[1] + 1
            tokens.append(s)
            byte_list.append(tuple(current_bytes))
            s = ""
            current_bytes = [current_bytes[1], current_bytes[1]]
        elif i in string.punctuation and i != "<" and text[idx-1:idx+1] != "</":
            if (s != ""):
                tokens.append(s)
                byte_list.append(tuple(current_bytes))
            current_bytes = [current_bytes[1], current_bytes[1] + 1]
            tokens.append(i)
            byte_list.append(tuple(current_bytes))
            current_bytes = [current_bytes[1], current_bytes[1]]
            s = ""
        elif i == "<":
            if s != "":
                tokens.append(s)
                byte_list.append(tuple(current_bytes))
            s = ""
            current_bytes = [current_bytes[1], current_bytes[1] + 1]
            s += str(i)
        else:
            s += str(i)
            current_bytes[1] = current_bytes[1] + 1
    if s != "":
        tokens.append(s)
        byte_list.append(tuple(current_bytes))
    for idx, token in enumerate(tokens):
        token_dict = {}
        token_dict["token"] = token
        token_dict["start_byte"] = byte_list[idx][0]
        token_dict["end_byte"] = byte_list[idx][1]
        token_dict["html_token"] = (token[0] == "<" and token[-1] == ">")
        to_return.append(token_dict)
    return to_return

def get_long_answer_candidates(tokenized_text):
    candidates = []
    for i, token_dict in enumerate(tokenized_text):
        token = token_dict["token"]
        if token[0] == "<" and token[1] != "/" and token[-1] == ">":
            html_token = token[1:-1]
            for i2, token_dict2 in enumerate(tokenized_text[i:]):
                token2 = token_dict2["token"]
                if token2[0] == "<" and token2[1] == "/" and token2[-1] == ">" and token2[2:-1] == html_token:
                    candidates.append({"start_byte": token_dict["start_byte"], "end_byte": token_dict2["end_byte"]-1, "start_token": i, "end_token": i2+i,
                        "top_level": True})
                    break
    return candidates


with open('sparknotes_dataset.json', 'r') as f:
    book_doc = json.load(f)

nlp = en_core_web_sm.load()
counter = 0
nq_doc = []
for chapter in book_doc["data"]:
    example = {}
    qa_list = chapter["qa_list"]
    for qa in qa_list:
        if counter % 500 == 0:
            print(counter)
        example["example_id"] = counter
        counter = counter + 1
        example["question_text"] = qa["question"]
        example["question_tokens"] = [str(token) for token in nlp(qa["question"])]
        example["document_url"] = chapter["summary_url"]
        example["document_html"] = "<p>" + "</p><p>".join([" ".join(i) for i in chapter["summary"]]) + "</p>"
        example["document_title"] = chapter["title"] + ", " + chapter["chapters_covered"]
        summary = " ".join([" ".join(i) for i in chapter["summary"]])
        doc_tokens = []
        start_idx = 0
        for p in chapter["summary"]:
            if p == []:
                print("p")
                print("chapter url", chapter["summary_url"])
            new_p_token = {"token" : "<p>", "start_byte": start_idx, "end_byte": start_idx + 3, "html_token": True}
            doc_tokens.append(new_p_token)
            start_idx += 3
            p = " ".join(p)
            tokenized_p = tokenize(p, start_idx)
            start_idx = tokenized_p[-1]["end_byte"] + 1
            doc_tokens.extend(tokenized_p)
            new_p_token2 = {"token" : "</p>", "start_byte": start_idx, "end_byte": start_idx + 4, "html_token": True}
            doc_tokens.append(new_p_token2)
            start_idx = start_idx + 4
        example["document_tokens"] = doc_tokens
        example["long_answer_candidates"] = get_long_answer_candidates(doc_tokens)
        example["annotations"] = []
        # print(example)
        nq_doc.append(example)
with open('sparknotes_data_nqformat.json', 'w+') as file:
    json.dump(nq_doc, file)


