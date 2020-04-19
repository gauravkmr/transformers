import codecs, copy, torch, csv, json
from bs4 import BeautifulSoup
from spacy.lang.en import English
import numpy as np

'''
Processes long question sparknotes page and converts into Google NQ input format
Input: (int) Example number
Output: (dict) processed example
'''

def get_url(example_num):
    document_url = ""
    with open("%d_url" % example_num) as f:
        doc = f.readlines()
    return doc[0]

def get_bert_embedding(model, summary):
    to_return = []
    for i, paragraph in enumerate(summary):
        to_return.append([])
        for sentence in paragraph:
            marked_text = "[CLS] " + sentence + " [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensors)
            token_embeddings = torch.stack(encoded_layers, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings[8:, :, :] # Take mean over last 4 embeddings
            token_embeddings = torch.sum(token_embeddings, dim=0)
            token_embeddings = torch.mean(token_embeddings, dim=0)
            to_return[i].append(token_embeddings.numpy()) # convert sentence embedding form Torch tensor to numpy array
    return to_return

def get_cosine_similarity(a1, a2):
    return cosine_similarity(a1.reshape(1, -1), a2.reshape(1, -1))


def process_longq(example_num, document_url):
    f = codecs.open("%d_content" % example_num, 'r', 'utf-8')
    soup = BeautifulSoup(f.read(), 'html.parser')
    myh3 = soup.findAll("h3", {"class": "heading__fiveKeyQs__question"})
    question = myh3[0].text.strip()
    # question_tokens, _ = tokenize(question)
    myp = soup.findAll("p")
    answer = myp[0].text
    # tokens, _ = tokenize(answer)

    # document_tokens = [{"token": tokens[i], "start_byte": 0, "end_byte": 0, "html_token": false} for i in range(len(tokens))]
    title_html = soup.findAll("a", {"id": "tag--interior-title-link"})[0]
    title = title_html.text.strip()
    author_html = soup.findAll("a", {"id": "interiorpage_author_link1"})[0]
    author = author_html.text.strip()
    document = {"document_url": document_url, "document_html": soup.get_text(), "question": question, "answer": answer, "author": author,
    "title": title}

    return document

def process_shortqs(example_num, document_url):
    # print(example_num)
    f = codecs.open("%d_content" % example_num, 'r', 'utf-8')
    soup = BeautifulSoup(f.read(), 'html.parser')
    quiz_questions = [soup.findAll("div", {"class" : "quick-quiz-question"})][0]
    questions = []
    for q_question in quiz_questions:
        curr_answers = []
        true_answer_idx = 1000
        question = ""
        children = q_question.findChildren("li", recursive=True)
        for i, child in enumerate(children):
            if len(child['class']) == 2: # Incorrect answer doesn't contain true answer class, has len 2
                curr_answers.append(child.text.strip())
            else:
                curr_answers.append(child.text.strip())
                true_answer_idx = i
        all_text = q_question.findChildren("h3", recursive=False)[0]
        question_number = all_text.findChildren("div", recursive=False)[0].text
        question = all_text.text.replace(question_number, '').strip()
        questions.append((question, curr_answers, true_answer_idx))

    title = soup.findAll("h1", {"class": "TitleHeader_title"})[0].text.strip()
    maybe_author = soup.findAll("a", {"class": "TitleHeader_authorLink"})
    if len(maybe_author) > 0: # author is link
        author = maybe_author[0].text.strip()
    else: # author is plaintext
        author = soup.findAll("div", {"class": "TitleHeader_authorName"})[0].text.strip()
    chapters = soup.findAll("h2", {"class": "interior__page__title"})[0].text.strip()
    # documents = {"qa_list": [], "document_html": soup.get_text(), "document_url": document_url, "author": author,  "title": title,
    # "chapters_covered": chapters}
    documents = {"qa_list": [], "document_html": "", "document_url": document_url, "author": author,  "title": title,
    "chapters_covered": chapters}
    for question, answers, true_answer_idx in questions:
        document = {"question": question, "answers": answers, "label": true_answer_idx}
        documents["qa_list"].append(document)
    return documents

def process_summary(example_num, document_url):
    f = codecs.open("%d_content" % example_num, 'r', 'utf-8')
    # f = codecs.open("8867_content", 'r', 'utf-8')
    soup = BeautifulSoup(f.read(), 'html.parser')
    summary_container = soup.findAll("div", {"class": "studyGuideText"})[0]
    summary_elements = summary_container.findChildren(["h3", "h4", "p"])
    paragraphs = []
    for i, ele in enumerate(summary_elements):
        if(ele.name == "p"):
            paragraphs.append(ele.text)
        elif("Summary" in ele.text) or ("Commentary" in ele.text and i == 0 and document_url.split("/")[6] == ""): # Catch sparknotes typo where all of content is under analysis header
            continue
        elif("Analysis" in ele.text) or ("Commentary" in ele.text) or ("analysis" in ele.text) or ("Analyis" in ele.text): # Catching all of Sparknotes stupid edge cases
            paragraphs.append("ANALYSIS_REACHED")
            break
        else: # subtitle - ignore
            continue
    sentencized_paragraphs = []
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    for p in paragraphs:
        doc = nlp(p)
        sentences = [sent.string.strip() for sent in doc.sents]
        sentencized_paragraphs.append(sentences)

    title = soup.findAll("h1", {"class": "TitleHeader_title"})[0].text.strip()
    author = ""
    maybe_author = soup.findAll("a", {"class": "TitleHeader_authorLink"})
    if len(maybe_author) > 0: # author is link
        author = maybe_author[0].text.strip()
    else: # author is plaintext
        author = soup.findAll("div", {"class": "TitleHeader_authorName"})[0].text.strip()
    chapters = soup.findAll("h2", {"class": "interior__page__title"})[0].text.strip()
    documents = {"summary": sentencized_paragraphs, "document_html": soup.get_text(), "document_url": document_url, "author": author,  "title": title,
    "chapters_covered": chapters}
    return documents

def get_section_mappings(range_start, range_finish):
    summaries = {}
    short_qs = {}
    for i in range(range_start, range_finish):
        if i == 10930: # Broken link
            continue
        document_url = get_url(i)
        split_doc = document_url.split("/")
        book = split_doc[4]
        section = split_doc[5]
        if section == "summary":
            continue
        if len(split_doc) < 7: # ignore
            continue
        elif split_doc[6] == "": # section 1 of summary
            if book not in summaries:
                summaries[book] = {}
            if section not in summaries[book]:
                summaries[book][section] = [i]
            else:
                summaries[book][section].insert(0, i)
        elif split_doc[6][1:10] == "quickquiz":
            if book not in short_qs:
                short_qs[book] = {}
            short_qs[book][section] = i
        elif len(split_doc) == 9 and split_doc[5] != "quotes": # later section
            if book not in summaries:
                summaries[book] = {}
            if section not in summaries[book]:
                summaries[book][section] = [i]
            else:
                summaries[book][section].insert(int(split_doc[7]), i) # If index doesnt exist Python appends to end
    return summaries, short_qs

def combine_summaries(indices):
    combined_summary = []
    for idx in indices:
        sum_doc = process_summary(idx, get_url(idx))
        summary = sum_doc["summary"]
        for p in summary:
            if len(p) == 0: # empty paragraph
                continue
            elif p[0] == "ANALYSIS_REACHED":
                sum_doc["summary"] = combined_summary
                return sum_doc
            else:
                combined_summary.append(p)
    sum_doc["summary"] = combined_summary
    return sum_doc

def create_dataset(summaries, short_qs):
    book_doc = []
    counter = 0
    for book, section_dict in summaries.items():
        if book in short_qs:
            for section, indices in section_dict.items():
                if section in short_qs[book]:
                    if counter % 500 == 0:
                        print(counter)
                    sum_doc1 = combine_summaries(indices)
                    if len(sum_doc1["summary"]) == 0: # No summary
                        print(sum_doc1)
                        counter = counter + 1
                        continue
                    else:
                        curr_dict = {}
                        curr_dict["author"] = sum_doc1["author"]
                        curr_dict["title"] = sum_doc1["title"]
                        curr_dict["chapters_covered"] = sum_doc1["chapters_covered"]
                        curr_dict["summary_html"] = sum_doc1["document_html"]
                        curr_dict["summary"] = sum_doc1["summary"]
                        curr_dict["summary_url"] = sum_doc1["document_url"]
                        qa_dict = process_shortqs(short_qs[book][section], get_url(short_qs[book][section]))
                        curr_dict["qa_html"] = qa_dict["document_html"]
                        curr_dict["qa_list"] = qa_dict["qa_list"]
                        curr_dict["qa_url"] = qa_dict["document_url"]
                        book_doc.append(curr_dict)
                        counter = counter + 1
    return book_doc

if __name__ == '__main__':
    # summaries, short_qs = get_section_mappings(1, 14425)
    # book_doc = create_dataset(summaries, short_qs)

    # with open('sparknotes_with_summaries.json', 'w+') as file:
    #     json.dump({"data" : book_doc}, file)


    # print(len(book_doc["data"]))

    # final_lst = []
    # example_id = 0
    # counter = 0
    # for sum_doc, questions in book_doc:
    #     counter = counter + 1
    #     if counter % 500 == 0:
    #         print(counter)
    #     question_list = [[qa["question"] for qa in questions if len(qa["answers"]) == 4]]
    #     summary = sum_doc["summary"]
    #     # print(question_list)
    #     # print(summary)
    #     embedded_summary = get_bert_embedding(model, summary)
    #     embedded_questions = get_bert_embedding(model, question_list)
    #     # print(embedded_questions)
    #     for j, e_question in enumerate(embedded_questions[0]):
    #         # print(e_question)
    #         best_score_q = 0
    #         best_paragraph = 1000000
    #         for i, e_paragraph in enumerate(embedded_summary):
    #             # print(type(e_paragraph))
    #             best_score_p = 0
    #             for e_sentence in e_paragraph:
    #                 sim = get_cosine_similarity(e_question.reshape(1, -1), e_sentence.reshape(1, -1))
    #                 if sim > best_score_p:
    #                     best_score_p = sim
    #             if best_score_p > best_score_q:
    #                 best_score_q = best_score_p
    #                 best_paragraph = i
    #         answers = questions[j]["answers"]
    #         if j < len(questions) and best_paragraph < len(summary):
    #             row = [example_id, questions[j]["question"], " ".join(summary[best_paragraph])]
    #         else:
    #             print(str(i) + " " + str(j))
    #             print(questions)
    #             print(embedded_questions)
    #             print(summary)
    #             print(embedded_summary)
    #             print(len(summary))
    #             print(len(embedded_summary))
    #             print(len(questions))
    #             print(len(embedded_questions[0]))
    #         row.extend(answers)
    #         row.append(questions[j]["label"])
    #         final_lst.append(row)
    #         example_id = example_id + 1
    # with open('data.csv', 'w+') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(final_lst)





# summary = process_summary(14419, get_url(14419))
#     questions = process_shortqs(14420, get_url(14420))
#     summary = summary["summary"]
#     questions = [[qa["question"] for qa in questions["qa_list"]]]
#     print(summary)
#     print(questions)
#     embedded_summary = get_bert_embedding(model, summary)
#     embedded_questions = get_bert_embedding(model, questions)
#     # print(embedded_questions)
#     for j, e_question in enumerate(embedded_questions[0]):
#         # print(e_question)
#         best_score_q = 0
#         best_paragraph = 1000000
#         for i, e_paragraph in enumerate(embedded_summary):
#             # print(type(e_paragraph))
#             best_score_p = 0
#             for e_sentence in e_paragraph:
#                 sim = get_cosine_similarity(e_question.reshape(1, -1), e_sentence.reshape(1, -1))
#                 print(sim)
#                 if sim > best_score_p:
#                     best_score_p = sim
#                     print("sim")
#                     print(j)
#                     print(i)
#                     print(sim)
#             if best_score_p > best_score_q:
#                 best_score_q = best_score_p
#                 best_paragraph = i
#                 print("best p")
#                 print(j)
#                 print(i)
#         print("chosen pargraph is ")
#         print(best_paragraph)
#         print(summary[best_paragraph])
#         # print(j)
#         print(questions[0][j])
#         print("")
#         print("===")


    # books = set()
    # num_shortqs = 0
    # num_quizzes = 0
    # num_longq = 0
    # num_summaries = 0
    # num_toc = 0
    # for i in range(1, 10287):
    #     if(i % 1000 == 0):
    #         print(i)
    #     document_url = get_url(i)
    #     split_doc = document_url.split("/")
    #     books.add(split_doc[4])
    #     if split_doc[5] == "key-questions":
    #         # process_longq(i, document_url)
    #         num_longq += 1
    #         continue
    #     elif split_doc[5] == "table-of-contents":
    #         num_toc += 1
    #         continue
    #     elif split_doc[6] == " ":
    #         # process_summary(i, document_url)
    #         num_summaries += 1
    #         continue
    #     else:
    #         shortq = process_shortqs(i, document_url)
    #         num_shortqs += len(shortq["qa_list"])
    #         num_quizzes += 1
    #         # process_shortqs(i, document_url)
    # print("num_shortqs", num_shortqs)
    # print("num_quizzes", num_quizzes)
    # print("num_longq", num_longq)
    # print("num_summaries", num_summaries)
    # print("num_toc", num_toc)
    # print("questions per quiz", num_shortqs/ num_quizzes)
    # print("short questions per book", num_shortqs / len(books))
    # print("long questions per book", num_longq / len(books))

    # # print(process_shortqs(7, get_url(7)))




