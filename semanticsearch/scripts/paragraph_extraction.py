import codecs, copy, torch, csv, json, torch
import numpy as np
# import numpy.linalg as LA
from models import InferSent
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

def get_cosine_similarity(a1, a2):
    return cosine_similarity(a1.reshape(1, -1), a2.reshape(1, -1))

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

if __name__ == '__main__':
    embedding_method = "sentence_bert"

    if embedding_method == "infersent":
        MODEL_PATH = "encoder/infersent2.pkl"
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH))
        W2V_PATH = 'fastText/crawl-300d-2M.vec'
        model.set_w2v_path(W2V_PATH)
        model.build_vocab_k_words(K=100000)
    elif embedding_method == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
    elif embedding_method == "sentence_bert":
        model = SentenceTransformer('bert-base-nli-mean-tokens')
    elif embedding_method == "tfidf":
        vectorizer = TfidfVectorizer()

    with open('sparknotes_dataset.json', 'r') as f:
        book_doc = json.load(f)

    final_lst = []
    example_id = 0
    counter = 0
    for doc in book_doc["data"]:
        counter = counter + 1
        if counter % 100 == 0:
            print(counter)
            print("===========================")
        questions = doc["qa_list"]
        question_list = [[qa["question"] for qa in questions if len(qa["answers"]) == 4]]
        summary = doc["summary"]
        if len(summary) == 0 or len(question_list) == 0 or [] in question_list or [] in summary:
            continue

        if embedding_method == "infersent":
            embedded_summary = [model.encode(i, bsize=128, tokenize=False, verbose=False) for i in summary]
            embedded_questions = [model.encode(j, bsize=128, tokenize=False, verbose=False) for j in question_list]

        elif embedding_method == "bert":
            embedded_summary = get_bert_embedding(model, summary)
            embedded_questions = get_bert_embedding(model, question_list)

        elif embedding_method == "sentence_bert":
            embedded_summary = [model.encode(i) for i in summary]
            embedded_questions = [model.encode(j) for j in question_list]

        elif embedding_method == "tfidf":
            for j, question in enumerate(question_list[0]):
                best_score_q = -1000000
                best_paragraph = 1000000
                for i, paragraph in enumerate(summary):
                    embedded_summary, embedded_question = vectorizer.fit_transform([" ".join(paragraph), question])
                    embedded_summary = embedded_summary.todense()
                    embedded_question = embedded_question.todense()
                    sim = get_cosine_similarity(embedded_question.reshape(1, -1), embedded_summary.reshape(1, -1))
                    if sim > best_score_q:
                        best_score_q = sim
                        best_paragraph = i
                answers = questions[j]["answers"]
                if j < len(questions) and best_paragraph < len(summary):
                    row = [example_id, questions[j]["question"], " ".join(summary[best_paragraph])]
                else:
                    print(str(i) + " " + str(j))
                    print(questions)
                    print(embedded_questions)
                    print(summary)
                    print(embedded_summary)
                    print(len(summary))
                    print(len(embedded_summary))
                    print(len(questions))
                    print(len(embedded_questions[0]))
                row.extend(answers)
                row.append(questions[j]["label"])
                final_lst.append(row)
                example_id = example_id + 1

        if embedding_method == "bert" or embedding_method == "infersent" or embedding_method == "sentence_bert":
            for j, e_question in enumerate(embedded_questions[0]):
                best_score_q = -1000000
                best_paragraph = 1000000
                for i, e_paragraph in enumerate(embedded_summary):
                    best_score_p = -100000
                    for e_sentence in e_paragraph:
                        sim = get_cosine_similarity(e_question.reshape(1, -1), e_sentence.reshape(1, -1))
                        if sim > best_score_p:
                            best_score_p = sim
                    if best_score_p > best_score_q:
                        best_score_q = best_score_p
                        best_paragraph = i
                answers = questions[j]["answers"]
                if j < len(questions) and best_paragraph < len(summary):
                    row = [example_id, questions[j]["question"], " ".join(summary[best_paragraph])]
                else:
                    print(str(i) + " " + str(j))
                    print(questions)
                    print(embedded_questions)
                    print(summary)
                    print(embedded_summary)
                    print(len(summary))
                    print(len(embedded_summary))
                    print(len(questions))
                    print(len(embedded_questions[0]))
                row.extend(answers)
                row.append(questions[j]["label"])
                final_lst.append(row)
                example_id = example_id + 1

    with open(embedding_method + '_data.csv', 'w+') as file:
        writer = csv.writer(file)
        writer.writerows(final_lst)
