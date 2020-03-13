import csv
import os


train_perc = 0.8
val_perc = 0.1
test_perc = 0.1

def _read_csv(input_file):
	with open(input_file, "r", encoding="utf-8") as f:
		return list(csv.reader(f))

def _write_csv(out_file, data):
	with open(out_file, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['id', 'question', 'contexts', 'ending0', 'ending1', 'ending2', 'ending3', 'label'])
		writer.writerows(data)

def list_splitter(data):
	elements = len(data)
	train_end = int(elements*train_perc)
	val_end = int(elements*(train_perc + val_perc))
	return data[:train_end], data[train_end:val_end], data[val_end:]

def  main():
	data = _read_csv("../sparknotes_processed_data/BERT_processed_data.csv")
	train, val, test = list_splitter(data)

	base_data_dir = "../data/"
	_write_csv(base_data_dir + "train.csv", train)
	_write_csv(base_data_dir + "val.csv", val)
	_write_csv(base_data_dir + "test.csv", test)


if __name__ == "__main__":
	main()