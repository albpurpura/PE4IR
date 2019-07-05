from utils.input_output import load_dataset, build_context_dataset
import tables
import os
import numpy as np

root = r'/mnt/DATA/Prob_IR/'
collection_folder = r'corpus'
stoplist_file = r'indri_stoplist_eng.txt'
encoded_docs_filename = r'encoded_docs_model'
word_index_filename = r'word_index'
context_dataset_name = r'context_data'

print("Loading data...")
docs, words = load_dataset(root, encoded_docs_filename, word_index_filename)
docs = [x for x in docs if x] # Remove empty docs
file_size = len(docs)//3

atom = tables.Int32Atom()
with tables.open_file(os.path.join(root, context_dataset_name), mode='w') as f:
    f.create_earray(f.root, 'data', atom, (0, 3))

for i in range(4):
    print("Building context dataset: " + str(i))
    start = i*file_size
    end = i*file_size + file_size if i < 3 else len(docs)

    context = build_context_dataset(docs[start:end], 10)
    context = np.array(context)

    with tables.open_file(os.path.join(root, context_dataset_name), mode='a') as f:
        print("Saving...")
        f.root.data.append(context)

    del context