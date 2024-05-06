from langchain.text_splitter import NLTKTextSplitter

# Load a long document
with open('../data/my_file.txt', encoding= 'unicode_escape') as f:
    sample_text = f.read()

text_splitter = NLTKTextSplitter(chunk_size=500)
texts = text_splitter.split_text(sample_text)
print(texts)