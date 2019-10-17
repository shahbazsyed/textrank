from summa import summarizer
import glob


documents = glob.glob('data/*.txt')
print(documents)
for doc in documents:
    print(f'Processing {doc} ...')
    out_path = 'summaries'+doc.replace('data','').replace('.txt','_lex_summary.txt')
    with open(doc, 'r') as inf, open(out_path, 'w') as out:
        text = inf.read()
        summary = summarizer.summarize(text,embedding_similarity=True, ratio=0.3)
        if len(summary):
            out.write(summary)
        else:
            print("Skipping empty summary")

        