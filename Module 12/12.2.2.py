from transformers import pipeline

# Named entity recognition
ner = pipeline(
    task="ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
    device=-1,
)

text = "Anthropic was founded in San Francisco by Dario Amodei and Daniela Amodei."
entities = ner(text)
for ent in entities:
    print(f"{ent['word']!r:25s} ->{ent['entity_group']} ({ent['score']:.3f})")