from retriever import Retriever

r = Retriever()
query = "Can I take liquid medication over 100ml in my hand baggage?"
hits = r.search(query, k=5)

print("Query:", query)
print("\nTop hits:\n")
for h in hits:
    print(f"- {h['chunk_id']} | {h['section']} | score={h['score']:.3f}")
    print(h["text"][:400].replace("\n", " "))
    print()
