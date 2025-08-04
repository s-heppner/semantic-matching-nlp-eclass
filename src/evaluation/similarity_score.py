import json
import numpy as np
import pandas as pd

# Load JSON data
json_path = "../../data/embeddings/original/eclass-13-embeddings-baai.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df["embedding_array"] = df["embedding"].apply(np.array)

# Filter embeddings with "STRAK" in preferred name
strak_df = df[df["preferred-name"].str.contains("STRAK", case=False, na=False)].copy()

print("\nPairwise cosine similarity between 'STRAK' embeddings:\n")
for i in range(len(strak_df)):
    for j in range(i + 1, len(strak_df)):
        name_i = strak_df.iloc[i]["preferred-name"]
        name_j = strak_df.iloc[j]["preferred-name"]
        vec_i = strak_df.iloc[i]["embedding_array"]
        vec_j = strak_df.iloc[j]["embedding_array"]
        sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
        print(f"{name_i} | {name_j}: {sim:.3f}")

# Pick 5 random non-STRAK embeddings
non_strak_df = df[~df["preferred-name"].str.contains("STRAK", case=False, na=False)].copy()

random_sample = non_strak_df.sample(5, random_state=42).copy()

print("\nCosine similarity between each 'STRAK' embedding and 5 random others:\n")
for _, strak_row in strak_df.iterrows():
    name_strak = strak_row["preferred-name"]
    vec_strak = strak_row["embedding_array"]

    for _, rand_row in random_sample.iterrows():
        name_rand = rand_row["preferred-name"]
        vec_rand = rand_row["embedding_array"]
        sim = np.dot(vec_strak, vec_rand) / (np.linalg.norm(vec_strak) * np.linalg.norm(vec_rand))
        print(f"{name_strak} | {name_rand}: {sim:.3f}")

print("\nRandom preferred names sampled:")
print(random_sample["preferred-name"].tolist())