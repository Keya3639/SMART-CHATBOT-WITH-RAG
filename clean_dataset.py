import pandas as pd

# Load dataset
df = pd.read_csv("data/rag.csv", sep="\t", header=None,
                 names=["id", "query", "domain", "intent", "response"])

# Step 1: Drop rows where query or response is missing
df = df.dropna(subset=["query", "response"]).reset_index(drop=True)

# Step 2: Ensure all columns are strings before applying .str methods
df["query"] = df["query"].astype(str).str.lower().str.strip()
df["response"] = df["response"].astype(str).str.strip()
df["domain"] = df["domain"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()

# Step 3: Save cleaned dataset
df.to_csv("data/rag_cleaned.csv", sep="\t", index=False)

print("Dataset cleaned and saved as 'rag_cleaned.csv'")
print("Sample data:\n", df.head())
