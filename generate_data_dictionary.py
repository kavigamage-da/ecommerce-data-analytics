# generate_data_dictionary.py
import pandas as pd

# List of CSVs and sections
csv_files = {
    "customer_profiles_10k.csv": "Customer Profiles",
    "purchase_history_10k.csv": "Purchase History",
    "engagement_behavior_10k.csv": "Engagement Behavior",
    "marketing_promotions_10k.csv": "Marketing Promotions"
}

# Create the data dictionary markdown
with open("data/data_dictionary.md", "w") as f:
    f.write("# Data Dictionary\n\n")
    
    for file, section in csv_files.items():
        df = pd.read_csv(f"data/{file}")
        f.write(f"## {section}\n\n")
        f.write("| Column Name | Data Type | Description |\n")
        f.write("|-------------|----------|-------------|\n")
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            desc = col.replace("_", " ").capitalize()
            f.write(f"| {col} | {dtype} | {desc} |\n")
        
        f.write("\n")

print("✅ data_dictionary.md created in data/ folder")