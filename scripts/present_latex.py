import pandas as pd

# Read the CSV file
df = pd.read_csv("evalutaion_results.csv")

# Open the LaTeX file for writing
with open("present_evaluation_results.tex", "w") as f:
    # Start the document
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage[a4paper, margin=2.5cm]{geometry}\n")
    f.write("\\usepackage{tabularx}\n")
    f.write("\\pagenumbering{gobble}\n")
    f.write("\\begin{document}\n\n")

    # Group the data by Dataset
    for dataset, df_dataset in df.groupby("Dataset"):
        f.write(f"\\section{{{dataset}}}\n\n")
        
        # Group the data by FeatureSet within each Dataset
        for featureset, df_featureset in df_dataset.groupby("FeatureSet"):
            f.write(f"\\subsection{{{featureset}}}\n")
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Results for {featureset} on {dataset} Dataset}}\n")
            f.write("\\begin{tabular}{l|rr|rr|rr|rr}\n")
            f.write("\\hline\n")
            f.write("Model & Accuracy & STD & Precision & STD & Recall & STD & F1-Score & STD \\\\\n")
            f.write("\\hline\n")
            
            # Write each model's results
            for _, row in df_featureset.iterrows():
                f.write(f"{row['Model']} & {row['Accuracy_Mean']:.4f} & {row['Accuracy_STD']:.4f} & "
                        f"{row['Precision_Mean']:.4f} & {row['Precision_STD']:.4f} & "
                        f"{row['Recall_Mean']:.4f} & {row['Recall_STD']:.4f} & "
                        f"{row['F1_Score_Mean']:.4f} & {row['F1_Score_STD']:.4f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")

    # End the document
    f.write("\\end{document}\n")
