"""
    Contains helper function for data cleaning and transformation purpose.
    
    Meant to run as script
"""
import pandas as pd
import os
class MyDataCleaner():
    def __init__(self):
        pass
    
    # Remove { ’ -> ' ; ” -> " }
    @staticmethod
    def clean_csv(filepath:str, new_filepath:str, remove:list[tuple[str, str]], *, if_strip=False):
        """
        loads the csv file, cleans it and stores it

        Parameters
        ----------
        filepath : file_path for the csv
        new_filepath : saves the cleaned corpus(there will be no structure just a blob)
        remove : a list of tuple with (og_str, new_str). To just remove keep new_str=""
        """
        file = pd.read_csv(filepath)
        content = file.to_numpy()
        cleaned_content = []
        corpus = ""
        for record in content:
            row = []
            for value in record:
                clean = str(value) # to get a copy
                if if_strip:
                    clean = clean.strip()
                for og, new in remove:
                    clean = clean.replace(og, new)
                row.append(clean)
            cleaned_content.append(' '.join(row))
                
        corpus = '\n'.join(cleaned_content)
        with open(new_filepath, mode="w", encoding="utf-8") as file:
            out = file.write(corpus)
            print(f"** Wrote: {out} chr in file:{new_filepath}")
    
    @staticmethod
    def merge_files(files:list[str], merged_filepath:str ,if_replace=False):
        """merges the content of files in the provided dirs
        
        Parameters
        ---
            files : a list of filepaths whose content to merge
            merged_filepath : out file
            if_replace : will remove all the files that were merged 
        """
        with open(merged_filepath, mode="w", encoding="utf-8") as mg_file :
            total_chr = 0
            for file in files:
                with open(file, mode="r", encoding="utf-8") as f:
                    for content in f.readlines():
                        total_chr += mg_file.write(content)
                if if_replace:
                    os.remove(file)
        print(f"** Total chars written: {total_chr}")

if __name__ == "__main__":
    replace = [("’", "'"), ("”", '"')]
    tokenizer_corpus = ""
    MyDataCleaner.clean_csv(
        "./data/kaggle/news1_Fake.csv", 
        "./data/#synthetic/news1_fake.txt", 
        replace, 
        if_strip=True
    )
    
    MyDataCleaner.clean_csv(
        "./data/kaggle/news1_True.csv", 
        "./data/#synthetic/news1_true.txt", 
        replace, 
        if_strip=True
    )
    
    MyDataCleaner.merge_files(
        files = ["./data/#synthetic/news1_fake.txt", "./data/#synthetic/news1_true.txt"],
        merged_filepath= "./data/#synthetic/tokenizer_corpus.txt",
        if_replace=True
    )
            