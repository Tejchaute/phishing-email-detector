import pandas as pd
import re

def clean_single_email(text):
    text = str(text).strip().lower()
    return text

def clean_email_data(df):
    # Drop duplicates
    df = df.drop_duplicates(subset=['subject', 'body'])
    
    # Drop empty / missing
    df = df.dropna(subset=['body', 'label'])
    
    # Normalize text
    for col in ['subject', 'body']:
        df[col] = df[col].astype(str).str.strip().str.lower()
    
    # Keep only 0/1 labels
    df = df[df['label'].isin([0, 1])]
    return df

def remove_noisy_rows(df):
    # Length filters
    df = df[df['subject'].str.len() > 5]
    df = df[df['body'].str.len() > 20]  # stricter than before
    
    # Remove rows with weird encoding artifacts
    encoding_garbage = r'(Ã|¢Â|ð|ÿ|þ|�)'
    df = df[~df['body'].str.contains(encoding_garbage, na=False)]
    
    # Remove rows that are mostly non-alphabetic
    df = df[df['body'].apply(lambda x: len(re.findall(r'[a-zA-Z]', x)) / len(x) > 0.5 if len(x) > 0 else False)]
    
    # Remove rows that look like lists of emails / usernames
    df = df[~df['body'].str.contains(r'@[a-z0-9.-]+', na=False)]  # email-like text
    
    # Remove common noisy phrases
    noisy_keywords = [
        'added:', 'submission notes', 'virus total', 'virscan.org',
        'best regards', 'sincerely', 'dear sir', 'mr.', 'regards', '--'
    ]
    pattern = '|'.join([re.escape(k) for k in noisy_keywords])
    df = df[~df['body'].str.contains(pattern, case=False, na=False)]
    
    return df

def merge_and_clean_csv_files(file_paths, output_path):
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path, encoding='utf-8')
        df_clean = clean_email_data(df)
        df_clean = remove_noisy_rows(df_clean)
        dfs.append(df_clean)
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Final pass
    merged_df = clean_email_data(merged_df)
    merged_df = remove_noisy_rows(merged_df)
    
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Merged and cleaned CSV saved to {output_path}")

if __name__ == "__main__":
    file_list = [
        r'D:\phishing-email-detector\data\CEAS_08.csv',
        r'D:\phishing-email-detector\data\Nigerian_Fraud.csv',
        r'D:\phishing-email-detector\data\SpamAssasin.csv',
        r'D:\phishing-email-detector\data\Nazario.csv'
    ]
    merge_and_clean_csv_files(file_list, r'D:\phishing-email-detector\data\merged_cleaned_emails.csv')
