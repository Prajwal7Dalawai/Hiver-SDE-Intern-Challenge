import pandas as pd
from pathlib import Path


DATA_PATH = Path("./twcs/twcs.csv")

def analyze_brand(df, brand):
    brand_tweets = df[df["author_id"] == brand]

    # IDs of tweets written by the brand
    brand_tweet_ids = set(brand_tweets["tweet_id"])

    # Customer tweets
    customer_tweets = df[df["inbound"] == True]

    # Customer tweets that have a response
    # from the selected brand
    responded = customer_tweets[
        customer_tweets["response_tweet_id"].apply(
            lambda x: any(
                str(tweet_id) in str(x).split(",")
                for tweet_id in brand_tweet_ids
            )
            if pd.notna(x)
            else False
        )
    ]

    return {
        "brand": brand,
        "brand_tweets": len(brand_tweets),
        "customer_tweets": len(customer_tweets),
        "customer_tweets_with_brand_response": len(responded)
    }


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("\n===== DATASET OVERVIEW =====")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\n===== MISSING VALUES =====")
    missing = df.isnull().sum()

    for column, count in missing.items():
        percentage = count / len(df) * 100
        print(f"{column:30} {count:10,} ({percentage:.2f}%)")

    print("\n===== INBOUND / OUTBOUND =====")
    print(df["inbound"].value_counts())
    print("\nPercentage:")
    print(df["inbound"].value_counts(normalize=True).mul(100).round(2))

    print("\n===== UNIQUE AUTHORS =====")
    print(f"Unique authors: {df['author_id'].nunique():,}")

    print("\n===== TOP AUTHORS =====")
    print(df["author_id"].value_counts().head(30))

    print("\n===== CANDIDATE SUPPORT ACCOUNTS =====")

    top_authors = df["author_id"].value_counts().head(30)

    for author, count in top_authors.items():

        brand_tweets = df[df["author_id"] == author]

        inbound = brand_tweets["inbound"].sum()
        outbound = (~brand_tweets["inbound"]).sum()

        print(
            f"{author:25} "
            f"total={count:8,} "
            f"inbound={inbound:8,} "
            f"outbound={outbound:8,}"
        )

    print("\n===== RESPONSE RELATIONSHIPS =====")
    print(
        f"Tweets with response_tweet_id: "
        f"{df['response_tweet_id'].notna().sum():,}"
    )

    print(
        f"Tweets with in_response_to_tweet_id: "
        f"{df['in_response_to_tweet_id'].notna().sum():,}"
    )

    print("\n===== DATE RANGE =====")

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce"
    )

    print(f"Earliest: {df['created_at'].min()}")
    print(f"Latest:   {df['created_at'].max()}")

    

    result = analyze_brand(df, "AmazonHelp")

    print("\n===== BRAND ANALYSIS =====")
    for key, value in result.items():
        print(f"{key}: {value:,}")


if __name__ == "__main__":
    main()