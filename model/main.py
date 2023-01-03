import pandas as pd
from UserItem_model import UserBasedRecommendation


def main():
    target_user_id = 1

    df = pd.read_csv(
        r"C:\Programming\GitHub\book-recomendation-system\data\data.csv", index_col=0)

    train = df.query("user != @target_user_id")
    test = df.query("user == @target_user_id")

    model = UserBasedRecommendation()
    model.fit(train, user_id_col_name='user',
              item_id_col_name='id_book', grade_col_name='grade')

    score = model.predict(test)
    print(score)


if __name__ == "__main__":
    main()
