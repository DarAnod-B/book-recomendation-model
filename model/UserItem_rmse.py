from sklearn.metrics import mean_squared_error
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# Добавить запись в файл результатов
def create_test_train(df):
    train_user, test_user = train_test_split(df["user"].unique())

    train = df.query("user in @train_user")
    test = df.query("user in @test_user")

    books_not_in_train = list(set(test["id_book"]) - set(train["id_book"]))
    test.query("id_book not in @books_not_in_train", inplace=True)

    print(books_not_in_train)
    return train, test


def UserBasedRecommendation_RMSE(UserBasedRecommendation,
                                 df,
                                 n_neighbors,
                                 metric,
                                 number_of_verified_users=100):

    train, test = create_test_train(df)

    model = UserBasedRecommendation(n_neighbors=n_neighbors,
                                    metric=metric)
    model.fit(train)

    y_actual = []
    y_predicted = []

    for num_user in tqdm(test["user"].unique()[:number_of_verified_users]):
        test_user = test.query("user==@num_user")
        test_user_train, test_user_test = train_test_split(test_user)

        item_score = model.predict(test_user_train)
        score = item_score.merge(test_user_test, how='inner', on='id_book')

        y_actual += list(score["predictive_grade"])
        y_predicted += list(score["grade"])

    mse = mean_squared_error(y_actual, y_predicted)
    rmse = math.sqrt(mse)

    return rmse
