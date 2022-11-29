from sklearn.metrics import mean_squared_error
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import optuna
from UserItem import UserBasedRecommendation


# Добавить запись в файл результатов
def create_test_train(df):
    train_user, test_user = train_test_split(df["user"].unique())

    train = df.query("user in @train_user")
    test = df.query("user in @test_user")

    id_books = pd.DataFrame(columns=(test.columns))

    train = train.merge(
        id_books, how='left').fillna(0).astype('int64')

    test = test[sorted(test.columns)]


#     books_not_in_train = list(set(train["id_book"]) - set(test["id_book"]))
#     test.query("id_book not in @books_not_in_train", inplace=True)

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


def tune(objective):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}\n")
    print(f"Optimized parameters: {params}\n")

    with open("result.csv", "w") as file:
        file.write(f"{best_score} \n")
        file.write(f"{params}\n\n")

    return params


def UserBased_objective(trial):
    param_grid = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 45),
        "metric": trial.suggest_categorical("metric", ["correlation", "cosine", "euclidean", "cityblock"]),
    }

    scores = UserBasedRecommendation_RMSE(UserBasedRecommendation,
                                          **param_grid,
                                          df=df,
                                          number_of_verified_users=150)
    return scores


df = pd.read_csv("data.csv", index_col=0)
UserBased_params = tune(UserBased_objective)
UserBased = UserBasedRecommendation(**UserBased_params)
