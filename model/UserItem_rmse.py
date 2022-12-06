from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def create_actual_predict(model, test, num_user):
    test_user = test.query("user==@num_user")
    test_user_train, test_user_test = train_test_split(test_user)

    item_score = model.predict(test_user_train)
    score = item_score.merge(test_user_test, how='inner', on='id_book')

    y_actual = list(score["grade"])
    y_predicted = list(score["predictive_grade"])

    return y_actual, y_predicted


def create_test_train(df):
    train_user, test_user = train_test_split(df["user"].unique())

    train = df.query("user in @train_user")
    test = df.query("user in @test_user")

    books_not_in_train = list(set(test["id_book"]) - set(train["id_book"]))
    test.query("id_book not in @books_not_in_train", inplace=True)

    print(books_not_in_train)
    return train, test


def create_actual_predict_list(model,
                               df,
                               n_neighbors,
                               metric,
                               number_of_verified_users=100):

    train, test = create_test_train(df)

    model = model(n_neighbors=n_neighbors,
                  metric=metric)
    model.fit(train)

    y_actual_list = []
    y_predicted_list = []

    for num_user in tqdm(test["user"].unique()[:number_of_verified_users]):
        y_actual, y_predicted = create_actual_predict(model, test, num_user)

        y_actual_list += y_actual
        y_predicted_list += y_predicted

    return y_actual_list, y_predicted_list


def rmse(y_actual, y_predicted):
    mse = mean_squared_error(y_actual, y_predicted)
    rmse = sqrt(mse)
    return rmse


def UserBasedRecommendation_RMSE(processing_type,
                                 model,
                                 data,
                                 *args,
                                 number_of_verified_users=100):
    if processing_type == 'tune':
        y_actual, y_predicted = create_actual_predict_list(model,
                                                           data,
                                                           *args,
                                                           number_of_verified_users=number_of_verified_users)
    elif processing_type == 'test':
        y_actual, y_predicted = create_actual_predict(model,
                                                      data,
                                                      *args)

    return rmse(y_actual, y_predicted)
