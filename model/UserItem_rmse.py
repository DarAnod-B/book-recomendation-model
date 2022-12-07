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
    # Переменная для добавления пользователя с первым экземпляра каждой книги в тренировочный датасет
    # иначе тренировочный и тестовый датасеты будут иметь разную размерность.
    user_first_copy_book = set(df.groupby(by="id_book").first()["user"])

    train = df.query("user in @train_user or user in @user_first_copy_book")
    test = df.query("user in @test_user and user not in @user_first_copy_book")
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
                                 number_of_verified_users=100,
                                 **args):

    if processing_type == 'tune':
        y_actual, y_predicted = create_actual_predict_list(model,
                                                           data,
                                                           number_of_verified_users=number_of_verified_users,
                                                           **args)
    elif processing_type == 'test':
        y_actual, y_predicted = create_actual_predict(model,
                                                      data,
                                                      **args)

    return rmse(y_actual, y_predicted)
