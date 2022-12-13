from sklearn.neighbors import NearestNeighbors
import pandas as pd
from enum import Enum


class MinBook(Enum):
    min_dataset_book = 100
    min_user_book = 5


class UserBasedRecommendation:
    """
    User based recommendation system.
    This class makes recommendations to the user using User-Based Collaborative Filtering
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "correlation",
    ) -> None:
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.user_id_col_name = None
        self.item_id_col_name = None
        self.grade_col_name = None
        self.knn = None
        self.X_pt_grade = None

    def fit(self, X: pd.DataFrame, user_id_col_name: str, item_id_col_name: str, grade_col_name: str) -> None:
        self.user_id_col_name = user_id_col_name
        self.item_id_col_name = item_id_col_name
        self.grade_col_name = grade_col_name

        assert X.shape[
            0] > MinBook.min_dataset_book.value, f"The dataset must contain at least {MinBook.min_dataset_book.value} ratings from users to create a recommendation."

        # Training the KNN model and creating a pivot table of training data.
        X_pt_grade = X.pivot_table(index=self.user_id_col_name,
                                   columns=self.item_id_col_name,
                                   values=self.grade_col_name).fillna(0).astype('int64')
        X_pt_grade = X_pt_grade[sorted(
            X_pt_grade.columns)]

        knn = NearestNeighbors(metric=self.metric)
        knn.fit(X_pt_grade.values)

        self.knn = knn
        self.X_pt_grade = X_pt_grade

    def predict(self, y: pd.DataFrame) -> pd.DataFrame:
        # Predicting a list of books the user will like in the form of a dataset with tables id_book and predictive_grade.
        y_pt_grade = self._prepare_y(y)
        similar_users = self._find_similar_user(y_pt_grade)
        recommendation = self._recommend(y_pt_grade, similar_users)
        return recommendation

    def _prepare_y(self, y: pd.DataFrame) -> pd.DataFrame:
        assert y.shape[0] > MinBook.min_user_book.value, f"The user must have at least {MinBook.min_user_book.value} book to make a recommendation."

        # Converts the incoming dataset into a pivot table suitable for working with KNN model.
        y_pt_grade = y.pivot_table(index=self.user_id_col_name,
                                   columns=self.item_id_col_name,
                                   values=self.grade_col_name)

        # Adding books that are not in the user's ratings.
        id_books = pd.DataFrame(columns=(self.X_pt_grade.columns))
        y_pt_grade = y_pt_grade.merge(
            id_books, how='left').fillna(0).astype('int64')

        y_pt_grade = y_pt_grade[sorted(y_pt_grade.columns)]
        return y_pt_grade

    def _find_similar_user(self, y_pt_grade: pd.DataFrame) -> pd.DataFrame:
        distances, user_indices_in_knn = self.knn.kneighbors(y_pt_grade,
                                                             n_neighbors=self.n_neighbors)
        distances = distances[0]
        user_indices_in_knn = user_indices_in_knn[0]

        user_indices = self.X_pt_grade.iloc[user_indices_in_knn].index

        data = {self.user_id_col_name: user_indices.tolist(),
                'weight': distances.tolist()}
        similar_users = pd.DataFrame(data).sort_values(
            by='weight', ascending=False)

        return similar_users

    def _recommend(self, y_pt_grade: pd.DataFrame, similar_users: pd.DataFrame) -> pd.DataFrame:
        X_stack = self.X_pt_grade.stack().reset_index()
        X_stack.rename(columns={0: self.grade_col_name}, inplace=True)

        # Dataframe with columns "user weight id_book grade"
        similar_users_info = similar_users.merge(X_stack, on=self.user_id_col_name, how='left')\
            .query(f"{self.grade_col_name} > 0")

        mean_grade_similar_users = similar_users_info.groupby(by=self.user_id_col_name,
                                                              as_index=False)[self.grade_col_name].mean().round(2)
        similar_users_info = similar_users_info.merge(mean_grade_similar_users,
                                                      on=self.user_id_col_name,
                                                      suffixes=('', '_mean'))
        mean_grade_target_user = y_pt_grade[y_pt_grade.gt(0)].mean(axis=1)[
            0].round(2)

        sum_weight = similar_users['weight'].abs().sum().round(2)

        item_score = similar_users_info.groupby(self.item_id_col_name)\
            .apply(self._prediction_calculation, mean_grade_target_user=mean_grade_target_user, sum_weight=sum_weight)

        item_score = item_score.sort_values(ascending=False)\
            .reset_index()\
            .rename(columns={0: 'predictive_grade'})

        return item_score

    def _prediction_calculation(self, similar_users_info: pd.DataFrame, mean_grade_target_user: float, sum_weight: float) -> pd.DataFrame:
        predicted_grade_target_user = mean_grade_target_user + (
            (similar_users_info['grade_mean'] - similar_users_info[self.grade_col_name])*similar_users_info['weight']).sum() / sum_weight

        return predicted_grade_target_user
