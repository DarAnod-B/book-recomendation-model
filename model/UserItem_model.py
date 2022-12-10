from sklearn.neighbors import NearestNeighbors
import pandas as pd


class UserBasedRecommendation:
    """
    User based recommendation system.
    This class makes recommendations to the user using User-Based Collaborative Filtering
    """

    def __init__(
        self,
        n_neighbors=5,
        metric="correlation",
    ):
        self.metric = metric
        self.n_neighbors = n_neighbors

    def fit(self, X):
        # Training the KNN model and creating a pivot table of training data.
        X_pt_grade = X.pivot_table(index='user',
                                   columns='id_book',
                                   values='grade').fillna(0).astype('int64')
        X_pt_grade = X_pt_grade[sorted(
            X_pt_grade.columns)]

        knn = NearestNeighbors(metric=self.metric)
        knn.fit(X_pt_grade.values)

        self.knn = knn
        self.X_pt_grade = X_pt_grade

    def predict(self, y):
        # Predicting a list of books the user will like in the form of a dataset with tables id_book and predictive_grade.
        y_pt_grade = self._prepare_y(y)
        similar_users = self._find_similar_user(y_pt_grade)
        recommendation = self._recommend(y_pt_grade, similar_users)
        return recommendation

    def _prepare_y(self, y):
        # Converts the incoming dataset into a pivot table suitable for working with KNN model.
        y_pt_grade = y.pivot_table(index='user',
                                   columns='id_book',
                                   values='grade')

        # Adding books that are not in the user's ratings.
        id_books = pd.DataFrame(columns=(self.X_pt_grade.columns))
        y_pt_grade = y_pt_grade.merge(
            id_books, how='left').fillna(0).astype('int64')

        y_pt_grade = y_pt_grade[sorted(y_pt_grade.columns)]
        return y_pt_grade

    def _find_similar_user(self, y_pt_grade):
        distances, user_indices_in_knn = self.knn.kneighbors(y_pt_grade,
                                                             n_neighbors=self.n_neighbors)
        distances = distances[0]
        user_indices_in_knn = user_indices_in_knn[0]

        user_indices = self.X_pt_grade.iloc[user_indices_in_knn].index

        data = {'user': user_indices.tolist(),
                'weight': distances.tolist()}
        similar_users = pd.DataFrame(data).sort_values(
            by='weight', ascending=False)

        return similar_users

    def _recommend(self, y_pt_grade, similar_users):
        X_stack = self.X_pt_grade.stack().reset_index()
        X_stack.rename(columns={0: 'grade'}, inplace=True)

        # Dataframe with columns "user weight id_book grade"
        similar_users_info = similar_users.merge(X_stack, on='user', how='left')\
            .query("grade > 0")

        mean_grade_similar_users = similar_users_info.groupby(by="user",
                                                              as_index=False)['grade'].mean().round(2)
        similar_users_info = similar_users_info.merge(mean_grade_similar_users,
                                                      on='user',
                                                      suffixes=('', '_mean'))
        mean_grade_target_user = y_pt_grade[y_pt_grade.gt(0)].mean(axis=1)[
            0].round(2)

        sum_weight = similar_users['weight'].abs().sum().round(2)

        item_score = similar_users_info.groupby('id_book')\
            .apply(self._prediction_calculation, mean_grade_target_user=mean_grade_target_user, sum_weight=sum_weight)

        item_score = item_score.sort_values(ascending=False)\
            .reset_index()\
            .rename(columns={0: 'predictive_grade'})

        return item_score

    @staticmethod
    def _prediction_calculation(similar_users_info, mean_grade_target_user, sum_weight):
        predicted_grade_target_user = mean_grade_target_user + (
            (similar_users_info['grade_mean'] - similar_users_info['grade'])*similar_users_info['weight']).sum() / sum_weight

        return predicted_grade_target_user
