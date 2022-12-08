import pandas as pd
import optuna
from UserItem_model import UserBasedRecommendation
from UserItem_rmse import UserBasedRecommendation_RMSE


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
                                          df,
                                          **param_grid,
                                          number_of_verified_users=1)
    return scores


if __name__ == "__main__":
    df = pd.read_csv(
        r"C:\Programming\GitHub\book_recomendation_system\model\data.csv", index_col=0)
    UserBased_params = tune(UserBased_objective)
    UserBased = UserBasedRecommendation(**UserBased_params)
