import pandas as pd
from UserItem_model import UserBasedRecommendation

df = pd.read_csv(
        r"C:\Programming\GitHub\book_recomendation_system\data\data.csv", index_col=0)

model = UserBasedRecommendation()
model.fit(df)

test = model.creation_y()
score = model.predict(test)