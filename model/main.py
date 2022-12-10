import pandas as pd
from UserItem_model import UserBasedRecommendation
from user_book_list import creating_custom_book_list

df = pd.read_csv(
    r"C:\Programming\GitHub\book_recomendation_system\data\data.csv", index_col=0)
user_book_list = creating_custom_book_list(df)

model = UserBasedRecommendation()
model.fit(df)

score = model.predict(user_book_list)
