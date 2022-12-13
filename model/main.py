import pandas as pd
from UserItem_model import UserBasedRecommendation

df = pd.read_csv(
    r"C:\Programming\GitHub\book-recomendation-system\data\data.csv", index_col=0)

train = df.query("user != 1")
test = df.query("user == 1")

model = UserBasedRecommendation()
model.fit(train, user_id_col_name='user',
          item_id_col_name='id_book', grade_col_name='grade')

score = model.predict(test)
print(score)
