import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

world_cup=pd.read_csv('World Cup 2019 Dataset.csv')
result=pd.read_csv('results.csv')
fixtures=pd.read_csv('fixtures.csv')
ranking=pd.read_csv('icc_rankings.csv')

world_cup.head()

fixtures.head()

ranking.head()

india=result[(result['Team_1']=='India')|(result['Team_2']=='India')]
india.head()

World_cup_teams=['England', ' South Africa', 'West Indies', 'Pakistan', 'New Zealand', 'Sri Lanka', 'Afghanistan', 'Australia', 'Bangladesh', 'India']
team1=result[result['Team_1'].isin(World_cup_teams)]
team2=result[result['Team_2'].isin(World_cup_teams)]
teams=pd.concat((team1,team2))
teams=teams.drop_duplicates()

team_result=teams.drop(['date','Margin','Ground'],axis=1)
team_result.head()

final_result= pd.get_dummies(team_result, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
final_result.head()

X=final_result.drop(['Winner'],axis=1)
y=final_result['Winner']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
print("Traning accuracy: ",train_score)
print("Testing accuracy: ",test_score)

fixtures.insert(1,'Team_1_position',fixtures['Team_1'].map(ranking.set_index('Team')['Position']))
fixtures.insert(2,'Team_2_position',fixtures['Team_2'].map(ranking.set_index('Team')['Position']))
fixture=fixtures.iloc[:45,:]
fixture.head()

final_set=fixture[['Team_1','Team_2']]
final_set = pd.get_dummies(final_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
for col in (set(final_result.columns)-set(final_set.columns)):
    final_set[col]=0
final_set=final_set.sort_index(axis=1)
final_set=final_set.drop(['Winner'],axis=1)
final_set.head()

prediction=model.predict(final_set)

for index,tuples in fixture.iterrows():
    print("Teams: " + tuples['Team_1']+ " and " + tuples['Team_2'])
    print("Winner:"+ prediction[index])

for i in range(len(prediction)):
    fixture['Result'].iloc[i]=prediction[i]

fixture['Result'].value_counts().plot(kind='bar')


def predict_result(matches, final_result, ranking, model, match_type):
    # obtaining team position
    team_position = []
    for match in matches:
        team_position.append([ranking.loc[ranking['Team'] == match[0], 'Position'].iloc[0],
                              ranking.loc[ranking['Team'] == match[1], 'Position'].iloc[0]])

    # transforming data into useful information
    final = pd.DataFrame()
    final[['Team_1', 'Team_2']] = pd.DataFrame(matches)
    final_set = final
    final_set = pd.get_dummies(final_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

    for col in (set(final_result.columns) - set(final_set.columns)):
        final_set[col] = 0
    final_set = final_set.sort_index(axis=1)
    final_set = final_set.drop(['Winner'], axis=1)

    # predict winner
    prediction = model.predict(final_set)

    # Results from League mathes
    if match_type == 'League':
        print("League Matches")

        final_fixture = fixtures[0:45]
        for index, tuples in final_fixture.iterrows():
            print("Teams: " + tuples['Team_1'] + " and " + tuples['Team_2'])
            print("Winner: " + prediction[index])
            fixtures['Result'].iloc[index] = prediction[index]

        Semi_final_teams = []
        for i in range(4):
            Semi_final_teams.append(fixture['Result'].value_counts().index[i])
        matches = [(Semi_final_teams[0], Semi_final_teams[3]), (Semi_final_teams[1], Semi_final_teams[2])]
        match_type = "Semi-Final"
        predict_result(matches, final_result, ranking, model, match_type)

    # Result from semi-final
    elif match_type == 'Semi-Final':
        print("\nSemi-Final Matches")
        final_fixture = fixtures[45:47]
        for index, tuples in final_fixture.iterrows():
            fixtures['Team_1'].iloc[index] = final['Team_1'].iloc[index - 45]
            fixtures['Team_2'].iloc[index] = final['Team_2'].iloc[index - 45]
            fixtures['Team_1_position'].iloc[index] = team_position[index - 45][0]
            fixtures['Team_2_position'].iloc[index] = team_position[index - 45][1]
        final_fixture = fixtures[45:47]
        for index, tuples in final_fixture.iterrows():
            print("Teams: " + tuples['Team_1'] + " and " + tuples['Team_2'])
            print("Winner: " + prediction[index - 45])
            fixtures['Result'].iloc[index] = prediction[index - 45]
        matches = [(prediction[0], prediction[1])]
        match_type = "Final"
        predict_result(matches, final_result, ranking, model, match_type)

    # Result of Final
    elif match_type == 'Final':
        print("\nFinal Match")
        final_fixture = fixtures[47:48]
        for index, tuples in final_fixture.iterrows():
            fixtures['Team_1'].iloc[index] = final['Team_1'].iloc[index - 47]
            fixtures['Team_2'].iloc[index] = final['Team_2'].iloc[index - 47]
            fixtures['Team_1_position'].iloc[index] = team_position[index - 47][0]
            fixtures['Team_2_position'].iloc[index] = team_position[index - 47][1]
        final_fixture = fixtures[47:48]
        for index, tuples in final_fixture.iterrows():
            print("Teams: " + tuples['Team_1'] + " and " + tuples['Team_2'])
            print("Winner: " + prediction[0] + "\n")
            fixtures['Result'].iloc[index] = prediction[index - 47]
        print("Winner Of the tournament is: " + fixtures['Result'].iloc[47])