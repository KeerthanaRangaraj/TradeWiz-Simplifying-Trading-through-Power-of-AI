from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your dataset
df = pd.read_csv("C:/Users/agnib/Desktop/soupPolicyBot.csv")  # Use forward slashes in file paths

# Function to generate recommendations based on user input
def get_recommendations(user_input):
    # Your recommendation logic here...
    # Ensure to properly handle 'user_input' parameter

    # For instance, assuming 'searchTerms' are obtained from 'user_input'
    searchTerms = user_input.get('searchTerms', '')  # Replace 'searchTerms' with the actual parameter name
    
    new_row = df.iloc[-1, :].copy()
    new_row.iloc[-1] = " ".join(searchTerms)
    df.loc[len(df)] = new_row

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['Soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    
    sim_scores = list(enumerate(cosine_sim2[-1, :]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    ranked_titles = []
    for i in range(1, min(11, len(sim_scores))):  # Ensure it doesn't exceed the length of sim_scores
        indx = sim_scores[i][0]
        ranked_titles.append([df['Title'].iloc[indx], df['URL'].iloc[indx]])

    return ranked_titles
input_data=[]
input_count=0
@app.route('/webhook', methods=['POST'])
def webhook():
    global input_data, input_count
    
    data = request.get_json(silent=True)
    user_input = data['queryResult']['parameters']
    
    print("Data: ",data)
    print("\n User Input: ",list(user_input.values())[0],type(list(user_input.values())[0]),(list(user_input.values())[0][0]))
    if type(list(user_input.values())[0])==list:
        input_data.append((list(user_input.values())[0][0]))
    else:    
        input_data.append(list(user_input.values())[0])
    print(input_data)
    input_count=input_count+1
    if input_count==9:
        recommendations = get_recommendations(user_input)
        print("\n Reco: ",recommendations,len(recommendations))

        response_text = f"Here are your recommendations:\n"
        response_text += "\n".join([f"{i}: {rec}" for i, rec in enumerate(recommendations, 1)])

        print(response_text)
        response= {
                    "fulfillmentMessages": [
                        {
                            "text": {
                                "text": [
                                    response_text
                                ]
                            }
                        }]}
        input_data=[]
        return jsonify(response)


    
    return "", 204

if __name__ == '__main__':  # Fix the condition to properly check for '__main__'
    app.run()
