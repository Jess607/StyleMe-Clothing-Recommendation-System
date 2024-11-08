#import libraries
from Outfit_methods.dress_methods import get_dress_recommendations
from Outfit_methods.pants_methods import get_shirts_recommendations 
from Outfit_methods.skirt_methods import get_skirt_recommendations
from flask import Flask, request, jsonify


#instantiate flask app
app = Flask(__name__)


#function handles the gathering of recommendations for every clothing item
def get_recommendations(skirts=None, shirts=None, bags=None, trousers=None, shoes=None, dresses=None):
    skirt_list=get_skirt_recommendations(skirts=skirts, shirts=shirts, shoes=shoes, bags=bags)
    trouser_list=get_shirts_recommendations(shirts=shirts, trousers=trousers, shoes=shoes, bags=bags )
    dress_list=get_dress_recommendations(dresses=dresses, shoes=shoes, bags=bags)
    full_list=skirt_list + trouser_list + dress_list
    return full_list



# Define the route for the recommendation API
@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Parse input parameters from the query string
    data=request.json
    skirts = data.get('skirts')
    shirts = data.get('shirts')
    bags = data.get('bags')
    trousers = data.get('trousers')
    shoes = data.get('shoes')
    dresses = data.get('dresses')

    # Call the get_recommendations method with the parsed parameters
    recommendation_list = get_recommendations(
        skirts=skirts,
        shirts=shirts,
        bags=bags,
        trousers=trousers,
        shoes=shoes,
        dresses=dresses
    )

    # Return the recommendations as a JSON response
    return jsonify({ "suggested_outfits":recommendation_list })

if __name__ == '__main__':
    app.run(debug=True)









