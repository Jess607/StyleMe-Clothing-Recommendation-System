import torch
from PIL import Image, ImageTk
import os
from io import BytesIO
import io
from imp import reload
import matplotlib.pyplot as plt
import torchvision.models as models
from preprocess import preprocess_image
import torchvision.transforms as transforms
from get_image_from_folder import get_images_from_folder
from calculate_similarities import calculate_color_similarity, calculate_cosine_similarity, calculate_color_histogram
from render_best_outfits import render_best_outfit_skirts
from get_id import get_id

# Load the pre-trained ResNet model with 18 layers
resnet18 = models.resnet18(pretrained=True)
# Alternatively, you can load other versions of ResNet, such as resnet34, resnet50, etc.

# Set the model to evaluation mode
resnet18.eval()




def shirt_and_skirts_only(shirts_folder_id, skirts_folder_id):
    #Store the data for the best matches in a list
    best_matches=[]
    #Get images for the skirts and shirts
    shirt_images, shirt_ids=get_images_from_folder(shirts_folder_id)
    skirts_images, skirt_ids=get_images_from_folder(skirts_folder_id)
    #Preprocess images and convert to tensors
    shirt_tensors=[preprocess_image(image) for image in shirt_images]
    skirt_tensors=[preprocess_image(image) for image in skirts_images]
    #Extract features from tensors using resnet18
    with torch.no_grad():
        features_list1=[resnet18(input_tensor).squeeze().numpy() for input_tensor in shirt_tensors]
        features_list2=[resnet18(input_tensor).squeeze().numpy() for input_tensor in skirt_tensors]
    #Iterate through every component of wardrobe and find similarities
    for i in range(len(shirt_images)):
        #Get color histogram (H,S,V format) for shirt images
        hist1 = calculate_color_histogram(shirt_images[i])
        for j in range(len(skirts_images)):
            #Get color histogram for skirts
            hist2 = calculate_color_histogram(skirts_images[j])
            #Calculate cosine similarity between features
            feature_similarity = calculate_cosine_similarity(features_list1[i], features_list2[j])
            #Calculate color similarity
            color_similarity  = calculate_color_similarity(hist1, hist2)
            #Final similarity score is 50% of feature similarity and 50% of color similarity
            final_similarity_score = 0.5 * feature_similarity + 0.5 * color_similarity
             #Append best matches
            best_matches.append([i,j,final_similarity_score])
    best_matches.sort(key=lambda x: x[2], reverse=True)
    #Return best five outfits
    best_match_five=best_matches[:5]
    #Get file id of images of outfits
    recommended_skirts=get_id(best_match_five, [shirt_ids, skirt_ids])
    #Render the best outfit using matplotlib
    render_best_outfit_skirts(skirts_images, skirt_ids, shirt_images, shirt_ids, recommended_skirts)
    #Return ids of outfit combinations
    return recommended_skirts



def shirts_and_skirts_with_bags_and_shoes(shirts_folder_id, skirts_folder_id, bags_folder_id, shoes_folder_id):
    best_matches=[]
    shirt_images, shirt_ids=get_images_from_folder(shirts_folder_id)
    skirts_images, skirts_ids=get_images_from_folder(skirts_folder_id)
    bags_images, bags_ids=get_images_from_folder(bags_folder_id)
    shoes_images, shoes_ids=get_images_from_folder(shoes_folder_id)
    shirt_tensors=[preprocess_image(image) for image in shirt_images]
    trouser_tensors=[preprocess_image(image) for image in skirts_images]
    bag_tensors=[preprocess_image(image) for image in bags_images]
    shoe_tensors=[preprocess_image(image) for image in shoes_images]
    with torch.no_grad():
        features_list1=[resnet18(input_tensor).squeeze().numpy() for input_tensor in shirt_tensors]
        features_list2=[resnet18(input_tensor).squeeze().numpy() for input_tensor in trouser_tensors]
        features_list3=[resnet18(input_tensor).squeeze().numpy() for input_tensor in bag_tensors]
        features_list4=[resnet18(input_tensor).squeeze().numpy() for input_tensor in shoe_tensors]
    for i in range(len(shirt_images)):
        hist1 = calculate_color_histogram(shirt_images[i])
        for j in range(len(skirts_images)):
            hist2 = calculate_color_histogram(skirts_images[j])
            feature_similarity = calculate_cosine_similarity(features_list1[i], features_list2[j])
            color_similarity  = calculate_color_similarity(hist1, hist2)
            shirt_trouser_similarity_score = 0.5 * feature_similarity + 0.5 * color_similarity
            for k in range(len(shoes_images)):
                hist3 = calculate_color_histogram(shoes_images[k])
                trouser_shoe_similarity = calculate_cosine_similarity(features_list2[j], features_list4[k])
                color_trouser_shoe_similarity  = calculate_color_similarity(hist2, hist3)
                final_trouser_shoe_similarity = 0.5 * trouser_shoe_similarity + 0.5 * color_trouser_shoe_similarity
                for l in range(len(bags_images)):
                    hist4 = calculate_color_histogram(bags_images[k])
                    bag_shoe_similarity = calculate_cosine_similarity(features_list3[k], features_list4[l])
                    color_bag_shoe_similarity  = calculate_color_similarity(hist3, hist4)
                    final_bag_shoe_similarity = 0.5 * bag_shoe_similarity + 0.5 * color_bag_shoe_similarity
                    final_similarity_score= 0.4*shirt_trouser_similarity_score + 0.2 * final_bag_shoe_similarity + 0.4*final_trouser_shoe_similarity
                    best_matches.append([i,j,k,l,final_similarity_score])
    best_matches.sort(key=lambda x: x[4], reverse=True)
    best_match_five=best_matches[:5]
    recommended_skirts=get_id(best_match_five, [shirt_ids, skirts_ids, bags_ids, shoes_ids])
    render_best_outfit_skirts(skirts_images, skirts_ids, shirt_images, shirt_ids, recommended_skirts, bags_images, bags_ids, shoes_images, shoes_ids)
    return recommended_skirts



def shirts_and_skirts_with_bags(shirts_folder_id, skirts_folder_id, bags_folder_id):
    best_matches=[]
    shirt_images, shirt_ids=get_images_from_folder(shirts_folder_id)
    skirts_images, skirts_ids=get_images_from_folder(skirts_folder_id)
    bags_images, bags_ids=get_images_from_folder(bags_folder_id)
    shirt_tensors=[preprocess_image(image) for image in shirt_images]
    trouser_tensors=[preprocess_image(image) for image in skirts_images]
    bag_tensors=[preprocess_image(image) for image in bags_images]
    with torch.no_grad():
        features_list1=[resnet18(input_tensor).squeeze().numpy() for input_tensor in shirt_tensors]
        features_list2=[resnet18(input_tensor).squeeze().numpy() for input_tensor in trouser_tensors]
        features_list3=[resnet18(input_tensor).squeeze().numpy() for input_tensor in bag_tensors]
    for i in range(len(shirt_images)):
        hist1 = calculate_color_histogram(shirt_images[i])
        for j in range(len(skirts_images)):
            hist2 = calculate_color_histogram(skirts_images[j])
            feature_similarity = calculate_cosine_similarity(features_list1[i], features_list2[j])
            color_similarity  = calculate_color_similarity(hist1, hist2)
            shirt_trouser_similarity_score = 0.5 * feature_similarity + 0.5 * color_similarity
            for k in range(len(bags_images)):
                hist3 = calculate_color_histogram(bags_images[k])
                trouser_bag_feature_similarity = calculate_cosine_similarity(features_list2[j], features_list3[k])
                trouser_bag_color_similarity  = calculate_color_similarity(hist2, hist3)
                trouser_bag_similarity_score = 0.5 * trouser_bag_feature_similarity + 0.5 * trouser_bag_color_similarity
                final_score=0.6* shirt_trouser_similarity_score + 0.4 * trouser_bag_similarity_score
                best_matches.append([i,j,k,final_score])
    best_matches.sort(key=lambda x: x[3], reverse=True)
    best_match_five=best_matches[:5]
    recommended_skirts=get_id(best_match_five, [shirt_ids, skirts_ids, bags_ids])
    render_best_outfit_skirts(skirts_images, skirts_ids, shirt_images, shirt_ids, recommended_skirts, bags_images, bags_ids)    
    return recommended_skirts



def shirts_and_skirts_with_shoes(shirts_folder_id, skirts_folder_id, shoes_folder_id):
    best_matches=[]
    shirt_images, shirt_ids=get_images_from_folder(shirts_folder_id)
    skirts_images, skirts_ids=get_images_from_folder(skirts_folder_id)
    shoes_images, shoes_ids=get_images_from_folder(shoes_folder_id)
    shirt_tensors=[preprocess_image(image) for image in shirt_images]
    trouser_tensors=[preprocess_image(image) for image in skirts_images]
    shoe_tensors=[preprocess_image(image) for image in shoes_images]
    with torch.no_grad():
        features_list1=[resnet18(input_tensor).squeeze().numpy() for input_tensor in shirt_tensors]
        features_list2=[resnet18(input_tensor).squeeze().numpy() for input_tensor in trouser_tensors]
        features_list3=[resnet18(input_tensor).squeeze().numpy() for input_tensor in shoe_tensors]
    for i in range(len(shirt_images)):
        hist1 = calculate_color_histogram(shirt_images[i])
        for j in range(len(skirts_images)):
            hist2 = calculate_color_histogram(skirts_images[j])
            feature_similarity = calculate_cosine_similarity(features_list1[i], features_list2[j])
            color_similarity  = calculate_color_similarity(hist1, hist2)
            shirt_trouser_similarity_score = 0.5 * feature_similarity + 0.5 * color_similarity
            for k in range(len(shoes_images)):
                hist3 = calculate_color_histogram(shoes_images[k])
                trouser_bag_feature_similarity = calculate_cosine_similarity(features_list2[j], features_list3[k])
                trouser_bag_color_similarity  = calculate_color_similarity(hist2, hist3)
                trouser_bag_similarity_score = 0.5 * trouser_bag_feature_similarity + 0.5 * trouser_bag_color_similarity
                final_score=0.6* shirt_trouser_similarity_score + 0.4 * trouser_bag_similarity_score
                best_matches.append([i,j,k,final_score])
    best_matches.sort(key=lambda x: x[3], reverse=True)
    best_match_five=best_matches[:5]
    recommended_skirts=get_id(best_match_five, [shirt_ids, skirts_ids, shoes_ids])
    render_best_outfit_skirts(skirts_images, skirts_ids, shirt_images, shirt_ids, recommended_skirts, shoes_images, shoes_ids)
    return recommended_skirts

def get_skirt_recommendations(skirts=None, shirts=None, shoes=None, bags=None):
    skirt_list=[]
    if skirts is not None and skirts is not None and shoes is not None and bags is not None:
        skirt_list=shirts_and_skirts_with_bags_and_shoes(shirts_folder_id=shirts, skirts_folder_id=skirts, bags_folder_id=bags, shoes_folder_id=shoes)
    elif skirts is not None and skirts is not None and bags is not None:
        skirt_list=shirts_and_skirts_with_bags(shirts_folder_id=shirts, skirts_folder_id=skirts, bags_folder_id=bags)
    elif skirts is not None and skirts is not None and shoes is not None:
        skirt_list=shirts_and_skirts_with_shoes(shirts_folder_id=shirts, skirts_folder_id=skirts, shoes_folder_id=shoes)
    elif skirts is not None and shirts is not None:
        skirt_list=shirt_and_skirts_only(shirts_folder_id=shirts, skirts_folder_id=skirts)
    return skirt_list
    

#print(shirt_and_skirts_only('1betg3ShGBh7IWmdmrY5HFs0G7WAqjffK', '1lgStI-JQBItFBwI3Zgp1tt656S3EK9y2'))
print(shirts_and_skirts_with_bags_and_shoes('1lgStI-JQBItFBwI3Zgp1tt656S3EK9y2','1betg3ShGBh7IWmdmrY5HFs0G7WAqjffK', '1i46TTlXXD5eSQPNMxkBb7QQgGdOJfOIM', '18EMeELFWs-IzVcwKe1PyMxMmRolz_86e' ))
#print(shirts_and_skirts_with_bags('1betg3ShGBh7IWmdmrY5HFs0G7WAqjffK', '1lgStI-JQBItFBwI3Zgp1tt656S3EK9y2','1i46TTlXXD5eSQPNMxkBb7QQgGdOJfOIM'))
print(shirts_and_skirts_with_shoes('1betg3ShGBh7IWmdmrY5HFs0G7WAqjffK', '1lgStI-JQBItFBwI3Zgp1tt656S3EK9y2','18EMeELFWs-IzVcwKe1PyMxMmRolz_86e'))