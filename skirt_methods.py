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
from get_image_from_folder import get_images_from_drive
from calculate_similarities import calculate_color_similarity, calculate_cosine_similarity, calculate_color_histogram


# Load the pre-trained ResNet model with 18 layers
resnet18 = models.resnet18(pretrained=True)
# Alternatively, you can load other versions of ResNet, such as resnet34, resnet50, etc.

# Set the model to evaluation mode
resnet18.eval()


def get_id(best_matches, image_id):
    final_lis=[]
    for i in best_matches:
            lis=[]
            for j in range(len(i)-1):
                  a=image_id[j][i[j]]
                  lis.append(a)
            final_lis.append(lis)
    return final_lis



def shirt_and_skirts_only(shirts_folder_id, skirts_folder_id):
    best_matches=[]
    shirt_images, shirt_ids=get_images_from_drive(shirts_folder_id)
    skirts_images, skirt_ids=get_images_from_drive(skirts_folder_id)
    shirt_tensors=[preprocess_image(image) for image in shirt_images]
    skirt_tensors=[preprocess_image(image) for image in skirts_images]
    with torch.no_grad():
        features_list1=[resnet18(input_tensor).squeeze().numpy() for input_tensor in shirt_tensors]
        features_list2=[resnet18(input_tensor).squeeze().numpy() for input_tensor in skirt_tensors]
    for i in range(len(shirt_images)):
        hist1 = calculate_color_histogram(shirt_images[i])
        for j in range(len(skirts_images)):
            hist2 = calculate_color_histogram(skirts_images[j])
            feature_similarity = calculate_cosine_similarity(features_list1[i], features_list2[j])
            color_similarity  = calculate_color_similarity(hist1, hist2)
            final_similarity_score = 0.5 * feature_similarity + 0.5 * color_similarity
            best_matches.append([i,j,final_similarity_score])
    best_matches.sort(key=lambda x: x[2], reverse=True)
    best_match_five=best_matches[:5]
    recommended_skirts=get_id(best_match_five, [shirt_ids, skirt_ids])
    return recommended_skirts



def shirts_and_skirts_with_bags_and_shoes(shirts_folder_id, skirts_folder_id, bags_folder_id, shoes_folder_id):
    best_matches=[]
    shirt_images, shirt_ids=get_images_from_drive(shirts_folder_id)
    skirts_images, skirts_ids=get_images_from_drive(skirts_folder_id)
    bags_images, bags_ids=get_images_from_drive(bags_folder_id)
    shoes_images, shoes_ids=get_images_from_drive(shoes_folder_id)
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
    recommended_shirts=get_id(best_match_five, [shirt_ids, skirts_ids, shoes_ids, bags_ids])
    return recommended_shirts



def shirts_and_skirts_with_bags(shirts_folder_id, skirts_folder_id, bags_folder_id):
    best_matches=[]
    shirt_images, shirt_ids=get_images_from_drive(shirts_folder_id)
    skirts_images, skirts_ids=get_images_from_drive(skirts_folder_id)
    bags_images, bags_ids=get_images_from_drive(bags_folder_id)
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
    recommended_shirts=get_id(best_match_five, [shirt_ids, skirts_ids, bags_ids])

    return recommended_shirts



def shirts_and_skirts_with_shoes(shirts_folder_id, skirts_folder_id, shoes_folder_id):
    best_matches=[]
    shirt_images, shirt_ids=get_images_from_drive(shirts_folder_id)
    skirts_images, skirts_ids=get_images_from_drive(skirts_folder_id)
    shoes_images, shoes_ids=get_images_from_drive(shoes_folder_id)
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
    recommended_shirts=get_id(best_match_five, [shirt_ids, skirts_ids, shoes_ids])
    return recommended_shirts

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
    
