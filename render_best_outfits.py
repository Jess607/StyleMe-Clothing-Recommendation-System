import numpy as np
import matplotlib.pyplot as plt




#Returns images of the best outfit for dresses
def render_best_outfit_dresses(dress_images, dress_ids, recommended_dresses, bags_images=None, bags_ids=None, shoes_images=None, shoes_ids=None):
    #Get number of components in wardrobe
    num=len(recommended_dresses[0])
    #List stores the numpy array of images
    list_of_items=[]
    #The first outfit is the best outfit recommended
    best_recommended_outfit=recommended_dresses[0]
    #Get index of out the id of the first dress
    best_dress=dress_ids.index(best_recommended_outfit[0])
    #Convert to numpy array
    img_array_dress=np.array(dress_images[best_dress])
    #Append to list of items
    list_of_items.append(img_array_dress)
    #If the wardrobe has a bags in it
    if bags_images is not None:
        best_bag=bags_ids.index(best_recommended_outfit[1])
        img_array_bag=np.array(bags_images[best_bag])
        list_of_items.append(img_array_bag)
    #If the wardrobe has shoes in it 
    if shoes_images is not None:
        best_shoe=shoes_ids.index(best_recommended_outfit[2]) 
        img_array_shoe=np.array(shoes_images[best_shoe])
        list_of_items.append(img_array_shoe)
        #Iterate through the recommended outfit and render using matplotlib
    for i in range(len(recommended_dresses[0])):
        plt.subplot(1,num,i+1)
        plt.imshow(list_of_items[i])
        plt.axis('off')
    plt.show()

#Returns images for best outfit combinations with skirts
def render_best_outfit_skirts(skirt_images, skirt_ids, shirt_images, shirt_ids, recommended_skirts, bags_images=None, bags_ids=None, shoes_images=None, shoes_ids=None):
    num=len(recommended_skirts[0])
    list_of_items=[]
    best_recommended_outfit=recommended_skirts[0]
    best_skirt=skirt_ids.index(best_recommended_outfit[1])
    best_shirt=shirt_ids.index(best_recommended_outfit[0])
    img_array_skirt=np.array(skirt_images[best_skirt])
    img_array_shirt=np.array(shirt_images[best_shirt])
    list_of_items.append(img_array_skirt)
    list_of_items.append(img_array_shirt)
    if bags_images is not None:
        best_bag=bags_ids.index(best_recommended_outfit[2])
        img_array_bag=np.array(bags_images[best_bag])
        list_of_items.append(img_array_bag)
    if shoes_images is not None:
        best_shoe=shoes_ids.index(best_recommended_outfit[-1]) 
        img_array_shoe=np.array(shoes_images[best_shoe])
        list_of_items.append(img_array_shoe)
    for i in range(len(recommended_skirts[0])):
        plt.subplot(1,num,i+1)
        plt.imshow(list_of_items[i])
        plt.axis('off')
    plt.show()


#Returns images for best outfit combinations with pants
def render_best_outfit_pants(pants_images, pants_ids, shirt_images, shirt_ids, recommended_pants, bags_images=None, bags_ids=None, shoes_images=None, shoes_ids=None):
    num=len(recommended_pants[0])
    list_of_items=[]
    best_recommended_outfit=recommended_pants[0]
    best_pant=pants_ids.index(best_recommended_outfit[1])
    best_shirt=shirt_ids.index(best_recommended_outfit[0])
    img_array_pant=np.array(pants_images[best_pant])
    img_array_shirt=np.array(shirt_images[best_shirt])
    list_of_items.append(img_array_pant)
    list_of_items.append(img_array_shirt)
    if bags_images is not None:
        best_bag=bags_ids.index(best_recommended_outfit[2])
        img_array_bag=np.array(bags_images[best_bag])
        list_of_items.append(img_array_bag)
    if shoes_images is not None:
        best_shoe=shoes_ids.index(best_recommended_outfit[-1]) 
        img_array_shoe=np.array(shoes_images[best_shoe])
        list_of_items.append(img_array_shoe)
    for i in range(len(recommended_pants[0])):
        plt.subplot(1,num,i+1)
        plt.imshow(list_of_items[i])
        plt.axis('off')
    plt.show()