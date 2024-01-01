import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_combined_grid(folder_path, epochs, images_per_epoch):
    cols = images_per_epoch  
    rows = epochs 

    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    

    
    fig.subplots_adjust(wspace=0.05, hspace=0.2)

    for epoch in range(epochs):
        for img_num in range(images_per_epoch):
            ax = axes[epoch, img_num]
            file_name = f'image_at_epoch_{epoch}_num_{img_num}.png'
            img_path = os.path.join(folder_path, file_name)
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                ax.imshow(img)
          
            if epoch == 0:
                ax.set_title(f'Img {img_num + 1}', fontsize=10)
            
          
            if img_num == 0:
                ax.set_ylabel(f'Epoch {epoch + 1}', fontsize=10, rotation=0, labelpad=30)

    plt.tight_layout()
    plt.show()


image_folder = 'generated_images'
num_epochs = 10  
images_per_epoch = 4  

create_combined_grid(image_folder, num_epochs, images_per_epoch)
