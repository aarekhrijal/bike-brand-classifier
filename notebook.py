# Install libraries
!pip install -Uqq ddgs fastai

# Imports
from ddgs import DDGS
from fastcore.all import *
from fastai.vision.all import *
import time

def search_images(term, max_images=100):
    return L(DDGS().images(term, max_results=max_images)).itemgot('image')

# Download images
bike_brands = ['honda bike', 'yamaha bike', 'ktm bike']
path = Path('bikes')

for brand in bike_brands:
    dest = (path/brand)
    dest.mkdir(exist_ok=True, parents=True)
    urls = search_images(brand, max_images=100)
    download_images(dest, urls=urls)
    time.sleep(3)
    print(f'Done: {brand}')

# Clean corrupted images
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f'Failed: {len(failed)}')

# Create dataloaders
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, seed=42,
                                    item_tfms=Resize(224))

# Train model
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(5)

# Check results
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# Export model
learn.export('/kaggle/working/bike_classifier.pkl')
