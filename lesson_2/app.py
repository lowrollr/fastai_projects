import gradio as gr
from fastai.vision.all import load_learner

model = load_learner('export.pkl')

categories = tuple(sorted(['eagle', 'hawk', 'falcon', 'owl', 'vulture']))

def classify_bird(img):
    pred, idx, probs = model.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.inputs.Image(shape=(256, 256))
label = gr.outputs.Label(num_top_classes=5)

examples = ['./example_images/' + e for e in ['eagle.jpg', 'hawk.jpg', 'falcon.jpg', 'owl.jpg', 'vulture.jpg']]

iface = gr.Interface(
    fn=classify_bird,
    inputs=image, 
    outputs=label, 
    examples=examples, 
    title='Birds of Prey Classifier', 
    description='Classifies images as one of 5 classes: eagle, hawk, falcon, owl, vulture',
    allow_flagging='never'
)
iface.launch()