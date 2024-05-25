import gradio as gr
from predict import predict


title = 'Classifying Names with a Character-Level RNN'

article = '''
Basic character-level Recurrent Neural Network (RNN) to classify word.\n
The model was trained on a few thousand surnames from 18 languages of origin, and predict which language a name is from based on the spelling.
'''

gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=1, label="Input surname"),
    outputs="text",
    title=title,
    article=article,
    examples=[["Hazaki"], ["Rossi"], ['Ivanov'], ['Ronaldo']],
    allow_flagging="never"
).launch()