import pandas as pd
import numpy as np

import streamlit as st
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder



# defining the function which will make the prediction using 
# the data which the user inputs
def prediction(indgred1,indgred2,indgred3,indgred4):  
    indgred=[indgred1,indgred2,indgred3,indgred4]
    model = tf.keras.models.load_model("cooking_deep.h5")
    indgred_sentence=[",".join(sent for sent in indgred)]
    vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=None, output_mode='int', output_sequence_length=65,split=lambda x: tf.strings.split(x, ','), standardize=lambda x: tf.strings.lower(x))
    vectorization_layer.adapt(indgred_sentence)
    vectorizer = tf.keras.models.Sequential()
    vectorizer.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    vectorizer.add(vectorization_layer)
    indgred_test=vectorizer.predict(indgred_sentence)
    new_predictions=model.predict(indgred_test)
    le = LabelEncoder()
    indgred_cuisine= np.argmax(new_predictions, axis=1)
    map={6:'greek',16:'southern_us',4:'filipino',7:'indian',10:'jamaican',17:'spanish',9:'italian',13:'mexican',3:'chinese',1:'british',18:'thai',19:'vietnamese',2:'cajun_creole',0:'brazilian',5:'french',11:'japanese',8:'irish',12:'korean',14:'moroccan',15:'russian'}
    return map[int(indgred_cuisine)]
      
  
# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    st.title("Cuisine Classification")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:#ADD8E6;padding:13px">
    <h1 style ="color:black;text-align:center;">Let's find the Cuisines</h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    indgred1= st.text_input("indgedients1")
    indgred2= st.text_input("indgedients2")
    indgred3= st.text_input("indgedients3")
    indgred4= st.text_input("indgedients4")
    result=""
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
   # sample()
    if st.button("Predict"):
        result = prediction(indgred1,indgred2,indgred3,indgred4)
    st.success('The Cuisine is {}'.format(result))
     
if __name__=='__main__':
    main()

