# import pathlib
# import sys
# sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
# print('**********************', sys.path)

import streamlit as st
import requests
# import json

# import App.config as config

# import sys
# from streamlit.web import cli as stcli


st.title("Image captioning web-app")

image = st.file_uploader("Upload the image")

if image is not None:
    st.image(image.getvalue())

if st.button("Caption"):

    # create dictionary of file
    file = {'file': image.getvalue()}

    response = requests.post("http://backend:8001/caption", files = file)
    st.write(response.json())


## For saving the uploaded file and then accessing it:

# if image is not None:
#     with open(config.root_path/'Data/Uploaded_images/image.jpg', 'wb') as file:
#         file.write(image.getbuffer())
#         file.close()

#     img = plt.imread(config.root_path/'Data/Uploaded_images/image.jpg')
#     st.image(img)

# if st.button("Caption"):
#     img_path = {'path': str(config.root_path/'Data/Uploaded_images/image.jpg')}
#     st.write(img_path)
#     response = requests.post("http://localhost:8001/caption", data = json.dumps(img_path))
#     st.write(response.json())


# if __name__ == '__main__':
#     sys.argv = ["streamlit", "run", "F:\Git repository\Image captioning\App\Streamlit\streamlit_UI.py"]
#     sys.exit(stcli.main())