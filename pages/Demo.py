import streamlit as st
import torch
from utils import *
from PIL import Image, ImageDraw, ImageFont, ImageColor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#models checkpoint
srgan_checkpoint = "./models/checkpoint_srgan.pth_25.tar"
srresnet_checkpoint = "./models/checkpoint_srresnet_129.pth.tar"
#load models
srgan = torch.load(srgan_checkpoint,map_location=torch.device(device))['generator']
srgan.eval()
srresnet = torch.load(srresnet_checkpoint,map_location=torch.device(device))['model']
srresnet.eval()

def is_image(file):
    # Check if the file type is an image
    return file.type.startswith('image/')

def superresolve_srgan(lr_img):
    sr_img_srgan = srgan(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')
    st.image(sr_img_srgan,caption="SRGAN")

def superresolve_srresnet(lr_img):
    sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')
    st.image(sr_img_srresnet,caption="SRResNet")


def main():
    st.markdown('<h1 align="center">Image Super Resolution</h1>',unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a file",type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        #check if it is an image
        if is_image(uploaded_file):
            col1,col2,col3 = st.columns(3)
            lr_img = Image.open(uploaded_file,mode='r')
            lr_img = lr_img.convert('RGB')
            with col1:
                st.markdown("Input Image")
                st.image(lr_img, caption="Uploaded Image")
            with col2:
                choice=st.selectbox("Select Model",("SRResNet","SRGAN"))   
                convertbtn=st.button(label="Convert to High Quality", key="btn")
            with col3:
                st.markdown("SuperResolved Image")
                if convertbtn:
                    if choice=="SRGAN":
                        superresolve_srgan(lr_img)
                    elif choice=="SRResNet":
                        superresolve_srresnet(lr_img)
        else:
            st.error("Please upload a valid image file (JPEG or PNG).")

if __name__ == "__main__":
    main()