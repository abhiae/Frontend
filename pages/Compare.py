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

def get_lr_image(hr_img):
    lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
                       Image.BICUBIC)
    return lr_img

def superresolve_bicubic(lr_img):
    bicubic_img = lr_img.resize((lr_img.width*4, lr_img.height*4), Image.BICUBIC)
    st.image(bicubic_img,caption="Bicubic")

    
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
        if is_image(uploaded_file):
            hr_img = Image.open(uploaded_file,mode='r')
            hr_img = hr_img.convert('RGB')
            lr_img = get_lr_image(hr_img)
            st.image(lr_img,caption = "LR Image")
            st.markdown("Super Resolved Images")
            col1,col2 = st.columns(2)
            with col1:
                superresolve_bicubic(lr_img)
                superresolve_srresnet(lr_img)
            with col2:
                superresolve_srgan(lr_img)
                st.image(hr_img,caption="Original")
    else:
        st.error("Please upload a valid image file (JPEG or PNG).")

if __name__ == "__main__":
    main()