import streamlit as st
import os
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

def downsample(hr_img):
    hr_img = Image.open(hr_img, mode="r")
    hr_img = hr_img.convert('RGB')
    lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
                       Image.BICUBIC)
    return hr_img,lr_img

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

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            images.append(filename)
    return images


def main():
    st.markdown('<h2 align="center">Image Super Resolution</h1>',unsafe_allow_html=True)
    # load images from the folder into a array
    folder_path="./images"
    images_list = load_images_from_folder(folder_path)

    selected_image = st.selectbox("Select an Image",images_list)

    if selected_image:
        image_path = os.path.join(folder_path,selected_image)

        hr_img,lr_img = downsample(image_path)
        st.markdown("Input Image")
        st.image(lr_img,caption = "LR Image")

        st.markdown("Super Resolved Images")


        col1,col2 = st.columns(2)
        with col1:
            superresolve_bicubic(lr_img)
            superresolve_srresnet(lr_img)
        with col2:
            superresolve_srgan(lr_img)
            st.image(hr_img,caption="Original")
            
if __name__ == "__main__":
    main()