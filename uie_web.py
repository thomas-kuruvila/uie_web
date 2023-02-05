from termios import TAB1
import cv2
import numpy as np
import io
import streamlit as st


file_image = st.file_uploader("Upload the raw image")

if file_image is not None:
        numpy_img = np.asarray(bytearray(file_image.read()),dtype=np.uint8)
        cap = cv2.imdecode(numpy_img,cv2.IMREAD_COLOR)
        bins = st.number_input("Input the Number of bins:",min_value=0,max_value=300,value=256)

        print('PROCESSING LIVE FEED')

        st.text("Uploaded Image:")
        st.image(cap)

        frame = cap
        tab1,tab2 = st.tabs(["Algorithm-1","Algorithm-2"])

        with tab1:
        #bytes_image = file_image.getvalue()

                #frame=cv2.resize(frame,(500,500)) #resize the frame
                #cv2.imshow("ORIGINAL", frame) #shows unenhanced image

                img_inv = cv2.bitwise_not(frame)
                #cv2.imshow("INVERTED", img_inv)

                #Get the dark channel in inverted image

                #masked = frame - img_inv
                #cv2.imshow("Masked",masked)

                #individual 3 channel histogram equalization
                b,g,r=cv2.split(img_inv)

                ###BLUE CHANNEL
                b_flattened = b.flatten()
                b_hist = np.zeros(bins)
                for pix in b:
                        b_hist[pix] += 1
                cum_sum = np.cumsum(b_hist)
                norm = (cum_sum - cum_sum.min()) * 180
                # normalization of the pixel values
                n_ = cum_sum.max() - cum_sum.min()
                uniform_norm = norm / n_
                uniform_norm = uniform_norm.astype('int')

                # flat histogram
                b_eq = uniform_norm[b_flattened]
                # reshaping the flattened matrix to its original shape
                b_eq = np.reshape(a=b_eq, newshape=b.shape)
                b_eq=np.uint8(b_eq)


                ###GREEN CHANNEL
                g_flattened = g.flatten()
                g_hist = np.zeros(bins)
                for pix in g:
                        g_hist[pix] += 1

                cum_sum = np.cumsum(g_hist)
                norm = (cum_sum - cum_sum.min()) * 255
                # normalization of the pixel values
                n_ = cum_sum.max() - cum_sum.min()
                uniform_norm = norm / n_
                uniform_norm = uniform_norm.astype('int')

                # flat histogram
                g_eq = uniform_norm[g_flattened]
                # reshaping the flattened matrix to its original shape
                g_eq = np.reshape(a=g_eq, newshape=g.shape)
                g_eq=np.uint8(g_eq)


                ###RED CHANNEL
                r_flattened = r.flatten()
                r_hist = np.zeros(bins)
                for pix in r:
                        r_hist[pix] += 1

                cum_sum = np.cumsum(r_hist)
                norm = (cum_sum - cum_sum.min()) * 255
                # normalization of the pixel values
                n_ = cum_sum.max() - cum_sum.min()
                uniform_norm = norm / n_
                uniform_norm = uniform_norm.astype('int')

                # flat histogram
                r_eq = uniform_norm[r_flattened]
                # reshaping the flattened matrix to its original shape
                r_eq = np.reshape(a=r_eq, newshape=r.shape)
                r_eq=np.uint8(r_eq)

                image_eq=cv2.merge((b_eq,g_eq,r_eq))
                img1= cv2.bitwise_not(image_eq)
                #img1=image_eq

                st.text("Rectified Image")
                st.image(img1)

        with tab2:
                clipLimitVal = float(st.slider("Set Clip Limit",min_value=0,max_value=255,value=5))
                hsv_img = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
                    #shsv_img = cv2.bitwise_not(hsv_img)

                    #create 3 color channels
                h,s,v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]

                    #APPLY CLAHE
                tileGridSizeVal = st.selectbox("Set Grid Size",[(2,2),(4,4),(8,8),(12,12)])
                clahe = cv2.createCLAHE(clipLimit = clipLimitVal, tileGridSize = tileGridSizeVal)
                v = clahe.apply(v)
                #h = clahe.apply(h)
                    #s = clahe.apply(s)

                hsv_img = np.dstack((h,s,v))

                rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
                    #plt.imshow(rgb);

                    #img1= cv2.bitwise_not(rgb)
                st.text("Rectified Image")
                st.image(rgb)
