import os
import uuid
import streamlit as st
from pathlib import Path
from inference import Infernece


if "save_pth" not in st.session_state:
    st.session_state.save_pth = None

if "key_words" not in st.session_state:
    st.session_state.key_words = []

if "process" not in st.session_state:
    st.session_state.process = False


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "pdf", "txt"}


def get_file_extension(filename):
    return filename.split(".")[-1].lower()


st.set_page_config(layout="wide")
st.markdown(
    """
    <div style="text-align: center;">
        <span style="font-size: 24px; color: orange;">
            Satellite Image Super Resolution
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

root_pth = str(Path(__file__).parent)
Path(root_pth).joinpath("ui_uploads").mkdir(exist_ok=True, parents=True)

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown(
        """
    <div style="text-align: left;">
        <span style="font-size: 14px; color: orange;">
            Upload your image:
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("", type=None)

    if uploaded_file is not None:

        file_extension = get_file_extension(uploaded_file.name)

        if file_extension in ALLOWED_EXTENSIONS:

            unique_id = str(uuid.uuid4())

            file_path = os.path.join(
                root_pth, "ui_uploads", f"{unique_id}.{file_extension}"
            )

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.save_pth = file_path

            if st.session_state.save_pth:
                if st.button("Process"):
                    st.session_state.process = True

        else:
            st.error("File type not allowed. Please upload a valid file.")


with col2:
    if st.session_state.process:

        inference = Infernece()
        hr, lr, predicted_lr, val_results = inference(st.session_state.save_pth)

        ssim = round(val_results["ssim_total"], 3)
        psnr = round(val_results["psnr"], 3)
        mse = round(val_results["mse"].detach().item(), 3)

        # # print ssim,psnr and mse
        # st.markdown(f""":orange[**Metrics of Super Resolution:**]""")

        # st.markdown(f""":green[**SSIM**] = :red[**{ssim}**]""")
        # st.markdown(f""":green[**PSNR**] = :red[**{psnr}**]""")
        # st.markdown(f""":green[**MSE**] = :red[**{mse}**]""")

        # st.markdown(
        #     """
        #     <div style="text-align: center;">
        #         <span style="font-size: 20px; color: orange;">
        #             <strong>Metrics of Super Resolution</strong>
        #         </span>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )

        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <table style="border-collapse: collapse; width: 50%;">
                    <tr>
                        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2; font-size: 20px; color: green;">Metric</th>
                        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2; font-size: 20px; color: green;">Value</th>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-size: 20px; color: green;">SSIM</td>
                        <td style="border: 1px solid #ddd; padding: 8px; font-size: 20px; color: red;">{ssim}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-size: 20px; color: green;">PSNR</td>
                        <td style="border: 1px solid #ddd; padding: 8px; font-size: 20px; color: red;">{psnr}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-size: 20px; color: green;">MSE</td>
                        <td style="border: 1px solid #ddd; padding: 8px; font-size: 20px; color: red;">{mse}</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1x, c2x, c3x = st.columns(3)
        with c1x:
            st.markdown(""":red[**Low Resolution Image**]""")
            st.markdown(f""":orange[**Width, Height**] = :red[**{lr.size}**]""")

            st.image(lr)

        with c2x:
            st.markdown(""":red[**Up-scaled Image**]""")
            st.markdown(
                f""":orange[**Width, Height**] = :red[**{predicted_lr.size}**]"""
            )
            st.image(predicted_lr)

            with c3x:
                st.markdown(""":red[**Original Image:**]""")
                st.markdown(f""":orange[**Width, Height**] = :red[**{hr.size}**]""")
                st.image(hr)
