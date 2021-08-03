import intro
import streamlit as st

import visual
import individual

try:
    st.set_page_config(layout="wide")
    # Pages = {"Login": Login, "My Portfolio": myPortfolio, "Stock": stock, "Portfolio": portfolio, "Prediction": prediction}
    # image5 = Image.open("ccprojimg7.jpg")
    # st.sidebar.image(image5, use_column_width=True)
    intro.app()
    st.write("""   """)
    Pages = {"Home": visual, "Relational-analysis": individual}
    # st.sidebar.title('Navigation')

    selection = st.sidebar.selectbox("Menu", list(Pages.keys()))
    st.sidebar.markdown("    ")
    st.sidebar.write("Please select the date range to let us know for what period of time do you want to see the cypto closing price")
    st.sidebar.markdown("    ")
    page = Pages[selection]
    page.app()



except:
    pass
