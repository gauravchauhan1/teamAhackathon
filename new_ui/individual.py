import streamlit as st
import datetime
import matplotlib.pyplot as plt
import investpy
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def app():
    st.markdown("<h1 style='text-align: center;'>Hey Investor!</h1>", unsafe_allow_html=True)
    st.markdown('''
    
    - We know charts are boring but we wanted to give you some control! So following are the charts that lets you understand how your investment might be affected by someone else's investment.
    - Don't worry, rest assured, Well guide you through, so that you make the best out of it.
    ''')
    start_date = st.sidebar.date_input("Start date", datetime.date(2021, 1, 1))
    end_date = st.sidebar.date_input("End date", datetime.date(2021, 7, 31))
    data = investpy.get_crypto_historical_data(crypto='bitcoin',
                                               from_date=str(start_date.strftime('%d/%m/%Y')),
                                               to_date=str(end_date.strftime('%d/%m/%Y')))

    data1 = investpy.get_crypto_historical_data(crypto='dogecoin',
                                                from_date=str(start_date.strftime('%d/%m/%Y')),
                                                to_date=str(end_date.strftime('%d/%m/%Y')))

    data2 = investpy.get_crypto_historical_data(crypto='ethereum',
                                                from_date=str(start_date.strftime('%d/%m/%Y')),
                                                to_date=str(end_date.strftime('%d/%m/%Y')))

    data3 = investpy.get_crypto_historical_data(crypto='tether',
                                                from_date=str(start_date.strftime('%d/%m/%Y')),
                                                to_date=str(end_date.strftime('%d/%m/%Y')))

    # Retrieving tickers data

    tickerSymbol = True  # Select ticker symbol

    training_data = data.drop(['Currency'], axis=1)

    if tickerSymbol:
        df = pd.DataFrame({'BTC': data.Close,
                           'DOG': data1.Close,
                           'ETH': data2.Close,
                           'USDT': data3.Close})
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax2 = ax1.twinx()
        rspine = ax2.spines['right']
        rspine.set_position(('axes', 1.15))
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        fig.subplots_adjust(right=0.7)

        # df['BTC'].plot(ax=ax1, style='b-')
        # df['DOG'].plot(ax=ax2, style='g-')
        df['BTC'].plot(ax=ax1, style='b-')
        df['DOG'].plot(ax=ax1, style='r-', secondary_y=True)
        df['USDT'].plot(ax=ax2, style='g-')

        # legend
        ax2.legend([ax1.get_lines()[0],
                    ax1.right_ax.get_lines()[0],
                    ax2.get_lines()[0]],
                   ['BTC', 'DOG', 'USDT'])
        st.write("**Visualize Relative Changes of Closing Prices**")
        st.pyplot(fig)

        # Compute the correlation matrix
        corr = df.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(10, 10))

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, annot=True, fmt='.4f', mask=mask, center=0, square=True, linewidths=.5)
        st.write("**Effects of cryptocurrency on each other**")
        st.pyplot(f)

        df_return = df.apply(lambda x: x / x[0])
        df_return.plot(grid=True, figsize=(15, 10)).axhline(y=1, color="black", lw=2)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("**Profit/loss in 7 months**")
        st.pyplot()

        df_perc = df_return.tail(1) * 100
        ax = sns.barplot(data=df_perc)
        st.write("**Market value **")
        st.write(df_perc)

        budget = 2000  # USD
        df_coins = budget / df.head(1)

        ax = sns.barplot(data=df_coins)
        st.write("**Potential gains/loss per 2000 dollar**")
        st.write(df_coins)
        st.write("**Percentage Increase in 7 months**")
        st.pyplot()

        df_profit = df_return.tail(1) * budget

        ax = sns.barplot(data=df_profit)
        st.write("**Percentage Increase in 7 months**")
        st.write(df_profit)
        st.write("**How much money could have been made?**")
        st.pyplot()
