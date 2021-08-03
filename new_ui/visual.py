import streamlit as st
import investpy
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import seaborn as sns


def app():

    st.markdown("<h1 style='text-align: center;'>CRYPTOCURRENCY CLOSING PRICES vs PREDICTED PRICES</h1><br />", unsafe_allow_html=True)
    st.markdown('''
    - **These are the graphs that show the current trend of Cypto Closing Prices VS our predictions that the system has learnt.** ''')
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


    training_data = data.drop(['Currency'], axis=1)

    """df = pd.DataFrame({'BTC': data.Close,
                           'DOG': data1.Close,
                           'ETH': data2.Close,
                           'USDT': data3.Close})

    st.line_chart(data  =  df)"""

    col1, col2 = st.beta_columns(2)
    with col1:
         df = pd.DataFrame({'BTC': data.Close})
         st.line_chart(data = df)

    with col2:
        #MinMaxScaler is used to normalize the data
        scaler = MinMaxScaler()
        training_data = scaler.fit_transform(training_data)

        X_train = []
        Y_train = []
        training_data.shape[0]
        for i in range(60, training_data.shape[0]):
                X_train.append(training_data[i-60:i])
                Y_train.append(training_data[i,0])

        X_train, Y_train = np.array(X_train), np.array(Y_train)

        trained_model = tf.keras.models.load_model('team_A_model')

        part_60_days = data.tail(60)
        df= part_60_days.append(data, ignore_index = True)
        df = df.drop(['Currency'], axis = 1)
        inputs = scaler.transform(df)

        X_test = []
        Y_test = []
        for i in range (60, inputs.shape[0]):
            X_test.append(inputs[i-60:i])
            Y_test.append(inputs[i, 0])

        X_test, Y_test = np.array(X_test), np.array(Y_test)
        Y_pred = trained_model.predict(X_test)
        scale = 1/5.18164146e-05
        Y_test = Y_test*scale
        Y_pred = Y_pred*scale
        plt.figure(figsize=(14,5))
        plt.plot(Y_test, color = 'red', label = 'Real Bitcoin Price')
        plt.plot(Y_pred, color = 'green', label = 'Predicted Bitcoin Price')
        plt.title('Bitcoin Price Prediction using LSTM')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    col1, col2 = st.beta_columns(2)
    with col1:
         df = pd.DataFrame({'ETH': data2.Close})
         st.line_chart(data = df)
    with col2:
        """data2 = investpy.get_crypto_historical_data(crypto='ethereum',
                                               from_date='01/01/2021',
                                               to_date='31/07/2021')"""

        training_data = data2.drop(['Currency'], axis = 1)

        #MinMaxScaler is used to normalize the data
        scaler = MinMaxScaler()
        training_data = scaler.fit_transform(training_data)

        X_train = []
        Y_train = []
        training_data.shape[0]
        for i in range(60, training_data.shape[0]):
                X_train.append(training_data[i-60:i])
                Y_train.append(training_data[i,0])

        X_train, Y_train = np.array(X_train), np.array(Y_train)

        trained_model = tf.keras.models.load_model('team_A_model')

        part_60_days = data2.tail(60)
        df= part_60_days.append(data2, ignore_index = True)
        df = df.drop(['Currency'], axis = 1)
        inputs = scaler.transform(df)

        X_test = []
        Y_test = []
        for i in range (60, inputs.shape[0]):
            X_test.append(inputs[i-60:i])
            Y_test.append(inputs[i, 0])

        X_test, Y_test = np.array(X_test), np.array(Y_test)
        Y_pred = trained_model.predict(X_test)

        scale = 1/5.18164146e-05
        Y_test = Y_test*scale
        Y_pred = Y_pred*scale

        plt.figure(figsize=(14,5))
        plt.plot(Y_test, color = 'red', label = 'Real Ethereum Price')
        plt.plot(Y_pred, color = 'green', label = 'Predicted Ethereum Price')
        plt.title('Ethereum Price Prediction using LSTM')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        st.write("**Our model predictions for etheruem for future investment**")
        st.pyplot()


    col1, col2 = st.beta_columns(2)
    with col1:
        df = pd.DataFrame({'DOG': data1.Close})
        st.line_chart(data = df)


    with col2:
        """data1 = investpy.get_crypto_historical_data(crypto='dogecoin',
                                                    from_date=str(start_date.strftime('%d/%m/%Y')),
                                                    to_date=str(end_date.strftime('%d/%m/%Y')))"""
        training_data = data1.drop(['Currency'], axis = 1)

        #MinMaxScaler is used to normalize the data
        scaler = MinMaxScaler()
        training_data = scaler.fit_transform(training_data)

        X_train = []
        Y_train = []
        training_data.shape[0]
        for i in range(60, training_data.shape[0]):
                X_train.append(training_data[i-60:i])
                Y_train.append(training_data[i,0])

        X_train, Y_train = np.array(X_train), np.array(Y_train)

        trained_model = tf.keras.models.load_model('team_A_model')

        part_60_days = data1.tail(60)
        df= part_60_days.append(data1, ignore_index = True)
        df = df.drop(['Currency'], axis = 1)
        inputs = scaler.transform(df)


        X_test = []
        Y_test = []
        for i in range (60, inputs.shape[0]):
            X_test.append(inputs[i-60:i])
            Y_test.append(inputs[i, 0])

        X_test, Y_test = np.array(X_test), np.array(Y_test)
        Y_pred = trained_model.predict(X_test)
        scale = 1/5.18164146e-05
        Y_test = Y_test*scale
        Y_pred = Y_pred*scale
        plt.figure(figsize=(14,5))
        plt.plot(Y_test, color = 'red', label = 'Real Doigcoin Price')
        plt.plot(Y_pred, color = 'green', label = 'Predicted Dogcoin Price')
        plt.title('Dogcoin Price Prediction using LSTM')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        st.pyplot()

    col1, col2 = st.beta_columns(2)
    with col1:
        df = pd.DataFrame({'tether': data3.Close})
        st.line_chart(data = df)
    with col2:
        """data3 = investpy.get_crypto_historical_data(crypto='tether',
                                               from_date='01/01/2021',
                                               to_date='31/07/2021')"""

        training_data = data3.drop(['Currency'], axis = 1)

        #MinMaxScaler is used to normalize the data
        scaler = MinMaxScaler()
        training_data = scaler.fit_transform(training_data)

        X_train = []
        Y_train = []
        training_data.shape[0]
        for i in range(60, training_data.shape[0]):
                X_train.append(training_data[i-60:i])
                Y_train.append(training_data[i,0])

        X_train, Y_train = np.array(X_train), np.array(Y_train)

        trained_model = tf.keras.models.load_model('team_A_model')

        part_60_days = data3.tail(60)
        df= part_60_days.append(data3, ignore_index = True)
        df = df.drop(['Currency'], axis = 1)
        inputs = scaler.transform(df)

        X_test = []
        Y_test = []
        for i in range (60, inputs.shape[0]):
            X_test.append(inputs[i-60:i])
            Y_test.append(inputs[i, 0])

        X_test, Y_test = np.array(X_test), np.array(Y_test)
        Y_pred = trained_model.predict(X_test)

        scale = 1/5.18164146e-05
        Y_test = Y_test*scale
        Y_pred = Y_pred*scale

        plt.figure(figsize=(14,5))
        plt.plot(Y_test, color = 'red', label = 'Real Tether USDT Price')
        plt.plot(Y_pred, color = 'green', label = 'Predicted Tether USDT Price')
        plt.title('Tether USDT Price Prediction using LSTM')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        st.write("**Our model predictions for tether usdt for future investment**")
        st.pyplot()

