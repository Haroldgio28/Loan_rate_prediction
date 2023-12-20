import streamlit as st
import pandas as pd
import requests
from PIL import Image


st.markdown("""
    <style>
    .reportview-container {
        background-color: #fafafa;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
    h1 {
        color: #30475e;
    }
    h2 {
        color: #30475e;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #30475e;
        text-align: center;
        padding: 2px;
        font-size: 12px;
        line-height: 1; /* Reducido el interlineado */
    }
    .footer a {
        color: #30475e;
        text-decoration: none;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

logo_path = "app\logo.png"  # Reemplaza con la ruta de tu logo
logo = Image.open(logo_path)

with st.container():
    col1, col2 = st.columns([1, 2])

    # Mostrar el logo en una columna
    with col1:
        st.image(logo, width=150)  # Puedes ajustar el ancho según tus necesidades

    # Mostrar el nombre del banco en la otra columna
    with col2:
        st.markdown("# TrustEdge Bank")
        st.markdown("### Your Reliable Partner in Finance")


# URL del servicio Flask
url = "http://localhost:9696/predict"

# Función para realizar la solicitud HTTP POST al servicio Flask
def predict_interest_rate(input_data):
    # amount = cust['loan_amount']
    response = requests.post(url, json=input_data)
    # print(f'The customer {customer_id} will receive a loan of {amount} with a rate {response}%')

    # response = requests.post(URL, json=input_data)
    return response.json()

# Título de la aplicación
st.title('Loan Interest Rate Prediction')

# Opción para cambiar el customer_id (opcional)
customer_id = st.text_input("Customer ID", value="Pepito Perez")

# Interfaz de usuario para cargar un archivo CSV
uploaded_file = st.file_uploader("Upload your input CSV file", type="csv")

if uploaded_file is not None:
    # Lee los datos del archivo CSV
    data = pd.read_csv(uploaded_file)

    # Transforma los datos en un diccionario
    input_data = data.to_dict(orient='records')

    # Botón para realizar predicciones
    if st.button('Predict Interest Rate'):
        try:
            # Procesa cada fila del archivo CSV
            for row in input_data:
                amount = row.get("loan_amount", "Unknown")
                response = predict_interest_rate(row)
                # Asegúrate de que 'response' se maneje correctamente
                interest_rate = response['rate'] if 'rate' in response else "Error"
                respuesta = round(response['rate'],2)
                st.write(f"The customer {customer_id} will receive a loan of ${amount} with a rate {respuesta}%")
        except Exception as e:
            st.error(f"Error al obtener la predicción: {e}")
else:
    st.write("Please upload a CSV file to get predictions.")
# Ejecutar la aplicación: streamlit run your_script.py
    
footer_html = f"""
    <div class="footer">
        <p>Developed by Harold - Visit my <a href="https://github.com/Haroldgio28/Loan_rate_prediction" target="_blank">GitHub Repository</a></p>
        <p>Connect with me on <a href="https://www.linkedin.com/in/haroldgiovannyuribe" target="_blank">LinkedIn</a> and <a href="https://x.com/HaroldGio28" target="_blank">X</a></p>
    </div>
    """
st.markdown(footer_html, unsafe_allow_html=True)
