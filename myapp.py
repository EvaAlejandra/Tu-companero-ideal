import streamlit as st
import pandas as pd
from logica import inquilinos_compatibles
from ayudantes import generar_grafico_compatibilidad, generar_tabla_compatibilidad, obtener_id_inquilinos
import base64

import plotly.express as px
import plotly.graph_objects as go

#Funcion para seleccionar imagene de carpeta
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

#Función para definir el fondo con una imagen
def set_background(png_file, container_selector):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    {container_selector} {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: Local;
    }}
    
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True
    
    
if 'clicked2' not in st.session_state:
    st.session_state.clicked2 = False

def click_button2():
    st.session_state.clicked2 = True
    
st.set_page_config(layout="wide")
    
# Rutas de las imagenes real de tu imagen
image_path_fondo = "./Media/Tucompañeroideal2.png"
image_path_sidebar = "./Media/fondo.png"

# Establece la imagen como fondo del contenedor principal
set_background(image_path_fondo, '[data-testid="stAppViewContainer"] > .main')

# Establece la imagen como fondo del sidebar
#set_background(image_path_sidebar, '[data-testid="stSidebar"] > div:first-child')


#Donde guardaremos el resulado
resultado = None

# Mostrar una gran imagen en la parte superior.
st.title("  ")

with st.sidebar:
    st.header("¿Quién está viviendo ya en el piso?")
    
    # Configurar el sidebar con inputs y un botón.
    inquilinos = st.container()

    col1,col2,col3 = st.columns([1,1,1])

    # Configurar el sidebar con inputs y un botón.
    with inquilinos:
        inquilino1 = col1.text_input("Inquilino 1")
        inquilino2 = col2.text_input("Inquilino 2")
        inquilino3 = col3.text_input("Inquilino 3")


    st.button('Click to add more Inquilinos', on_click=click_button)

    if st.session_state.clicked:
        # The message and nested widget will remain on the page
        number_new_inquilinos = st.number_input("How many inquilinos do you want to add?", step=1, value=1)
        
        if number_new_inquilinos > 0:
            # Crear una lista con el número de columnas
            num_columns = [1] * number_new_inquilinos

            # Usar st.columns con el número de columnas especificado por la lista
            columns = st.columns(num_columns)

            # Mostrar el text_input en cada columna
            for i, column in enumerate(columns):
                # Crear un text_input en cada columna
                inquilino_input = column.text_input(f"Inquilino {i+4}")
    num_compañeros = st.text_input("¿Cuántos nuevos compañeros quieres que le recomiende?")
    st.button('Enviar', on_click=click_button2)
    
    
if st.session_state.clicked2:
    # Verifica que el número de compañeros sea un valor válido
    try:
        topn = int(num_compañeros)
    except ValueError:
        st.error("Por favor, ingresa un número válido para el número de compañeros.")
        topn = None
    
    # Obtener los identificadores de inquilinos utilizando la función
    id_inquilinos = obtener_id_inquilinos(inquilino1, inquilino2, inquilino3, topn)
    data = pd.DataFrame(id_inquilinos)
    
    #st.markdown("#### Inquilinos buscados")
    #Creamos la tabla de los inquilinos residentes
    #fig = go.Figure( data = go.Table(
    #    columnwidth=[1,3],
    #    
    #    header = dict( values = list(['Id'] ) , #nombre de las columnas
    #                  fill_color = ('#333333'), #Relleno de la casilla, para cambiar tipos de rosa cambiar el hexadecimal
    #                  align = 'center', #disposición del texto
    #                  font=dict(color='white')), #Color de la letra
    #    
    #    cells = dict( values = [data],fill_color = ('#FFFFFF'),
    #                  align = 'center')
    #        ))
    #
    #fig.update_layout( margin = dict( l=5, r=5, b=10, t=10), #margenes
    #                   height=75,  # altura de la figura
    #                   width=75,  # ancho de la figura
    #                   plot_bgcolor='rgba(0, 0, 0, 0)',  # fondo transparente
    #                   paper_bgcolor='rgba(0, 0, 0, 0)'  # color de fondo del papel transparente
    #                   #paper_bgcolor = backgroundcolor, #para poner el color del background
    #                  )
    #st.write(fig)
    
    st.header("Resultado:")
    
    if id_inquilinos and topn is not None:
        # Llama a la función inquilinos_compatibles con los parámetros correspondientes
        resultado = inquilinos_compatibles(id_inquilinos, topn)


# Verificar si 'resultado' contiene un mensaje de error (cadena de texto)
if isinstance(resultado, str):
    st.error(resultado)
# Si no, y si 'resultado' no es None, mostrar el gráfico de barras y la tabla
elif resultado is not None:
    cols = st.columns((1, 2))  # Divide el layout en 2 columnas
    
    with cols[0]:  # Esto hace que el gráfico y su título aparezcan en la primera columna
        st.write("Nivel de compatibilidad de cada nuevo compañero:")
        fig_grafico = generar_grafico_compatibilidad(resultado[1])
        st.pyplot(fig_grafico)
    
    with cols[1]:  # Esto hace que la tabla y su título aparezcan en la segunda columna
        st.write("Comparativa entre compañeros:")
        fig_tabla = generar_tabla_compatibilidad(resultado)
        st.plotly_chart(fig_tabla, use_container_width=True)

