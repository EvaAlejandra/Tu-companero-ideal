import streamlit as st
import pandas as pd
#from logica import inquilinos_compatibles
#from ayudantes import generar_grafico_compatibilidad, generar_tabla_compatibilidad, obtener_id_inquilinos
import base64
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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










# 2. CARGA DE DATOS
df = pd.read_csv('dataset_inquilinos.csv', index_col = 'id_inquilino')

df.columns = [
'horario', 'bioritmo', 'nivel_educativo', 'leer', 'animacion', 
'cine', 'mascotas', 'cocinar', 'deporte', 'dieta', 'fumador',
'visitas', 'orden', 'musica_tipo', 'musica_alta', 'plan_perfecto', 'instrumento'
]

# 3. ONE HOT ENCODING
# Realizar el one-hot encoding
encoder = OneHotEncoder()
df_encoded = encoder.fit_transform(df).toarray()

# Obtener los nombres de las variables codificadas después de realizar el one-hot encoding
encoded_feature_names = encoder.get_feature_names_out()

# 4. MATRIZ DE SIMILIARIDAD
# Calcular la matriz de similaridad utilizando el punto producto
matriz_s = np.dot(df_encoded, df_encoded.T)

# Define el rango de destino
rango_min = -100
rango_max = 100

# Encontrar el mínimo y máximo valor en matriz_s
min_original = np.min(matriz_s)
max_original = np.max(matriz_s)

# Reescalar la matriz
matriz_s_reescalada = ((matriz_s - min_original) / (max_original - min_original)) * (rango_max - rango_min) + rango_min

# Pasar a Pandas
df_similaridad = pd.DataFrame(matriz_s_reescalada,
        index = df.index,
        columns = df.index)


# 5. BÚSQUEDA DE INQUILINOS COMPATIBLES


def inquilinos_compatibles(id_inquilinos, topn):
    # Verificar si todos los ID de inquilinos existen en la matriz de similaridad
    for id_inquilino in id_inquilinos:
        if id_inquilino not in df_similaridad.index:
            return 'Al menos uno de los inquilinos no encontrado'

    # Obtener las filas correspondientes a los inquilinos dados
    filas_inquilinos = df_similaridad.loc[id_inquilinos]

    # Calcular la similitud promedio entre los inquilinos
    similitud_promedio = filas_inquilinos.mean(axis=0)

    # Ordenar los inquilinos en función de su similitud promedio
    inquilinos_similares = similitud_promedio.sort_values(ascending=False)

    # Excluir los inquilinos de referencia (los que están en la lista)
    inquilinos_similares = inquilinos_similares.drop(id_inquilinos)

    # Tomar los topn inquilinos más similares
    topn_inquilinos = inquilinos_similares.head(topn)

    # Obtener los registros de los inquilinos similares
    registros_similares = df.loc[topn_inquilinos.index]

    # Obtener los registros de los inquilinos buscados
    registros_buscados = df.loc[id_inquilinos]

    # Concatenar los registros buscados con los registros similares en las columnas
    resultado = pd.concat([registros_buscados.T, registros_similares.T], axis=1)

    # Crear un objeto Series con la similitud de los inquilinos similares encontrados
    similitud_series = pd.Series(data=topn_inquilinos.values, index=topn_inquilinos.index, name='Similitud')

    # Devolver el resultado y el objeto Series
    return(resultado, similitud_series)









# FUNCIÓN PARA GENERAR EL GRÁFICO DE COMPATIBILIDAD
def generar_grafico_compatibilidad(compatibilidad):
    compatibilidad = compatibilidad / 100  # Asegúrate de que esté en escala de 0 a 1 para porcentajes
    
    # Configuramos el gráfico de Seaborn
    fig, ax = plt.subplots(figsize=(3, 2))  # Ajusta el tamaño del gráfico según tus necesidades
    
    # Crea el gráfico de barras con los valores convertidos a porcentajes
    sns.barplot(x=compatibilidad.index, y=compatibilidad.values, ax=ax, color='lightblue', edgecolor=None)
    
    # Quitar bordes
    sns.despine(top=True, right=True, left=True, bottom=False)
    
    # Configurar las etiquetas de los ejes y rotar las etiquetas del eje x
    ax.set_xlabel('Identificador de Inquilino', fontsize=10)
    ax.set_ylabel('Similitud (%)', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    # Ajustar las etiquetas del eje y para mostrar porcentajes correctamente
    ax.set_yticklabels(['{:.1f}%'.format(y * 100) for y in ax.get_yticks()], fontsize=8)

    # Añadir etiquetas de porcentaje sobre cada barra
    for p in ax.patches:
        height = p.get_height()
        ax.annotate('{:.1f}%'.format(height * 100),
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points', fontsize=8)

    return(fig)


# FUNCIÓN PARA GENERAR LA TABLA DE COMPAÑEROS
def generar_tabla_compatibilidad(resultado):
    # Cambiar el nombre de la columna 'index' y ajustar el ancho de las columnas
    resultado_0_with_index = resultado[0].reset_index()
    resultado_0_with_index.rename(columns={'index': 'ATRIBUTO'}, inplace=True)
    
    # Configurar la tabla de Plotly
    fig_table = go.Figure(data=[go.Table(
        columnwidth = [20] + [10] * (len(resultado_0_with_index.columns) - 1),  # Ajustar el primer valor para el ancho de la columna 'ATRIBUTO'
        header=dict(values=list(resultado_0_with_index.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[resultado_0_with_index[col] for col in resultado_0_with_index.columns],
                   fill_color='white',
                   align='left'))
    ])
    
    # Configurar el layout de la tabla de Plotly
    fig_table.update_layout(
        width=700, height=320,  # Ajustar según tus necesidades
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return(fig_table)


#FUNCIÓN PARA GENERAR LA LISTA DE INQUILINOS SEMILLA
def obtener_id_inquilinos(inquilino1, inquilino2, inquilino3, topn):
    # Crea una lista con los identificadores de inquilinos ingresados y los convierte a enteros
    id_inquilinos = []
    for inquilino in [inquilino1, inquilino2, inquilino3]:
        try:
            if inquilino:  # Si hay algún texto en el input
                id_inquilinos.append(int(inquilino))  # Convierte a entero y agrega a la lista
        except ValueError:
            st.error(f"El identificador del inquilino '{inquilino}' no es un número válido.")
            id_inquilinos = []  # Vaciar la lista si hay un error
            break  # Salir del bucle

    return(id_inquilinos)






















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

