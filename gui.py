from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import pandas as pd
import pickle as pkl
from columns_types import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, UNPROCESSED_COLUMNS, ORDENED_COLUMNS

MODELS = (
    'Regresión lineal',
    'Support Vector Regression',
    'K-Nearest Neighbors'
)

st.set_page_config(page_title="Predicción precios de Portatiles",
                   page_icon=":computer:", layout="wide")

####################
df = pd.read_csv("notebooks_cleaned_v2.csv")
original_df = pd.read_csv("notebook_data.tsv", sep="\t")

scaler: MinMaxScaler = pkl.load(open("minmax_scaler.pkl", 'rb'))
labels = pkl.load(open("label_encoders_snds.pkl", "rb"))

lr_model: LinearRegression = pkl.load(
    open("linear_regression_model.pkl", "rb"))
####################


def min_max(column):
    return original_df[column].agg(['min', 'max'])


def create_slider(container: DeltaGenerator, label: str, range_val: tuple[int, int], value: int, step: int | None = None):
    return container.slider(label, range_val[0], range_val[1], value, step=step)[0]


class PC():
    def __init__(self):
        self.weight = original_df['weight'].min(axis=0),
        self.os = labels["operating_system_family_name"].classes_[0],
        self.screen_size = original_df['screen_size_value'].min(axis=0),
        self.screen_resolution = labels["screen_resolution_unicode"].classes_[
            0],
        self.screen_rr = original_df['screen_rr_value'].min(axis=0),
        self.battery = original_df['battery_mwh'].min(axis=0),
        self.processor_freq = original_df['processor_f_value'].min(axis=0),
        self.processor_cores = original_df['processor_p_core_count'].min(
            axis=0),
        self.processor_threads = original_df['processor_thread_count'].min(
            axis=0),
        self.ram = original_df['ram_quantity_value'].min(axis=0),
        self.ram_freq = original_df['ram_frequency_value'].min(axis=0),
        self.ram_type_name = labels["ram_type_name"].classes_[0],
        self.storage_capacity = original_df['sd_capacity_value'].min(axis=0),
        self.sd_drive_type_name = labels["sd_drive_type_name"].classes_[0]


tab1, tab2 = st.tabs(["Contexto", "Predicción"])

tab1.title("Contexto del proyecto")
tab1.markdown(
    '''La venta de hardware por internet es un negocio muy competitivo, eso hace que surgan páginas que recopilan precios de venta de productos de varias empresas y comparen sus valores, SoloTodo es una de ellas, esta página se especializa en brindar al usuario una serie de herramientas para que busque la alternativa que más le acomode.

El actual proyecto recopila datos públicos de equipos portátiles (notebooks) utilizando la API ofrecida por SoloTodo generando un conjunto de datos que contiene todas las portátiles registradas en el sitio web, con algunas especificaciones técnicas, como también puntajes en el rendimiento del procesador, tarjeta gráfica y evaluación de rendimiento en videojuegos, aplicaciones en general y movilidad, etc.

El objetivo de esta página es ofrecer una herramienta que permita predecir el precio de una portátil en base a sus especificaciones técnicas, para que el usuario pueda tener una idea del valor de un equipo antes de comprarlo.'''
)
tab1.image("https://www.solotodo.cl/logo_fondo_oscuro.svg")

pc = PC()
tab2.title("Modelos predictivos")

page_col1, page_col2 = tab2.columns(2)

attr_container = page_col1.container(border=True)
results_container = page_col2.container(border=True)

col1, col2 = attr_container.columns(2)

screen_attr = col1.container()
screen_attr.subheader("Atributos de pantalla")

pc.screen_size = create_slider(screen_attr, "Tamaño de pantalla (pulgadas)", min_max(
    'screen_size_value'), pc.screen_size, 0.1)

pc.screen_resolution = screen_attr.selectbox(
    "Resolución de pantalla", labels["screen_resolution_unicode"].classes_)

pc.screen_rr = screen_attr.selectbox(
    "Tasa de refresco de pantalla", sorted(original_df['screen_rr_value'].unique()))


cpu_attr = col1.container()
cpu_attr.subheader("Atributos de CPU")

pc.processor_freq = create_slider(cpu_attr, "Frecuencia del procesador (MHz)", min_max(
    'processor_f_value'), pc.processor_freq, 10)

pc.processor_cores = create_slider(cpu_attr, "Núcleos del procesador", min_max(
    'processor_p_core_count'), pc.processor_cores)
pc.processor_threads = create_slider(cpu_attr, "Hilos del procesador", min_max(
    'processor_thread_count'), pc.processor_threads, 2)

ram_attr = col2.container()
ram_attr.subheader("Atributos de RAM")

pc.ram = ram_attr.selectbox("Cantidad de RAM (GB)",
                            sorted(original_df['ram_quantity_value'].unique()))

pc.ram_freq = create_slider(ram_attr, 'Frecuencia de la RAM (MHz)', min_max(
    'ram_frequency_value'), pc.ram_freq)

pc.ram_type_name = ram_attr.selectbox(
    "Tipo de RAM", labels["ram_type_name"].classes_)


other_attr = col2.container()
other_attr.subheader("Otros atributos")

pc.weight = create_slider(other_attr, "Peso (g)", min_max('weight'), pc.weight)

pc.os = other_attr.selectbox(
    "Sistema operativo", labels["operating_system_family_name"].classes_)

pc.battery = create_slider(other_attr, "Batería (MWh)",
                           min_max('battery_mwh'), pc.battery, 100)

pc.storage_capacity = create_slider(other_attr, "Capacidad de almacenamiento (GB)", min_max(
    'sd_capacity_value'), pc.storage_capacity, 32)

pc.sd_drive_type_name = other_attr.selectbox(
    "Tipo de disco duro", labels["sd_drive_type_name"].classes_)


model = results_container.selectbox("Seleccione un modelo predictivo", MODELS)


def linear_regression(pc: PC):
    test = pd.DataFrame({
        "weight": [pc.weight],
        "operating_system_family_name": labels["operating_system_family_name"].transform([pc.os])[0],
        "screen_size_value": [pc.screen_size],
        "screen_resolution_unicode": labels["screen_resolution_unicode"].transform([pc.screen_resolution])[0],
        "screen_rr_value": [pc.screen_rr],
        "battery_mwh": [pc.battery],
        "processor_f_value": [pc.processor_freq],
        "processor_p_core_count": [pc.processor_cores],
        "processor_thread_count": [pc.processor_threads],
        "ram_quantity_value": [pc.ram],
        "ram_frequency_value": [pc.ram_freq],
        "ram_type_name": labels["ram_type_name"].transform([pc.ram_type_name])[0],
        "sd_capacity_value": [pc.storage_capacity],
        "sd_drive_type_name": labels["sd_drive_type_name"].transform([pc.sd_drive_type_name])[0]
    })

    # test_numerical = test[NUMERICAL_COLUMNS]

    test_numerical = pd.DataFrame(scaler.transform(test[NUMERICAL_COLUMNS]))
    test_numerical.columns = NUMERICAL_COLUMNS

    test_categorical = test[CATEGORICAL_COLUMNS]

    test = pd.concat([test_numerical, test_categorical, test[UNPROCESSED_COLUMNS]],
                     axis=1).rename(str, axis='columns')[ORDENED_COLUMNS]
    # st.write(test[ORDENED_COLUMNS])

    prediction = lr_model.predict(test)[0][0]

    results_container.markdown(
        f"## Predicción: :red-background[$ {'{:20,.2f}'.format(prediction if prediction > 0 else 0)} CLP]")

    results_container.markdown(
        f"#### Precisión: :red-background[{'{:5.4f}'.format(0.7493259578443761*100)} %]")


match model:
    case "Regresión lineal":
        linear_regression(pc)
    case _:
        st.write('Opcion desconocida')

# if model == "Regresión lineal":
#     linear_regression(pc)

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)
