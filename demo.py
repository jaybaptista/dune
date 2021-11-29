import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import astropy.units as u
import pandas as pd
from dune import mock

if ('N' not in st.session_state):
    st.session_state['N'] = 10

if ('rp' not in st.session_state):
    st.session_state['rp'] = .5

if ('a' not in st.session_state):
    st.session_state['a'] = .5

if ('mass' not in st.session_state):
    st.session_state['mass'] = 500

# @st.cache(allow_output_mutation=True)

st.title('DUNE Demo')

st.session_state.N = st.slider(
    'Number of Stars', min_value=10, max_value=1000, value=None, step=5)
st.session_state.rp = st.slider(
    'Plummer Radius [kpc]', min_value=.1, max_value=5., value=None, step=.1)
st.session_state.a = st.slider(
    'Scale Radius [kpc]', min_value=.1, max_value=5., value=None, step=.1)
st.session_state.mass = st.slider(
    'Mass [Msun]', min_value=100., max_value=1000., value=None, step=100.)

st.session_state.type = st.selectbox('DM Profile', ('NFW', 'Hernquist'))

rho = (6.4e7 * u.M_sun / (u.kpc**3))

data = None

if st.session_state.type == 'NFW':
    data = mock.GeneratePlummerNFW(st.session_state.N, st.session_state.rp *
                                   u.kpc, st.session_state.a * u.kpc, rho, st.session_state.mass * u.M_sun)
elif st.session_state.type == 'Hernquist':
    data = mock.GeneratePlummerHQ(st.session_state.N, st.session_state.rp *
                                   u.kpc, st.session_state.a * u.kpc, st.session_state.mass * u.M_sun)


df = pd.DataFrame(data)

@st.cache
def cvt_df(df):
    return df.to_csv().encode('utf-8')

csv_file = cvt_df(df)

if data is not None:
    st.download_button('Download', csv_file,
                       file_name=f'dune_{st.session_state.type}_{st.session_state.N}_{st.session_state.a}kpc_{st.session_state.rp}kpc_{st.session_state.mass}Msun.csv')

# fig, ax = plt.subplots()
# ax.scatter(data['x'], data['y'], c=data['vt'], s=.1)
# st.pyplot(fig)

fig = px.scatter_3d(df, x='x', y='y', z='z', color='vt')
fig.update_traces(marker_size=5)
st.plotly_chart(fig)
