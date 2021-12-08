import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots

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

st.title('DUNE Demo')

st.session_state.type = st.selectbox('DM Profile', ('NFW', 'Hernquist'))

st.session_state.N = st.slider(
    'Number of Stars', min_value=10, max_value=1000, value=None, step=5)
st.session_state.rp = st.slider(
    'Plummer Radius [kpc] (NFW only)', min_value=.1, max_value=5., value=None, step=.1)
st.session_state.a = st.slider(
    'Scale Radius (kpc in NFW, pc in Hernquist)', min_value=.1, max_value=50., value=None, step=.1)
st.session_state.mass = st.slider(
    'Mass [Msun]', min_value=100., max_value=1000., value=None, step=100.)

rho = (6.4e7 * u.M_sun / (u.kpc**3))

data = None
potential = None
force = None

if st.session_state.type == 'NFW':
    mock_output = mock.GeneratePlummerNFW(st.session_state.N, st.session_state.rp *
                                   u.kpc, st.session_state.a * u.kpc, rho=rho, mass=st.session_state.mass * u.M_sun)
    data = mock_output[0]
    potential = mock_output[1]
    force = mock_output[2]

elif st.session_state.type == 'Hernquist':
    mock_output = mock.GenerateHernquist(st.session_state.N, st.session_state.a * u.pc, st.session_state.mass * u.M_sun)
    data = mock_output[0]
    potential = mock_output[1]
    force = mock_output[2]


df = pd.DataFrame(data)

@st.cache
def cvt_df(df):
    return df.to_csv().encode('utf-8')

csv_file = cvt_df(df)

if data is not None:
    st.download_button('Download', csv_file,
                       file_name=f'dune_{st.session_state.type}_{st.session_state.N}_{st.session_state.a}kpc_{st.session_state.rp}kpc_{st.session_state.mass}Msun.csv')


fig = px.scatter_3d(df, x='x', y='y', z='z', color='vr', title='Stellar Population with Radial Velocities')
fig.update_traces(marker_size=5)
fig.update_layout(coloraxis_colorbar=dict(title="V_r [km/s]"))
st.plotly_chart(fig)

fig_analytics, axs = plt.subplots(3, 2, figsize=(9,14))

axs[0,0].hist(df.vd, histtype='step', color='k')
axs[1,0].hist(df.vt, histtype='step', color='k')
axs[2,0].hist(df.vr, histtype='step', color='k')

axs[0,1].scatter(df.r, df.vd, c='k', alpha=.2)

axs[1,1].scatter(data['r'], force(data['r']))
axs[2,1].scatter(data['r'], potential(data['r']))

axs[0,1].set_xscale('log')
axs[1,1].set_xscale('log')
axs[2,1].set_xscale('log')

axs[0,1].set_xlabel('r [kpc]')
axs[1,1].set_xlabel('r [kpc]')
axs[2,1].set_xlabel('r [kpc]')

axs[0,0].set_xlabel('$\\sigma$ [km/s]')
axs[1,0].set_xlabel('$V_R$ [km/s]')
axs[2,0].set_xlabel('$V_T$ [km/s]')

axs[0,1].set_ylabel('$\\sigma$ [km/s]')
axs[1,1].set_ylabel('Specific Force [$m/s^2$]')
axs[2,1].set_ylabel('Specific Energy [$m^2/s^2$]')

axs[0,0].axvline(np.mean(df.vd), c='r')

axs[1,0].axvline(0, c='k', alpha=.5)
axs[2,0].axvline(0, c='k', alpha=.5)

axs[1,0].axvline(np.mean(df.vr), c='r')
axs[2,0].axvline(np.mean(df.vt), c='r')

axs[1,0].axvline(np.mean(df.vr)+np.std(df.vr), c='k', alpha=.3, ls='--')
axs[1,0].axvline(np.mean(df.vr)-np.std(df.vr), c='k', alpha=.3, ls='--')

axs[2,0].axvline(np.mean(df.vt)+np.std(df.vt), c='k', alpha=.3, ls='--')
axs[2,0].axvline(np.mean(df.vt)-np.std(df.vt), c='k', alpha=.3, ls='--')

axs[0,0].text(.025, 1.05, 'Velocity dispersion', transform=axs[0,0].transAxes)
axs[0,1].text(.025, 1.05, 'Radial Velocity', transform=axs[0,1].transAxes)
axs[1,0].text(.025, 1.05, 'Radial velocity', transform=axs[1,0].transAxes)
axs[1,1].text(.025, 1.05, 'Force Function', transform=axs[1,1].transAxes)
axs[2,0].text(.025, 1.05, 'Tangential velocity', transform=axs[2,0].transAxes)
axs[2,1].text(.025, 1.05, 'Potential Function', transform=axs[2,1].transAxes)

fig_analytics.tight_layout()
st.pyplot(fig_analytics)